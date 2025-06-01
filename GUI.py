#!/usr/bin/env python3
import sys
import os
import json
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QTextEdit, QFormLayout
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl

# Ensure project root is on PATH
sys.path.append(os.path.abspath(os.getcwd()))
# Import the recommend function and DataLoader to fetch users
from Model.Prediction.tgn_recommendation import recommend, load_and_preprocess, MyTGN
from Model.Prediction.hgn_recommendation import recommend as hgn_recommend
from Main import DataLoader
import torch

# HTML template with Leaflet map and Qt WebChannel integration
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Map</title>
<meta name="viewport" content="initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
<style>#map{height:100vh;width:100%;margin:0;padding:0;}</style>
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map').setView([39.955505, -75.155564], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);

// Add circle overlay that always fills the screen
var circle = L.circle(map.getCenter(), { radius: 500, color: 'red', weight: 2 }).addTo(map);

// Function to update circle and notify Python
function updateCircleAndUI() {
    var center = map.getCenter();
    var bounds = map.getBounds();
    var north = L.latLng(bounds.getNorth(), center.lng);
    var east = L.latLng(center.lat, bounds.getEast());
    var radius = Math.min(map.distance(center, north), map.distance(center, east));
    circle.setLatLng(center);
    circle.setRadius(radius);
    if (window.bridge && window.bridge.onMapMoved) {
        window.bridge.onMapMoved(center.lat, center.lng, radius);
    }
}

// Update on moveend and zoomend
map.on('moveend zoomend', updateCircleAndUI);

// Initial update
updateCircleAndUI();

// Setup Qt WebChannel bridge
new QWebChannel(qt.webChannelTransport, function(channel){
    window.bridge = channel.objects.bridge;
});

// Emit clicks on the map
map.on('click', function(e){
    window.bridge.onMapClicked(e.latlng.lat, e.latlng.lng);
});

// Callable from Python to add markers
function addMarker(id, lat, lng, info){
    var marker = L.marker([lat, lng]).addTo(map);
    marker.on('click', function(){
        window.bridge.onMarkerClicked(id, lat, lng, info);
    });
}
</script>
</body>
</html>
"""

class Bridge(QObject):
    mapClicked = pyqtSignal(float, float)
    markerClicked = pyqtSignal(str, float, float, str)
    mapMoved = pyqtSignal(float, float, float)


    @pyqtSlot(float, float, float)
    def onMapMoved(self, lat, lng, radius):
        self.mapMoved.emit(lat, lng, radius)

    @pyqtSlot(float, float)
    def onMapClicked(self, lat, lng):
        self.mapClicked.emit(lat, lng)

    @pyqtSlot(str, float, float, str)
    def onMarkerClicked(self, id, lat, lng, info):
        self.markerClicked.emit(id, lat, lng, info)

class MapWidget(QWebEngineView):
    def __init__(self):
        super().__init__()
        # Load the map HTML
        self.page().setHtml(HTML_TEMPLATE, QUrl("about:blank"))
        # Setup the channel and bridge
        self.channel = QWebChannel()
        self.bridge = Bridge()
        self.channel.registerObject('bridge', self.bridge)
        self.page().setWebChannel(self.channel)

    def add_marker(self, id, lat, lng, info):
        # JS call to add a marker with popup info
        js = f"addMarker({json.dumps(id)}, {lat}, {lng}, {json.dumps(info)});"
        self.page().runJavaScript(js)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restaurant Recommender GUI")
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)


        # Load recommendation mappings, business DataFrame, and features
        (self.source, self.dest, self.edge_attr, self.norm_ts, self.user_df, self.biz_df, self.user_to_index, self.business_to_index, self.user_feats, self.biz_feats, self.avg_rating) = load_and_preprocess()
        # Load pretrained TGN model
        total_nodes = len(self.user_to_index) + len(self.business_to_index)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MyTGN(total_nodes, self.user_feats.size(1), self.biz_feats.size(1))
        self.model.load_state_dict(torch.load("tgn_model.pt", map_location=device))
        self.model.to(device)
        self.model.eval()
        
        
        # Left sidebar: user search, parameters, and info panel
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("User Search"))
        self.user_combo = QComboBox()
        self.user_combo.setEditable(True)
        left_layout.addWidget(self.user_combo)

        # Load users into combo box
        loader = DataLoader()
        
        user_ids = list(self.user_df['user_id'][0:10])
        self.user_combo.addItems(user_ids)

        left_layout.addWidget(QLabel("Radius (km)"))
        self.radius_edit = QLineEdit("500")
        left_layout.addWidget(self.radius_edit)

        left_layout.addWidget(QLabel("Hour (0-23)"))
        current_hour = datetime.datetime.now().hour
        self.hour_edit = QLineEdit(str(current_hour))
        left_layout.addWidget(self.hour_edit)

        left_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TGN", "HGN"])
        left_layout.addWidget(self.model_combo)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)
        left_layout.addWidget(self.search_btn)

        left_layout.addWidget(QLabel("Details / Info"))
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        left_layout.addWidget(self.info)

        main_layout.addLayout(left_layout, 1)

        # Right panel: coordinate display and map
        right_layout = QVBoxLayout()
        form = QFormLayout()
        self.lng_edit = QLineEdit()
        self.lat_edit = QLineEdit()
        form.addRow("Longitude", self.lng_edit)
        form.addRow("Latitude", self.lat_edit)
        right_layout.addLayout(form)

        self.map_widget = MapWidget()
        # Connect clicks and map movement updates
        self.map_widget.bridge.mapClicked.connect(self.update_coords)
        self.map_widget.bridge.markerClicked.connect(self.show_marker_info)
        self.map_widget.bridge.mapMoved.connect(self.update_map_metrics)
        right_layout.addWidget(self.map_widget, 1)

        main_layout.addLayout(right_layout, 4)

    def update_coords(self, lat, lng):
        """Update coordinate fields when the map is clicked."""
        self.lat_edit.setText(f"{lat:.6f}")
        self.lng_edit.setText(f"{lng:.6f}")

    def show_marker_info(self, id, lat, lng, info):
        """Display restaurant info when a marker is clicked."""
        self.info.setPlainText(info)
        # Also update fields
        self.lat_edit.setText(f"{lat:.6f}")
        self.lng_edit.setText(f"{lng:.6f}")

    @pyqtSlot(float, float, float)
    def update_map_metrics(self, lat, lng, radius):
        """Update coordinate and radius fields when the map is moved or zoomed."""
        self.lat_edit.setText(f"{lat:.6f}")
        self.lng_edit.setText(f"{lng:.6f}")
        # radius is in meters, convert to kilometers
        self.radius_edit.setText(f"{radius / 1000:.3f}")

    def on_search(self):
        """Call recommend() and populate the map with results."""
        uid = self.user_combo.currentText()
        try:
            lat = float(self.lat_edit.text())
            lng = float(self.lng_edit.text())
        except ValueError:
            lat, lng = 0.0, 0.0
        try:
            radius = float(self.radius_edit.text())
        except ValueError:
            radius = 500.0
        try:
            hour = int(self.hour_edit.text())
        except ValueError:
            hour = datetime.datetime.now().hour

        # Call the recommend function
        model_sel = self.model_combo.currentText()
        if model_sel == "TGN":
            recs = recommend(self.model, uid, lat, lng, radius, hour, self.user_to_index, self.business_to_index, self.biz_df, avg_rating=self.avg_rating, user_feats=self.user_feats, biz_feats=self.biz_feats)
        else:
            day = datetime.datetime.now().strftime("%A")
            recs = hgn_recommend(uid, (lat, lng), radius, day, hour, top_k=10)
        # Clear existing markers
        clear_js = """
        map.eachLayer(function(layer){
            if(layer instanceof L.Marker) { map.removeLayer(layer); }
        });
        """
        self.map_widget.page().runJavaScript(clear_js)
        # Add each recommendation as a marker
        print('Recs:')
        for rec in recs:
            if model_sel == "TGN":
                bid, score = rec
                print("  ", bid, score)
                row = self.biz_df[self.biz_df['business_id']==bid].iloc[0]
                info = f"{row['name']}\n{row['address']}\nRating: {row['stars']}\nHours: {json.dumps(row['hours'])}"
                lat2, lng2 = row['latitude'], row['longitude']
            else:
                bid = rec['business_id']
                score = rec['predicted_rating']
                print("  ", bid, score)
                info = f"{rec['name']}\nPredicted: {score:.2f}"
                lat2, lng2 = rec['latitude'], rec['longitude']
            self.map_widget.add_marker(bid, lat2, lng2, info)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
