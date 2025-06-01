# Project README

## Overview
This project implements a recommendation system using Graph Neural Networks (GNNs) to provide personalized business recommendations based on user preferences and geographical location.

## Directory Structure

### Model/
This directory contains the core components of the recommendation system.

- **Data_Presprocessing/**: Contains scripts for loading, filtering, and preprocessing data from the Yelp dataset.
  - `Data_Preprocessing.ipynb`: A Jupyter notebook that filters and preprocesses business, review, and user data, saving the results in Parquet format.

- **Evaluation/**: Contains scripts for evaluating the performance of the recommendation models.
  - `test_recomend.ipynb`: A Jupyter notebook that tests the recommendation system using a Temporal Graph Network (TGN).
  - `train_evaluate-hgn.py`: A Python script that trains and evaluates a Hierarchical Graph Network (HGN) model.
  - `train_evaluate-tgn.py`: A Python script that trains and evaluates a Temporal Graph Network (TGN) model.

- **Prediction/**: Contains scripts for generating recommendations using the trained models.
  - `hgn_recommendation.py`: A script that generates recommendations using the HGN model.
  - `tgn_recommendation.py`: A script that generates recommendations using the TGN model.

- **Saved_Models/**: This directory is used to store the trained models for later use.

- **Training/**: Contains scripts for training the models. Note: The `hgn_recommendation.ipynb` and `tgn_recommendation.ipynb` files are incorrectly placed in this directory and should be moved to the "Prediction" directory.

### GUI.py
This is the main entry point for running the graphical user interface of the recommendation system.

## Instructions for Use

1. **Generating Preprocessed Data**:
   - Run the `Data_Preprocessing.ipynb` notebook located in the `Model/Data_Presprocessing/` directory to preprocess the Yelp dataset. This will create Parquet files containing filtered business, review, and user data.

2. **Training the Model**:
   - Use either `hgn_recommendation.ipynb` or `tgn_recommendation.ipynb` in the `Model/Training/` directory to train the respective models. These scripts will also evaluate the models and save the best weights.

3. **Generating Recommendations**:
   - Use `hgn_recommendation.py` or `tgn_recommendation.py` in the `Model/Prediction/` directory to generate recommendations for users. You can specify user ID, location, and other parameters to get personalized recommendations.

4. **Running the GUI**:
   - Execute `GUI.py` to launch the graphical user interface for the recommendation system. Follow the on-screen instructions to interact with the system.

## Model Training
The models are trained using the preprocessed data to learn user preferences and business characteristics. The training process involves:
- Loading the preprocessed data and creating training, validation, and test splits.
- Training the model on the training set while monitoring performance on the validation set.
- Using metrics such as loss and RMSE to evaluate the model's performance during training.
- Saving the best-performing model weights for later use.

## Model Evaluation
The models are evaluated using metrics such as Root Mean Square Error (RMSE) to assess their performance in predicting user ratings for businesses. The evaluation process involves:
- Testing the model on the test set to compute RMSE and other ranking metrics.
- Logging the results to track the model's performance over time.

This evaluation is crucial for ensuring that the recommendation system provides accurate and relevant suggestions to users.

## GUI Overview
The graphical user interface (GUI) of the recommendation system serves as the primary interaction point for users. It is designed to provide an intuitive and user-friendly experience, allowing users to easily access the functionalities of the recommendation system.

### GUI Features
- **User Input**: The GUI allows users to input their preferences, such as user ID and geographical location, which are essential for generating personalized recommendations.
- **Recommendation Display**: Once the user inputs their preferences, the GUI displays a list of recommended businesses tailored to the user's tastes and location. This feature leverages the underlying Graph Neural Network models to provide accurate and relevant suggestions.
- **Visual Feedback**: The GUI provides visual feedback, such as loading indicators and success messages, to enhance user experience and ensure that users are informed about the status of their requests.

### Relation to the Program
The GUI acts as a bridge between the user and the recommendation system's backend. It simplifies the interaction with complex algorithms and data processing, making it accessible to users without technical expertise. By utilizing the GUI, users can seamlessly generate recommendations without needing to understand the underlying processes.

### GUI Image
Below is an image of the GUI, showcasing its layout and features:

![GUI Image](GUI_Image.png)

## Conclusion
This project provides a comprehensive framework for building a recommendation system using graph neural networks. The modular structure allows for easy updates and improvements to the data processing, training, and prediction components.
