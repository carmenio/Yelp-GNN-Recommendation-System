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

## Conclusion
This project provides a comprehensive framework for building a recommendation system using graph neural networks. The modular structure allows for easy updates and improvements to the data processing, training, and prediction components.
