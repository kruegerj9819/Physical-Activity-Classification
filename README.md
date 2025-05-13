# Physical Activity Classification using Random Forest and Bagging Classifiers
This project aims to classify physical activity levels using two machine learning models: **Random Forest Classifier** and **Bagging Classifier**. The dataset used for this classification task contains labeled physical activity types, including walking, cycling, running, and more.

## Dataset
The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/diegosilvadefrana/fisical-activity-dataset).

## Activity Labels
The activities classified in the dataset include:
- Nordic walking
- Ascending stairs
- Cycling
- Descending stairs
- Ironing
- Lying
- Rope jumping
- Running
- Sitting
- Standing
- Transient activities (short-term activities)
- Vacuum cleaning
- Walking

## Installation
To run this project, clone this repository and install the necessary dependencies. You can use the following command to install the required libraries:
```
pip install -r requirements.txt
```

### Requirements
- pandas
- matplotlib
- scikit-learn

## How to Run
1. Clone this repository or download the necessary files.
2. Ensure that the dataset `clean_physical_activity.csv` is available in the same directory as the Python script.
3. Run the Python script `Classification.py` to train and evaluate the models.

```
python Classification.py
```

The script will:
- Load the dataset and preprocess the data.
- Split the data into training and test sets.
- Train a **Random Forest Classifier** and a **Bagging Classifier** using hyperparameter tuning.
- Evaluate the models using Out-of-Bag (OOB) score and confusion matrices.

## Results
The script will output:
- The best hyperparameters found for both classifiers.
- The OOB score for the Random Forest and Bagging classifiers.
- Confusion matrices for both classifiers to evaluate their performance on the test set.

The **Random Forest Classifier** will also display a bar chart representing the feature importances, which helps in understanding which features contributed the most to the predictions.

### Example Output
```
Best max_depth: 15
OOB Score (Random Forest): 0.88
Best n_estimators: 6
OOB Score (Bagging Classifier): 0.83
```

### Confusion Matrices
- The confusion matrix for the Random Forest Classifier will display normalized values for each activity label.
- The confusion matrix for the Bagging Classifier will also display normalized values for each activity label.

## Interpretation
### Importances Chart
![physical-activity-importances](https://github.com/user-attachments/assets/00c7c05b-cb8d-4ffc-a1fa-474d8a0d0a4f)

### Random Forest Confusion Matrix
![random-forest-confusion-matrix](https://github.com/user-attachments/assets/30fb660b-c801-4553-be78-b6d952738c4d)

### Bagging Classifier Confusion Matrix
![bagging-classifier-confusion-matrix](https://github.com/user-attachments/assets/0bb560cb-534a-46de-b5ed-4fc337387d6d)

Looking at the resulting confusion matrices, it seems that the Random Forest has a much higher precision than the Bagging Classifier on this dataset. The Bagging Classifier seems to have a lot of false positives which affect the overall accuracy of the model when looking at the diagnal. However, both models seem to have a hard time differentiating transient activities (short-term activities) from a lot of the other categories.
