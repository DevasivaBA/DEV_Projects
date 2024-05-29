# DEV_Projects
# Telco Customer Churn Prediction

## Overview

This project aims to predict customer churn for a telecommunications company using machine learning techniques.  Churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company. By accurately predicting which customers are likely to churn, the company can proactively implement retention strategies to reduce churn rates and improve profitability.

## Dataset

The project uses the "Telco Customer Churn" dataset available on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains information about customers of a telecom company, including demographic data, services subscribed to, account information, and whether or not the customer churned.

## Project Structure

- `churn_prediction.py`: The main Python script containing the code for data preprocessing, model training, hyperparameter tuning, evaluation, and feature importance analysis.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The original dataset file.
- `requirements.txt`: Lists the required Python libraries for running the code.
- `Initial_Model_roc_curve.json`: Interactive plot of the ROC curve for the initial model (Logistic Regression).
- `Best_Model_roc_curve.json`: Interactive plot of the ROC curve for the best model (Logistic Regression with tuned hyperparameters).
- `feature_importance.json`: Interactive plot showcasing the importance of each feature in the best model.


## Approach

The project follows these main steps:

1. **Data Loading and Preprocessing:**
   - Load the dataset into a Pandas DataFrame.
   - Handle missing values (in `TotalCharges`) by imputing with the mean.
   - Convert relevant features (e.g., `SeniorCitizen`) to categorical data types.
   - Perform one-hot encoding of categorical features.
   - Split the data into training and testing sets.

2. **Model Training and Evaluation (Logistic Regression):**
   - Train an initial logistic regression model on the training set.
   - Evaluate the model on the test set using precision, recall, F1 score, accuracy, and AUC-ROC.
   - Visualize the ROC curve to assess the model's ability to discriminate between classes.

3. **Hyperparameter Tuning (Logistic Regression):**
   - Use GridSearchCV to find the optimal hyperparameters for the logistic regression model, including `C`, `penalty`, `solver`, and `class_weight`.

4. **Retraining and Final Evaluation (Logistic Regression):**
   - Retrain the logistic regression model using the best hyperparameters found in the previous step.
   - Evaluate the model's performance on the test set.
   - Analyze feature importance to identify the most significant predictors of churn.

## Results

The best logistic regression model achieved the following performance on the test set:
 
| Metric       | Score |
| ----------- | ----- |
| Precision   | 0.62  |
| Recall      | 0.51  |
| F1 Score    | 0.56  |
| Accuracy    | 0.79  |
| AUC-ROC     | 0.83  |

*The specific results may vary slightly depending on the random state of the data split.*

## Future Work

- **Explore More Features:** Investigate the potential impact of creating interaction features or binning continuous features.
- **Try Different Algorithms:** Experiment with other classification algorithms like XGBoost, Random Forests, or Support Vector Machines to see if they can achieve better performance.
- **Cost-Sensitive Learning:** If the cost of false positives and false negatives is different, explore cost-sensitive learning approaches to optimize the model for business objectives.

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/](https://github.com/)<DevasivaBA>/<DEV_Projects>.git
