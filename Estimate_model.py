# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:03:49 2023

@author: SSpatacean
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_test = pd.read_excel(r'..\Testing_data.xlsm')  
data_train = pd.read_excel(r'..\Training_data.xlsm')

# Function to check for missing values and duplicates in the data
def check_missing_and_duplicates(data):

    # Data preprocessing
    duplicate_rows = data[data.duplicated(keep='first')]
    data = data.drop_duplicates()  # Remove duplicate rows
    data = data.dropna(axis=0)    # Remove rows with missing values
    
    return data
    

# Function to remove outliers using the IQR method
def remove_outliers_iqr(data):

    # Specify the numerical features for outlier removal
    numerical_features = ['Mortgage', 'Card Utilization', 'Card Balance', 'Card Balance_3m', 'Card Balance_6m', 'Card Balance_12m', 
                          'Amount Past Due', 'Delinquency Status', 'Delinquency Status_3m', 'Delinquency Status_6m', 'Delinquency Status_12m',
                          'Credit Inquiry', 'Credit Inquiry_3m', 'Credit Inquiry_6m', 'Credit Inquiry_12m', 'Open Trade', 'Open Trade_3m', 
                          'Open Trade_6m', 'Open Trade_12m', 'DDA Balance_9m']

    for feature in numerical_features:
        Q1 = data[feature].quantile(0.02)
        Q3 = data[feature].quantile(0.98)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
    return data

# Function to create binary dummies for race categories
def create_race_dummies(data):
    # Get the unique values in the "Race_Category" variable
    unique_race_categories = data['Race_Category'].unique()
    
    # Create binary variables for each unique value
    for category in unique_race_categories:
        data[f'Race_{category}'] = (data['Race_Category'] == category).astype(int)
    
    return data

# Function to transform specified columns using log1p
def transform_log1p_rename(data, columns_to_transform):
    for col in columns_to_transform:
        new_col_name = col + '_LOG1P'
        data[new_col_name] = np.log1p(data[col])
    data = data.drop(columns=columns_to_transform)  # Optionally, you can drop the original columns
    return data

# Function to transform specified columns using log1p
def transform_log1p(data, columns_to_transform):
    data[columns_to_transform] = np.log1p(data[columns_to_transform])
    return data


# Drop missing and duplicate values from testing and training data
data_test = check_missing_and_duplicates(data_test)
data_train = check_missing_and_duplicates(data_train)

# Remove outliers from the dataset
data_train = remove_outliers_iqr(data_train)
data_test = remove_outliers_iqr(data_test)

# Create binary dummies for race categories
data_train = create_race_dummies(data_train)
data_test = create_race_dummies(data_test)

# Specify columns for log1p transformation
columns_to_transform = ['Mortgage', 'Card Utilization', 'Card Balance', 'Card Balance_3m', 'Card Balance_6m',
                        'Card Balance_12m', 'Amount Past Due', 'Delinquency Status', 'Delinquency Status_3m',
                        'Delinquency Status_6m', 'Delinquency Status_12m', 'Credit Inquiry', 'Credit Inquiry_3m',
                        'Credit Inquiry_6m', 'Credit Inquiry_12m', 'Open Trade', 'Open Trade_3m', 'Open Trade_6m',
                        'Open Trade_12m', 'DDA Balance_9m']

# Apply log1p transformation to specified columns
data_train = transform_log1p(data_train, columns_to_transform)
data_test = transform_log1p(data_test, columns_to_transform)

### Estimating the Model ###

# All vars
model1_vars = ['Mortgage', 'Card Utilization', 'Card Balance', 'Card Balance_3m', 'Card Balance_6m', 'Card Balance_12m',
               'Amount Past Due', 'Delinquency Status', 'Delinquency Status_3m', 'Delinquency Status_6m',
               'Delinquency Status_12m', 'Credit Inquiry', 'Credit Inquiry_3m', 'Credit Inquiry_6m',
               'Credit Inquiry_12m', 'Open Trade', 'Open Trade_3m', 'Open Trade_6m', 'Open Trade_12m',
               'DDA Balance_9m', 'Race_White', 'Gender', 'Race_Black', 'Race_Hispanic', 'Race_Asian']

# No race discrim
model2_vars = ['Mortgage', 'Card Utilization', 'Card Balance', 'Card Balance_3m', 'Card Balance_6m', 'Card Balance_12m',
               'Amount Past Due', 'Delinquency Status', 'Delinquency Status_3m', 'Delinquency Status_6m',
               'Delinquency Status_12m', 'Credit Inquiry', 'Credit Inquiry_3m', 'Credit Inquiry_6m',
               'Credit Inquiry_12m', 'Open Trade', 'Open Trade_3m', 'Open Trade_6m', 'Open Trade_12m',
               'DDA Balance_9m']

# Rid of correlated vars using significance values
model3_vars = ['Mortgage', 'Card Utilization', 'Card Balance',
               'Amount Past Due', 'Delinquency Status_3m', 'Credit Inquiry_6m',
               'Open Trade_6m','DDA Balance_9m', 'Gender']

# Function to prepare data for modeling
def prep_modeling_data(data_train, model_vars):
    # Separate features and target variable
    X = data_train[model_vars]  # Select the features for the model
    y = data_train['Status']   # Select the target variable

    # Add a constant term for the intercept in the logistic regression model
    X = sm.add_constant(X)

    return y, X  # Return the target variable and the prepared feature matrix


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    total_predictions = len(y_true)

    for i in range(total_predictions):
        if y_true.iloc[i] == y_pred.iloc[i]:  # Assuming y_true and y_pred are Pandas Series
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


# Function to check group outcomes based on a specific column
def check_group_outcomes(data, y_pred, column_name):
    y_pred.rename('y_pred_int', inplace=True)
    df = data.merge(y_pred, left_index=True, right_index=True)

    # Proportions predicted for each group within the specified column
    proportions = df.groupby(column_name)['y_pred_int'].mean()
    return proportions


# Prepare modeling data for training and testing
y_train, X_train = prep_modeling_data(data_train, model3_vars)
y_test, X_test = prep_modeling_data(data_test, model3_vars)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary())

# Make predictions on the training and testing data
y_train_pred = result.predict(X_train)
y_test_pred = result.predict(X_test)

# Convert predicted probabilities to binary outcomes using a threshold of 0.5
y_train_pred_int = (y_train_pred >= 0.5).astype(int)
y_test_pred_int = (y_test_pred >= 0.5).astype(int)

# Calculate and print the accuracy for both training and testing data
accuracy_test = calculate_accuracy(y_test, y_test_pred_int)
accuracy_train = calculate_accuracy(y_train, y_train_pred_int)


#Check proportions of predictions by race 
#calcs % of people getting 1 by race on predictions
proportions_test = check_group_outcomes(data_test, y_test_pred_int, 'Race_Category')
proportions_train = check_group_outcomes(data_train, y_train_pred_int, 'Race_Category' )

proportions_test = check_group_outcomes(data_test, y_test_pred_int, 'Gender')
proportions_train = check_group_outcomes(data_train, y_train_pred_int, 'Gender' )


#Check proportions of actuals by race
#Interesting to check the difference in the proportion of test and train 
#predictions vs the proportions in the actual data. Proporitions in acutals below.
#% of people getting 1 in actuals ie. status.
proportions = data_test.groupby('Race_Category')['Status'].mean()
proportions = data_train.groupby('Race_Category')['Status'].mean()

proportions = data_test.groupby('Gender')['Status'].mean()
proportions = data_train.groupby('Gender')['Status'].mean()

####  Decision Tree Model   ####

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Predict the target values on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")







