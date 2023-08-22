import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer

# Read the CSV file into a DataFrame
df = pd.read_csv('1.csv')

# Extract the 'HDLlow' column and store it in a separate variable
target = df['HDLlow']

# Select all columns except for the 'HDLlow' column
data = df.drop('HDLlow', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using mutual_info_classif score function and fit to data
selector_mi = SelectKBest(mutual_info_classif, k=10)
X_mi = selector_mi.fit_transform(data_non_negative, target)

# Print selected features using mutual_info_classif score function
print("Selected Features (using mutual information score function):")
selected_features_mi = data.columns[selector_mi.get_support()].tolist()
print(selected_features_mi)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_mi})

# Save the selected features to a file
selected_features_df.to_csv('selected_features3.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features3.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Normalize class weights to sum up to 1
class_weight_sum = sum(class_weights.values())
class_priors = {k: v / class_weight_sum for k, v in class_weights.items()}

# Train a Gaussian Naive Bayes classifier with class priors
gnb = GaussianNB(priors=list(class_priors.values()))

# Train the Naive Bayes classifier on the selected features
gnb.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(gnb, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, gnb.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = gnb.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
else:
    print("Insufficient values in the confusion matrix.")
