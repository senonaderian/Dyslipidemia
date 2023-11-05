import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu  # Import the Mann-Whitney U test
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer

# Read the CSV file into a DataFrame
df = pd.read_csv('1.csv')

# Extract the 'dyslipd' column and store it in a separate variable
target = df['dyslipd']

# Select all columns except for the 'dyslipd' column
data = df.drop('dyslipd', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using chi2 score function and fit to data
selector_chi2 = SelectKBest(chi2, k=10)
X_chi2 = selector_chi2.fit_transform(data_non_negative, target)

# Print selected features using chi2 score function
print("Selected Features (using chi-square score function):")
selected_features_chi2 = data.columns[selector_chi2.get_support()].tolist()
print(selected_features_chi2)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_chi2})

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

# Perform Mann-Whitney U test and compare medians for selected features
u_statistics = []
p_values = []
feature_directions = []

for feature_name in selected_features_chi2:
    feature = X[feature_name]
    class_1_data = feature[target == 1]
    class_2_data = feature[target == 2]

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(class_1_data, class_2_data, alternative='two-sided')

    # Determine the direction based on medians
    if np.median(class_1_data) > np.median(class_2_data):
        direction = "Class 1 > Class 2"
    else:
        direction = "Class 2 > Class 1"

    # Store the results
    u_statistics.append(u_statistic)
    p_values.append(round(p_value, 4))  # Round p-value to 4 digits
    feature_directions.append(direction)

# Print the results for the selected features
feature_significance_df = pd.DataFrame(
    {'Feature': selected_features_chi2, 'U Statistic': u_statistics, 'p-value': p_values, 'Direction': feature_directions})
print("Feature Significance for Selected Features:")
print(feature_significance_df)

# Create a SelectKBest object using chi2 score function and fit to data
selector_chi2 = SelectKBest(chi2, k=10)
X_chi2 = selector_chi2.fit_transform(data_non_negative, target)

# Print selected features using chi2 score function
print("Selected Features (using chi-square score function):")
selected_features_chi2 = data.columns[selector_chi2.get_support()].tolist()
print(selected_features_chi2)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_chi2})

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

# Plot feature importances (not applicable for Naive Bayes)

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)

