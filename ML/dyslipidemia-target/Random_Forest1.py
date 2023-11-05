import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2  # Import SelectKBest and chi2
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

# Use the Mann-Whitney U test to assess feature significance
# Initialize empty lists for U statistics, p-values, and feature directions
u_statistics = []
p_values = []
feature_directions = []

# Loop through the selected features (change 'selected_features' to the actual list of feature names)
selected_features = ['Female Waist Circumference', 'serumvitD', 'serumvitDrotb', 'metabolicsyndrome', 'total blood pressure', 'copper', 'chromium', 'atocopherol', 'suger', 'mayesaier']

for feature_name in selected_features:
    feature = data_non_negative[:, data.columns.get_loc(feature_name)]
    class_1_data = feature[target == 1]
    class_2_data = feature[target == 2]

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(class_1_data, class_2_data, alternative='two-sided')

    # Determine the direction based on medians
    if np.median(class_1_data) > np.median(class_2_data):
        direction = "Class 1 > Class 2"
    else:
        direction = "Class 2 > Class 1"

    # Format p-value to display up to 5 decimal places
    formatted_p_value = f'{p_value:.5f}'

    # Store the results
    u_statistics.append(u_statistic)
    p_values.append(formatted_p_value)
    feature_directions.append(direction)

# Print the results for the selected features
feature_significance_df = pd.DataFrame(
    {'Feature': selected_features, 'U Statistic': u_statistics, 'p-value': p_values, 'Direction': feature_directions})
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
selected_features_df.to_csv('selected_features5.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features5.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Train a random forest classifier with class weighting
clf = RandomForestClassifier(criterion="entropy", random_state=42, class_weight=class_weights)

# Train the random forest classifier on the selected features
clf.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy (mutual_info_classif):", scores.mean())
print("F1 score (mutual_info_classif):", f1_score(y, clf.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity (mutual_info_classif):", sensitivity)
    print("Specificity (mutual_info_classif):", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# Plot feature importances
importances = clf.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20, 20))
plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation='vertical')
plt.show()

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)


