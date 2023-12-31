import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu

# Set Pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format  # Format float display to 4 decimal places

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

# Perform Mann-Whitney U test and compare medians
results = []

for feature in selected_features['selected_features']:
    group_1 = X[y == 1][feature]
    group_2 = X[y == 2][feature]

    # Perform Mann-Whitney U test
    stat, p = mannwhitneyu(group_1, group_2, alternative='two-sided')

    # Determine the direction of the difference in medians
    if group_1.median() > group_2.median():
        direction = "Class 1 > Class 2"
    elif group_2.median() > group_1.median():
        direction = "Class 2 > Class 1"
    else:
        direction = "No difference in medians"

    results.append([feature, stat, p, direction])

# Create a DataFrame to store the results
results_df = pd.DataFrame(results, columns=['Feature', 'U Statistic', 'p-value', 'Direction'])

# Print the results
print("Results:")
print(results_df)
