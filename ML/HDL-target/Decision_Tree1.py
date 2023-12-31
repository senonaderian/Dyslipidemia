import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu

# Set Pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format  # Format float display to 4 decimal places


# Read the CSV file into a DataFrame
df = pd.read_csv('2.csv')

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
from sklearn.feature_selection import SelectKBest, chi2
selector_chi2 = SelectKBest(chi2, k=10)
X_chi2 = selector_chi2.fit_transform(data_non_negative, target)

# Print selected features using chi2 score function
print("Selected Features (using chi-square score function):")
selected_features_chi2 = data.columns[selector_chi2.get_support()].tolist()
print(selected_features_chi2)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_chi2})

# Save the selected features to a file
selected_features_df.to_csv('selected_features1.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features1.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train a decision tree classifier with hyperparameter tuning
clf = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
clf_optimized = grid_search.best_estimator_

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf_optimized, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy (mutual_info_classif):", scores.mean())
print("F1 score (mutual_info_classif):", f1_score(y, clf_optimized.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf_optimized.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity (mutual_info_classif):", sensitivity)
    print("Specificity (mutual_info_classif):", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# Plot the decision tree
plt.figure(figsize=(30, 20))
# The 'feature_names' parameter should be a list
plot_tree(clf_optimized, filled=True, feature_names=list(X.columns), class_names=["1", "2"])
plt.savefig("decision_tree.png", dpi=200)
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
