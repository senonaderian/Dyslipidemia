import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
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

# Create a SelectKBest object using f_classif score function and fit to data
selector_f_classif = SelectKBest(f_classif, k=10)
X_f_classif = selector_f_classif.fit_transform(data_non_negative, target)

# Print selected features using f_classif score function
print("Selected Features (using f_classif score function):")
selected_features_f_classif = data.columns[selector_f_classif.get_support()].tolist()
print(selected_features_f_classif)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_f_classif})

# Save the selected features to a file
selected_features_df.to_csv('selected_features6.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features6.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_counts = np.bincount(y)
class_weights = {i: len(y) / (2 * class_counts[i]) if class_counts[i] != 0 else 1e-6 for i in range(len(class_counts))}

# Train an SVM classifier with class weighting
clf = SVC(random_state=42, class_weight=class_weights)

# Fit the SVM classifier on the entire dataset
clf.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the accuracy and F1 score using cross-validation
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, clf.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp) if tn + fp != 0 else np.nan

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

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
