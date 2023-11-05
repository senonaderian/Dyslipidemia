import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import mannwhitneyu  # Import the Mann-Whitney U test

# Read the CSV file into a DataFrame
df = pd.read_csv('2.csv')

# Extract the 'dyslipd' column and store it in a separate variable
target = df['dyslipd']

# Select all columns except for the 'dyslipd' column
data = df.drop('dyslipd', axis=1)

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

# Perform Mann-Whitney U test and compare medians for selected features
u_statistics = []
p_values = []
medians = []
feature_directions = []

for feature_name in selected_features_f_classif:
    feature = X[feature_name]
    class_1_data = feature[target == 1]
    class_2_data = feature[target == 2]

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(class_1_data, class_2_data, alternative='two-sided')

    # Determine the direction based on medians
    median_class_1 = np.median(class_1_data)
    median_class_2 = np.median(class_2_data)

    if median_class_1 > median_class_2:
        direction = "Class 1 > Class 2"
    else:
        direction = "Class 2 > Class 1"

    # Store the results
    u_statistics.append(u_statistic)
    p_values.append(p_value)
    medians.append((median_class_1, median_class_2))
    feature_directions.append(direction)

# Print the results for the selected features
feature_significance_df = pd.DataFrame({
    'Feature': selected_features_f_classif,
    'U Statistic': u_statistics,
    'p-value': p_values,
    'Direction': feature_directions
})
print("Feature Significance for Selected Features (Mann-Whitney U test and median comparison):")
pd.options.display.float_format = '{:.5f}'.format  # Set the display format for p-values
print(feature_significance_df)

# Calculate class weights
class_counts = np.bincount(y)
class_weights = {i: len(y) / (2 * class_counts[i]) if class_counts[i] != 0 else 1e-6 for i in range(len(class_counts))}

# Train an SVM classifier with class weighting
clf = SVC(random_state=42, class_weight=class_weights)

# Fit the SVM classifier on the entire dataset
clf.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Calculate the confusion matrix
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp) if tn + fp != 0 else np.nan

# Print the accuracy and F1 score using cross-validation
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, clf.predict(X), average="weighted"))
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)
