import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import mannwhitneyu

# Read the CSV file into a DataFrame
df = pd.read_csv('3.csv')

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
selected_features_df.to_csv('selected_features2.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features2.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Train a KNN classifier with class weighting
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Train the KNN classifier on the selected features
knn.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(knn, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, knn.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = knn.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# Calculate feature importances using permutation importance
result = permutation_importance(knn, X, y, n_repeats=10, random_state=42, scoring='accuracy')
importances = result.importances_mean
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20, 20))
plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances (Permutation Importance)')
plt.xticks(rotation='vertical')
plt.show()

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)


# Perform Mann-Whitney U test and compare medians
u_statistic_list = []
p_value_list = []
direction_list = []


for feature in selected_features['selected_features']:
    group_1 = X[y == 1][feature]
    group_2 = X[y == 2][feature]
    stat, p_value = mannwhitneyu(group_1, group_2)
    direction = "Class 1 > Class 2" if np.median(group_1) > np.median(group_2) else "Class 2 > Class 1"
    u_statistic_list.append(stat)
    p_value_list.append(round(p_value, 4))  # Round p-value to 4 digits
    direction_list.append(direction)


mannwhitney_results = pd.DataFrame({
    "Feature": selected_features['selected_features'],
    "U Statistic": u_statistic_list,
    "p-value": p_value_list,
    "Direction": direction_list
})

print("Mann-Whitney U Test Results:")
print(mannwhitney_results)
