import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer

# Read the CSV file into a DataFrame
df = pd.read_csv('2.csv')

# Extract the 'LDLrotbe' column and store it in a separate variable
target = df['LDLrotbe']

# Select all columns except for the 'LDLrotbe' column
data = df.drop('LDLrotbe', axis=1)

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
selected_features_df.to_csv('selected_features3.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features3.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Train a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Perform grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_clf = grid_search.best_estimator_

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the classifier on the test set
y_pred = best_clf.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average="weighted")
print("F1 score:", f1)

# Calculate the confusion matrix
labels = np.unique(y)
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion matrix shape:", cm.shape)

if cm.shape == (len(labels), len(labels)):
    classwise_sensitivity = []
    classwise_specificity = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
            classwise_sensitivity.append(sensitivity)
            print(f"Class {label} - Sensitivity: {sensitivity}")
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
            classwise_specificity.append(specificity)
            print(f"Class {label} - Specificity: {specificity}")
    print("Average sensitivity:", np.mean(classwise_sensitivity))
    print("Average specificity:", np.mean(classwise_specificity))
else:
    print("Insufficient values in the confusion matrix.")

from sklearn.feature_selection import f_classif

# Calculate ANOVA F-statistic and p-values for each feature and the target (multi-class)
f_statistic, p_values = f_classif(X, target)

# Create a DataFrame to store ANOVA F-statistic and p-values
anova_df = pd.DataFrame({'Feature': selected_features['selected_features'],
                         'F-Statistic': f_statistic,
                         'P-Value': p_values})

# Sort features by F-statistic magnitude
anova_df = anova_df.sort_values(by='F-Statistic', ascending=False)

# Print the sorted ANOVA F-statistics
print("ANOVA F-Statistics:")
print(anova_df)
