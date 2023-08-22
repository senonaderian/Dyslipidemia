import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
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
selected_features_df.to_csv('selected_features1.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features1.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {}
unique_classes = np.unique(y)
for cls in unique_classes:
    class_weights[cls] = len(y) / (2 * (y == cls).sum())

# Train a decision tree classifier with class weighting
clf = DecisionTreeClassifier(criterion="entropy", random_state=42, class_weight=class_weights)

# Train the decision tree classifier on the selected features
clf.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, clf.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf.predict(X)
labels = unique_classes
cm = confusion_matrix(y, y_pred, labels=labels)

print("Confusion matrix shape:", cm.shape)

if cm.shape == (len(labels), len(labels)):
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
    else:
        classwise_sensitivity = []
        classwise_specificity = []
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            classwise_sensitivity.append(sensitivity)
            classwise_specificity.append(specificity)
            print(f"Class {label} - Sensitivity: {sensitivity}, Specificity: {specificity}")
        print("Average sensitivity:", np.mean(classwise_sensitivity))
        print("Average specificity:", np.mean(classwise_specificity))
else:
    print("Insufficient values in the confusion matrix.")

# Plot the decision tree
plt.figure(figsize=(30, 20))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(cls) for cls in labels])
plt.savefig("decision_tree.png", dpi=300)
plt.show()

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
