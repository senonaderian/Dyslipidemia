import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer

# Read the CSV file into a DataFrame
df = pd.read_csv('3.csv')

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
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
selector_f_classif = SelectKBest(score_func=f_classif, k=10)
X_f_classif = selector_f_classif.fit_transform(data_non_negative, target)

# Print selected features using F-score score function
print("Selected Features (using F-score score function):")
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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = {}
unique_classes = np.unique(y)
for cls in unique_classes:
    class_weights[cls] = len(y) / (2 * (y == cls).sum())

# Train a K-Nearest Neighbors classifier with class weighting
clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Train the KNN classifier on the training set
clf.fit(X_train, y_train)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, clf.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf.predict(X)
labels = np.unique(y)
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
