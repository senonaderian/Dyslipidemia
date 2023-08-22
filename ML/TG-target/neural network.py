import pandas as pd
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.impute import SimpleImputer

# Read the CSV file into a DataFrame
df = pd.read_csv('1.csv')

# Extract the 'TGrotbe' column and store it in a separate variable
target = df['TGrotbe']

# Select all columns except for the 'TGrotbe' column
data = df.drop('TGrotbe', axis=1)

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
selected_features_df.to_csv('selected_features4.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features4.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class frequencies
class_counts = y.value_counts()

# Define the desired class imbalance ratio (adjust as needed)
imbalance_ratio = {
    1: 1.0,  # Keep the majority class as is
    2: 2.0,  # Duplicate the minority class samples
    3: 2.0,  # Duplicate the minority class samples
    4: 2.0,  # Duplicate the minority class samples
}

# Resample the data to achieve class imbalance
resampled_data = []
resampled_target = []
for label, count in class_counts.items():
    resampled_data.append(resample(X[y == label], n_samples=int(count * imbalance_ratio[label]), replace=True))
    resampled_target.extend([label] * int(count * imbalance_ratio[label]))
resampled_data = pd.concat(resampled_data)

# Create a Multi-Layer Perceptron (Neural Network) classifier
clf = MLPClassifier(random_state=42, max_iter=1000, learning_rate="constant", learning_rate_init=0.01, hidden_layer_sizes=(100, 100), activation='relu')

# Train the Neural Network classifier on the resampled data
clf.fit(resampled_data, resampled_target)

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