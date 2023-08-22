import pandas as pd
import numpy as np
np.random.seed(42)  # Set a specific random seed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler

# Read the CSV file into a DataFrame
df = pd.read_csv('2.csv')

# Extract the 'cholrotbe' column and store it in a separate variable
target = df['cholrotbe']

# Select all columns except for the 'cholrotbe' column
data = df.drop('cholrotbe', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using chi2 score function and fit to data
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector_mutual_info = SelectKBest(score_func=mutual_info_classif, k=10)
X_mutual_info = selector_mutual_info.fit_transform(data_non_negative, target)

# Print selected features using mutual information score function (Commented out)
print("Selected Features (using mutual information score function):")
selected_features_mutual_info = data.columns[selector_mutual_info.get_support()].tolist()
print(selected_features_mutual_info)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_mutual_info})

# Save the selected features to a file
selected_features_df.to_csv('selected_features4.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features4.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Convert target variable to numerical labels
labels = np.unique(y)
label_mapping = {label: i for i, label in enumerate(labels)}
y_numeric = np.array([label_mapping[label] for label in y])

# Oversample the minority classes
ros = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = ros.fit_resample(X, y_numeric)

# Create a Multi-Layer Perceptron (Neural Network) classifier
clf = MLPClassifier(random_state=42, max_iter=2000, learning_rate="constant", learning_rate_init=0.0009, hidden_layer_sizes=(100, 100), activation='relu')

# Fit the classifier with the oversampled data
clf.fit(X_oversampled, y_oversampled)

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf, X_oversampled, y_oversampled, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())

# Predict the oversampled data
y_pred_oversampled = clf.predict(X_oversampled)

# Calculate the F1 score
f1 = f1_score(y_oversampled, y_pred_oversampled, average="weighted")
print("F1 score:", f1)

# Calculate the confusion matrix
cm = confusion_matrix(y_oversampled, y_pred_oversampled, labels=labels)
print("Confusion matrix shape:", cm.shape)

if cm.shape == (len(labels), len(labels)):
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if tp + fn != 0 else np.nan
        specificity = tn / (tn + fp) if tn + fp != 0 else np.nan
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
            sensitivity = tp / (tp + fn) if tp + fn != 0 else np.nan
            specificity = tn / (tn + fp) if tn + fp != 0 else np.nan
            classwise_sensitivity.append(sensitivity)
            classwise_specificity.append(specificity)
            print(f"Class {label} - Sensitivity: {sensitivity}, Specificity: {specificity}")
        print("Average sensitivity:", np.nanmean(classwise_sensitivity))
        print("Average specificity:", np.nanmean(classwise_specificity))
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

