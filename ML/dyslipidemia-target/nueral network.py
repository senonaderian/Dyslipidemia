import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import tensorflow as tf
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

# Compute class weights
class_counts = np.bincount(y)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * np.where(class_counts != 0, class_counts, 1e-6))

# Perform Mann-Whitney U test and compare medians for selected features
u_statistics = []
p_values = []
feature_directions = []

for feature_name in selected_features_f_classif:
    feature = X[feature_name]
    class_1_data = feature[target == 1]
    class_2_data = feature[target == 2]

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(class_1_data, class_2_data, alternative='two-sided')

    # Determine the direction based on medians
    if np.median(class_1_data) > np.median(class_2_data):
        direction = "Class 1 > Class 2"
    else:
        direction = "Class 2 > Class 1"

    # Store the results
    u_statistics.append(u_statistic)
    p_values.append(p_value)
    feature_directions.append(direction)

# Print the results for the selected features with formatted p-values
feature_significance_df = pd.DataFrame(
    {'Feature': selected_features_f_classif, 'U Statistic': u_statistics, 'p-value': p_values, 'Direction': feature_directions})
print("Feature Significance for Selected Features:")
pd.options.display.float_format = '{:.5f}'.format  # Set the display format for p-values
print(feature_significance_df)

# Define the custom loss function with class weighting
def weighted_loss(y_true, y_pred):
    weights = class_weights[y_true]
    loss = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    weighted_loss = tf.reduce_mean(loss * weights)
    return weighted_loss

# Create a Multi-Layer Perceptron (Neural Network) classifier
mlp = MLPClassifier(random_state=42, max_iter=900, learning_rate_init=0.001, hidden_layer_sizes=(100, 50))

# Set the custom loss function
mlp.loss_ = weighted_loss

# Train the MLP classifier on the features
mlp.fit(X, y)

# Evaluate the classifier
y_pred = mlp.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = f1_score(y, y_pred, average="weighted")
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp) if tn + fp != 0 else np.nan

print("Accuracy:", accuracy)
print("F1 score:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
