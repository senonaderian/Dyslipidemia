import pandas as pd

# Load your dataset
data = pd.read_csv('result.csv')

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')
