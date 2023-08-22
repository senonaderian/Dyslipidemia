import pandas as pd

# Read input file and select columns of interest
df = pd.read_csv('3final.csv')

# Drop rows where all the columns are null
df.dropna(how='all', inplace=True)
filtered_df = df.dropna(how='all', subset=['FBS', 'Ferritin', 'anemia', 'ALT', 'Chol', 'HDL', 'HB', 'AST', 'serumvitD'])

# Save filtered data to output file
filtered_df.to_csv('result.csv', index=False)