import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('output2.csv')

# Get the names of columns with empty cells
columns_with_empty_cells = df.columns[df.isnull().any()].tolist()

# Iterate over columns with empty cells
for column in columns_with_empty_cells:
    empty_rows = df[df[column].isnull()]
    print("Rows with empty cells in column", column, ":")
    for index, row in empty_rows.iterrows():
        print(index)
