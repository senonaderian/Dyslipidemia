import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('output2.csv')

# List of rows to delete
rows_to_delete = [8, 28, 32, 33, 43 ,82, 99, 116, 118, 119, 122, 124, 128, 129, 138,
                  158, 169, 171, 210, 212, 213, 225, 230, 232, 233 , 237, 274, 280,
                  283, 304, 321, 326, 332, 345, 346, 348, 356, 365, 379,
                  387, 408, 419, 443, 485, 495]

# Delete the rows from the DataFrame
df = df.drop(rows_to_delete)

# Print the updated DataFrame
print(df.shape)
