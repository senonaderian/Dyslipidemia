import csv

def fill_empty_columns(csv_file_path, columns):
    # Open the CSV file and create a temporary list to store the modified rows
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    # Get the column indices to check
    header_row = rows[0]
    col_indices = [header_row.index(column) for column in columns]

    # Iterate over the rows (excluding the header row) and fill empty columns with "2"
    for row in rows[1:]:
        for col_index in col_indices:
            if row[col_index].strip() == '':
                row[col_index] = '2'

    # Write the modified rows back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Specify the CSV file path and the columns to check
csv_file_path = 'output2.csv'
columns_to_check = ['kamarzanan', 'kamarmardan']  # Replace with the actual column names

# Call the function to fill empty columns with "2"
fill_empty_columns(csv_file_path, columns_to_check)