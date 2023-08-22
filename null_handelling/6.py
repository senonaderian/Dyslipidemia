import csv

def fill_empty_column(csv_file_path, column_name):
    # Open the CSV file and create a temporary list to store the modified rows
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Iterate over the rows and fill empty "iaesegi" column with "0"
    for row in rows:
        if row[column_name].strip() == '':
            row[column_name] = '2'

    # Write the modified rows back to the CSV file
    fieldnames = rows[0].keys()
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Specify the CSV file path and the column name to check
csv_file_path = 'output2.csv'
column_name = 'rosmosbat'  # Replace with the actual column name

# Call the function to fill empty "rosmosbat" column with "2"
fill_empty_column(csv_file_path, column_name)
