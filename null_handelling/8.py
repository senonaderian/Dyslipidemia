import csv

def fill_empty_columns(csv_file_path, column_names):
    # Open the CSV file and create a temporary list to store the modified rows
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Iterate over the rows and fill empty values in specified columns with "0"
    for row in rows:
        for column_name in column_names:
            if row[column_name].strip() == '':
                row[column_name] = '0'

    # Write the modified rows back to the CSV file
    fieldnames = rows[0].keys()
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Specify the CSV file path and the list of column names to check
csv_file_path = 'output2.csv'
column_names = ['BMIrotbe', 'feshrdarj2', 'Q1a', 'Q2a', 'GADrotbe', 'Q1', 'Q2', 'diabetcas',
'cholrotbe', 'cholebala', 'serumvitDrotb', 'TGrotbe', 'TGbala', 'LDLrotbe','LDLbala', 'HDLlowwww', 'dyslipd', 'waisttoheightratio',
'prehyper', 'hyper1', 'hyper2', 'nansonati', 'nanrogani', 'berengsefid', 'berenjsaboosdar', 'gooshtkamcharb', 'gooshtporcharb',
'morgbapoost', 'morgbipoost', 'shirkamcharb', 'shirporcharb', 'shircacao', 'mastkamcharb', 'mastporcharb', 'mastkhame', 'mastchekide',
'panirmamooli', 'panirkhamei', 'panirmahali', 'rogjamedheiv', 'rogjamedgiahi', 'rogjameddonbe', 'mayecanola', 'mayesaier', 'namakioddar']  # Replace with the actual column names

# Call the function to fill empty values in specified columns with "0"
fill_empty_columns(csv_file_path, column_names)
