import pandas as pd
import numpy as np

def fill_empty_with_mean(csv_file_path, columns):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Iterate over the specified columns
    for column in columns:
        # Convert the column values to numeric, ignoring non-numeric values
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Calculate the mean of the non-empty numeric values in the column
        mean_value = np.nanmean(df[column])

        # Fill the empty values with the calculated mean
        df[column].fillna(mean_value, inplace=True)

    # Write the modified DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

# Specify the file path and columns to fill
csv_file_path = 'output2.csv'
columns_to_fill = ['FBS', 'Ferritin', 'anemia', 'ALT', 'Chol', 'HDL', 'HB', 'AST', 'serumvitD', 'TG', 'LDL', 'saranrognimejamed', 'protein',
                    'fat', 'saturfat', 'polyfat', 'linoleicfat', 'epa', 'sodium', 'iron', 'magnesium', 'zinc', 'manganese',
                    'fluoride', 'vitaminA', 'vitaminE', 'vitaminB1', 'vitaminB3', 'folate', 'pantacid', 'vitaminc', 'vitamink',
                    'solublefiber', 'crudefiber', 'glucose', 'kilocalories', 'carbohydrate', 'cholestrol', 'monofat', 'oleicfat',
                    'linolenicfat', 'DHA', 'potassium', 'calcium', 'phosphorus', 'copper', 'selenium', 'chromium', 'betacarotene',
                    'atocopherol', 'riboB2', 'pyridoxineB6', 'cobalaminB12', 'biotin', 'vitaminD', 'dieteryfiber', 'insolfiber', 'suger',
                    'caffeine', 'exmilk', 'exveg', 'exfruit', 'exbread', 'exmeat', 'exfat', 'percpro', 'perccarb', 'percfat', 'RFAC1_1',
                    'RFAC2_1', 'RFAC3_1']
# Call the function to fill the empty spaces with the mean values
fill_empty_with_mean(csv_file_path, columns_to_fill)
