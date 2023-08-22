import pandas as pd

# Merging 2 datasets based on their differences
df1 = pd.read_csv('1.csv', low_memory=False)
df2 = pd.read_csv('2.csv', low_memory=False)

# Specify the columns to copy from df1 to df2_copy
cols_to_copy_df1 = ['BMI', 'BMIrotbe', 'GADkoll', 'GADrotbe', 'HDLlowwww', 'LDLbala', 'LDLrotbe', 'NTI001', 'Physicalactiv', 'TGbala', 'TGrotbe', 'WbeHmardan', 'WbeHzanann', 'age10sal', 'anemia', 'chgh', 'cholebala', 'cholrotbe', 'conicityindex', 'creatinedr', 'creatinimmol', 'diabetcas', 'diabeti', 'diasrotb', 'dorekamarbala', 'dorkamarbebasan', 'dyslipd', 'ezafechagh', 'ezafevazn', 'faaliat', 'feshartotal', 'feshrdarj2', 'filter_$', 'gandsara', 'hyper1', 'hyper2', 'intersalt', 'intersaltgr', 'kamarmardan', 'kamarzanan', 'metabolicsyndrome', 'nacldariaftipredictiv', 'naclpredbala', 'potasiuedr', 'prcr24', 'prehyper', 'proffetion', 'ros2tagir', 'ros5tagir', 'rosegradeee', 'rosmosbat', 'saranemasrafenamak', 'saranenamak', 'saranenamakbala', 'saranenamakk', 'saranesang4', 'saranesangnamak', 'saranrogan', 'saranrogjamed', 'saranrogmaie', 'saranrognimejamed', 'serumvitDrotb', 'shahrepurmia', 'shahrroosta', 'shekarsara', 'shekarvagandsara', 'sisrotbe', 'sodiuedra', 'sodium24edr', 'vitDserkam', 'waisttoheight', 'waisttoheightratio']

# Specify the columns to copy from df2 to df1_copy
cols_to_copy_df2 = ['FAC1_1', 'FAC2_1', 'FAC3_1', 'RFAC1_1', 'RFAC2_1', 'RFAC3_1']

# Make copies of the DataFrames
df1_copy = df1.copy()
df2_copy = df2.copy()

# Copy the specified columns from df1 to df2_copy
df2_copy[cols_to_copy_df1] = df1[cols_to_copy_df1]

# Copy the specified columns from df2 to df1_copy
df1_copy[cols_to_copy_df2] = df2[cols_to_copy_df2]

# Save the modified DataFrames to new CSV files
df2_copy.to_csv('2_copy.csv', index=False)
df1_copy.to_csv('1_copy.csv', index=False)

# Read the first CSV file
df1 = pd.read_csv('2_copy.csv')

# Read the second CSV file
df2 = pd.read_csv('1_copy.csv')

# Get the column order from the second CSV file
desired_columns = df2.columns

# Rearrange the columns of the first DataFrame based on the desired order
df1 = df1[desired_columns]

# Save the rearranged DataFrame to a new CSV file
df1.to_csv('output.csv', index=False)

# Read in the CSV files
df1 = pd.read_csv('1_copy.csv', converters={'age': str, 'tahol': str, 'tahsilat': str, 'fesharekhoon': str, 'hamlegalbi': str, 'sektemagzi': str, 'saratan': str, 'asm': str, 'kamkhooni': str, 'charbiekhoon': str, 'kabedcharb': str})
df2 = pd.read_csv('output.csv', converters={'sex': str, 'codkhooshe': str, 'codkhanevar': str, 'age': str, 'tahol': str, 'tahsilat': str, 'diabet': str, 'ezterab': str, 'fesharekhoon': str, 'hamlegalbi': str, 'sektemagzi': str, 'saratan': str, 'asm': str, 'kamkhooni': str, 'charbiekhoon': str, 'kabedcharb': str})

# Iterate over the columns in df2
for column in df2.columns:
    # Check if the column exists in df1
    if column in df1.columns:
        # Add any extra data from df2 to df1
        df1[column] = df2[column].combine_first(df1[column])


# Save the merged DataFrame to a new CSV file
df1.to_csv('1final.csv', index=False)
