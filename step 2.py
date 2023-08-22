import pandas as pd

def convert_dtype(x):
    if not x:
        return ''
    try:
        return str(x)
    except:
        return ''

# Read in the two CSV files
df1 = pd.read_csv("1final.csv", low_memory=False)
df2 = pd.read_csv("ffq.csv", low_memory=False)

# Get the set of column names in each file
set1 = set(df1.columns)
set2 = set(df2.columns)

# Find the differences in the column names
diff1 = sorted(set1 - set2)
diff2 = sorted(set2 - set1)

# Print the results
if diff1:
    print(f"file1 has columns {diff1} that file2 doesn't have")
if diff2:
    print(f"file2 has columns {diff2} that file1 doesn't have")

# Make a copy of the 2 DataFrame
df2_copy = df2.copy()

# Specify the columns to copy from df1 to df2_copy
cols_to_copy = ['BMI', 'BMIrotbe', 'GADkoll', 'GADrotbe', 'HDLlowwww', 'LDLbala', 'LDLrotbe', 'NTI001', 'Physicalactiv',
                'TGbala', 'TGrotbe', 'WbeHmardan', 'WbeHzanann', 'age10sal', 'anemia', 'chgh', 'cholebala', 'cholrotbe',
                'conicityindex', 'creatinedr', 'creatinimmol', 'diabetcas', 'diabeti', 'diasrotb', 'dorekamarbala',
                'dorkamarbebasan', 'dyslipd', 'ezafechagh', 'ezafevazn', 'faaliat', 'feshartotal', 'feshrdarj2',
                'filter_$', 'gandsara', 'hyper1', 'hyper2', 'intersalt', 'intersaltgr', 'kamarmardan', 'kamarzanan',
                'metabolicsyndrome', 'nacldariaftipredictiv', 'naclpredbala', 'potasiuedr', 'prcr24', 'prehyper',
                'proffetion', 'ros2tagir', 'ros5tagir', 'rosegradeee', 'rosmosbat', 'saranemasrafenamak',
                'saranenamak', 'saranenamakbala', 'saranenamakk', 'saranesang4', 'saranesangnamak', 'saranrogan',
                'saranrogjamed', 'saranrogmaie', 'saranrognimejamed', 'serumvitDrotb', 'shahrepurmia', 'shahrroosta',
                'shekarsara', 'shekarvagandsara', 'sisrotbe', 'sodiuedra', 'sodium24edr', 'vitDserkam',
                'waisttoheight', 'waisttoheightratio']

# Copy the specified columns from df1 to df2_copy
df2_copy[cols_to_copy] = df1[cols_to_copy]

# Save the modified df2_copy to a new CSV file
df2_copy.to_csv('ffqcopy.csv', index=False)

# Read the first CSV file
df1 = pd.read_csv('ffqcopy.csv')

# Read the second CSV file
df2 = pd.read_csv('1final.csv')

# Get the column order from the second CSV file
desired_columns = df2.columns

# Rearrange the columns of the first dataframe based on the desired order
df1 = df1[desired_columns]

# Save the rearranged dataframe to a new CSV file
df1.to_csv('output2.csv', index=False)

# Read in the CSV files
df1 = pd.read_csv('1final.csv', converters={'age': convert_dtype,'tahol': convert_dtype, 'tahsilat': convert_dtype,
                                           'fesharekhoon': convert_dtype, 'hamlegalbi': convert_dtype, 'sektemagzi': convert_dtype,
                                           'saratan': convert_dtype, 'asm': convert_dtype, 'kamkhooni': convert_dtype,
                                           'charbiekhoon': convert_dtype, 'kabedcharb': convert_dtype})

df2 = pd.read_csv('output2.csv', converters={'sex': convert_dtype,'codkhooshe': convert_dtype,'codkhanevar': convert_dtype
                                            ,'age': convert_dtype,'tahol': convert_dtype,'tahsilat': convert_dtype
                                            ,'diabet': convert_dtype,'ezterab': convert_dtype,'fesharekhoon': convert_dtype
                                            ,'hamlegalbi': convert_dtype,'sektemagzi': convert_dtype,'saratan': convert_dtype
                                            ,'asm': convert_dtype,'kamkhooni': convert_dtype,'charbiekhoon': convert_dtype
                                            ,'kabedcharb': convert_dtype})

# Iterate over the columns in file 2
for column in df2.columns:
    # Check if the column exists in file 1
    if column in df1.columns:
        # Add any extra data from file 2 to file 1
        df1[column] = df2[column].combine_first(df1[column])

# Save the merged dataframe to a new CSV file
df1.to_csv('2final.csv', index=False)
