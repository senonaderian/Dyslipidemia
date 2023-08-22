import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('result.csv')

# Specify the columns to keep
columns_to_keep = ['VAR00001', 'num', 'sex', 'tedadaza', 'age', 'tahol', 'tahsilat', 'shogl',
                   'BMIrotbe', 'kamarzanan', 'kamarmardan', 'feshrdarj2', 'iaesegi',
                   'sabegegalb', 'sabegemagz', 'sabediabet', 'sabesaratan',
                   'sabegeasm', 'sabegekabed', 'sabegekamkhooni', 'sabegeezterab',
                   'sabegeporfeshari', 'sabegecharbiekhoon', 'Q1a', 'Q2a', 'rosmosbat', 'rosegradeee',
                   'WHS1', 'GADrotbe', 'Q1', 'Q2', 'FBS', 'diabetcas', 'diabeti', 'Ferritin',
                   'anemia', 'ALT', 'Chol', 'cholrotbe', 'cholebala', 'HDL', 'HB', 'AST',
                   'serumvitD', 'serumvitDrotb', 'TG', 'TGrotbe', 'TGbala', 'LDL', 'LDLrotbe',
                   'LDLbala', 'HDLlowwww', 'dyslipd', 'metabolicsyndrome','saranrognimejamed',
                   'faaliat', 'Physicalactiv', 'waisttoheightratio', 'feshartotal', 'prehyper',
                   'hyper1', 'hyper2', 'protein', 'fat', 'saturfat', 'polyfat', 'linoleicfat',
                   'epa', 'sodium', 'iron', 'magnesium', 'zinc', 'manganese', 'fluoride',
                   'iodine', 'vitaminA', 'vitaminE', 'vitaminB1', 'vitaminB3', 'folate',
                   'pantacid', 'vitaminc', 'vitamink', 'solublefiber', 'crudefiber', 'glucose',
                   'kilocalories', 'carbohydrate', 'cholestrol', 'monofat', 'oleicfat',
                   'linolenicfat', 'DHA', 'potassium', 'calcium', 'phosphorus', 'copper',
                   'selenium', 'chromium', 'betacarotene', 'atocopherol', 'riboB2', 'pyridoxineB6',
                   'cobalaminB12', 'biotin', 'vitaminD', 'dieteryfiber', 'insolfiber', 'suger',
                   'caffeine', 'exmilk', 'exveg', 'exfruit', 'exbread', 'exmeat', 'exfat', 'percpro',
                   'perccarb', 'percfat', 'nansonati', 'nanrogani', 'berengsefid', 'berenjsaboosdar',
                   'gooshtkamcharb', 'gooshtporcharb', 'morgbapoost', 'morgbipoost', 'shirkamcharb',
                   'shirporcharb', 'shircacao', 'mastkamcharb', 'mastporcharb', 'mastkhame', 'mastchekide',
                   'panirmamooli', 'panirkhamei', 'panirmahali', 'rogjamedheiv', 'rogjamedgiahi', 'rogjameddonbe',
                   'mayecanola', 'mayesaier', 'namakioddar', 'RFAC1_1', 'RFAC2_1', 'RFAC3_1', 'kilocalories',
                   'DDStotal','percprotein', 'perccarb', 'percfat', 'heiscoretotalfruit', 'heiscorewholefruit',
                   'heiscoretotalvegetable', 'heiscoredarkvegetable', 'heiscorewholegrain', 'heiscoredairy',
                   'heiscoretotalprotein', 'heiscoreplantprotein', 'HEIfattyacidratioscore', 'heiscorrefiendgrains',
                   'heiscoresodium', 'heiscoreaddedsugar', 'heiscoresfa', 'HEItotal2', 'DDSghalat', 'DDSsabzijat',
                   'DDSfruits', 'DDSgusht', 'DDSlabaniat', 'protein', 'carb', 'fatt']

# Create a new DataFrame with only the specified columns
df_filtered = df.loc[:, columns_to_keep]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('output2.csv', index=False)


