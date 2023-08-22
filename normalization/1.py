import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv('output2.csv')
data = data.apply(pd.to_numeric, errors='coerce')

target = data['HDLlow']
data = data.drop('HDLlow', axis=1)

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

std_data = std_scaler.fit_transform(data)
minmax_data = minmax_scaler.fit_transform(data)
robust_data = robust_scaler.fit_transform(data)

std_df = pd.DataFrame(std_data, columns=data.columns)
std_df['HDLlow'] = target

minmax_df = pd.DataFrame(minmax_data, columns=data.columns)
minmax_df['HDLlow'] = target

robust_df = pd.DataFrame(robust_data, columns=data.columns)
robust_df['HDLlow'] = target

std_df.to_csv('std_normalized_data.csv', index=False)
minmax_df.to_csv('minmax_normalized_data.csv', index=False)
robust_df.to_csv('robust_normalized_data.csv', index=False)
