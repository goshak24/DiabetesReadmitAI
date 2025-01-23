import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('data/diabetic_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['encounter_id', 'patient_nbr', 'payer_code'])


df['diag_1'] = df['diag_1'].replace('V', '10', regex=True)
df['diag_2'] = df['diag_2'].replace('V', '10', regex=True)
df['diag_3'] = df['diag_3'].replace('V', '10', regex=True)

df['diag_1'] = pd.to_numeric(df['diag_1'], errors='coerce')
df['diag_2'] = pd.to_numeric(df['diag_2'], errors='coerce')
df['diag_3'] = pd.to_numeric(df['diag_3'], errors='coerce')

# Drop rows where any of diag_1, diag_2, or diag_3 are NaN (i.e., where they contained non-numeric values)
df = df.dropna(subset=['diag_1', 'diag_2', 'diag_3'])


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Columns to encode
columns_to_encode = ['race', 'gender', 'age','weight', 'medical_specialty']  # Add more columns as needed

columns_from_15_onwards = df.columns[19:]
columns_to_encode.extend(columns_from_15_onwards)

# Apply LabelEncoder to each specified column
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Drop rows with '?' in specific columns
columns_to_check = ['race', 'diag_1', 'diag_2', 'diag_3']
df = df[~df[columns_to_check].isin(['?']).any(axis=1)]

# Save the formatted DataFrame to a new CSV
df.to_csv('data/diabetic_data_formatted.csv', index=False)

print(df)

