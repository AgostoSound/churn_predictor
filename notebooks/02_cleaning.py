import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame, binary_mappings: dict, categorical_cols: list) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)  # Drop duplicates.

    # Replace binary columns to boolean columns.
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Normalize categorical_cols.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=bool)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    scaler = StandardScaler()
    df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])

    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf],
                                labels=['0-12', '12-24', '24-48', '48-60', '60+'])
    df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True, dtype=bool)

    # Drop unimportant columns.
    unimportant_cols = ['customerID', 'tenure']
    df.drop(columns=unimportant_cols, inplace=True)

    print(' ')
    print('df')
    print(df[['MonthlyCharges', 'TotalCharges']])
    print(' ')
    print(df.dtypes)
    print(' ')

    return df


def prepare_data(file_path: str, binary_mappings: dict, categorical_cols: list) -> pd.DataFrame:
    df = pd.read_csv(file_path)  # Upload dataset.
    df = clean_data(df, binary_mappings, categorical_cols)  # Clean and preformat.

    # Split the data into training and testing sets.
    x = df.drop('Churn', axis=1)  # Features without the target variable.
    y = df['Churn']  # Target variable.

    return train_test_split(x, y, test_size=0.2, random_state=42)


binary_mappings = {
    'gender': {'Female': False, 'Male': True},
    'SeniorCitizen': {0: False, 1: True},
    'Partner': {'No': False, 'Yes': True},
    'Dependents': {'No': False, 'Yes': True},
    'PhoneService': {'No': False, 'Yes': True},
    'PaperlessBilling': {'No': False, 'Yes': True},
    'Churn': {'No': False, 'Yes': True},
}
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

x_train, x_test, y_train, y_test = prepare_data("data/telco_churn.csv", binary_mappings, categorical_cols)
