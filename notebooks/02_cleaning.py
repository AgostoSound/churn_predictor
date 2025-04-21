import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame, categorical_cols: list, binary_mappings: dict) -> pd.DataFrame:
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Replace null values with the mean of the column.
    df.drop_duplicates(inplace=True)  # Drop duplicates.

    # Replace binary columns to boolean columns.
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Convert categorical variables to binary variables.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=bool)

    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns  # Get numerical cols.
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # Automatically scales.

    print(' ')
    print('DATAFRAME')
    print(df)
    print(' ')

    return df


def prepare_data(file_path: str, categorical_cols: list, binary_mappings: dict) -> pd.DataFrame:
    df = pd.read_csv(file_path)  # Upload dataset.
    df = clean_data(df, categorical_cols, binary_mappings)  # Clean and preformat.

    # Split the data into training and testing sets.
    x = df.drop('Churn', axis=1)  # Features without the target variable.
    y = df['Churn']  # Target variable.

    return train_test_split(x, y, test_size=0.2, random_state=42)



categorical_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
binary_mappings = {
    'gender': {'Female': False, 'Male': True},
    'owns_house': {'No': False, 'Yes': True}
}
x_train, x_test, y_train, y_test = prepare_data("data/telco_churn.csv", categorical_cols, binary_mappings)






print(' ')
print('First 5 rows of the training set.')
print(x_train.head())  # First 5 rows of the training set.
print(' ')
print(' ')
print('First 5 rows of the target variable.')
print(y_train.head())  # First 5 rows of the target variable.
print(' ')
