# Exploratory Data Analysis (EDA) - Telco Customer Churn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.get_dataset import load_dataset


df = load_dataset("data/telco_churn.csv")

print(" ")
print(" ")

# Shape
print("Dataset shape:", df.shape)
print(" ")

# Preview
print("\nFirst 5 rows:")
print(df.head())
print(" ")

# Info & nulls
print("\nDataset info:")
print(df.info())
print(" ")

print("\nMissing values:")
print(df.isnull().sum())
print(" ")

# Descriptive stats
print("\nDescriptive statistics:")
print(df.describe())
print(" ")

# Target distribution
print("\nTarget variable 'Churn' distribution:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))
print(" ")

sns.countplot(data=df, x='Churn')
plt.title("Distribución de Churn")
plt.show()
print(" ")

# Histograms for numeric columns
df.select_dtypes(include='number').hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()
print(" ")

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de correlación")
plt.show()
print(" ")
