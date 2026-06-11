import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\priya\OneDrive\internship project\dataset.csv"

print("Loading Dataset...")

df = pd.read_csv(file_path)

# Task 1: Dataset Understanding
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

# Task 2: Data Cleaning
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print("\nMissing values handled!")

# Task 3: EDA
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Task 4: Visualization

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("\nProject Completed Successfully!")