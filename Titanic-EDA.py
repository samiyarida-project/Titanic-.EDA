Titanic Dataset - Simple EDA Project 

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing age values
df['age'] = df['age'].fillna(df['age'].mean())

# Create subplot window
fig, axes = plt.subplots(2, 3, figsize=(15,8))

# 1. Survival count
sns.countplot(x='survived', data=df, ax=axes[0,0])
axes[0,0].set_title("Survival Count")

# 2. Survival based on gender
sns.countplot(x='sex', hue='survived', data=df, ax=axes[0,1])
axes[0,1].set_title("Gender vs Survival")

# 3. Passenger class count
sns.countplot(x='class', data=df, ax=axes[0,2])
axes[0,2].set_title("Passenger Class")

# 4. Age distribution
sns.histplot(df['age'], kde=True, ax=axes[1,0])
axes[1,0].set_title("Age Distribution")

# 5. Fare distribution
sns.histplot(df['fare'], kde=True, ax=axes[1,1])
axes[1,1].set_title("Fare Distribution")

# Adjust layout 
plt.tight_layout()

# Show all plots
plt.show()

# Insights
print("\nEDA Completed Successfully")
print("Insights:")
print("1. Females survived more than males")
print("2. First class passengers had higher survival rate")
print("3. Most passengers were between 20-40 years old")
