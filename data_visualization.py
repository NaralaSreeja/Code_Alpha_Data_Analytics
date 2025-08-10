import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

# Remove rows with missing key values
df = df.dropna(subset=['age', 'fare', 'sex', 'pclass', 'survived'])

# ==========================
# 1. Survival by Gender
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df, palette='pastel')
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# ==========================
# 2. Age Distribution
# ==========================
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ==========================
# 3. Fare by Class
# ==========================
plt.figure(figsize=(6,4))
sns.boxplot(x='pclass', y='fare', data=df, palette='pastel')
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# ==========================
# 4. Correlation Heatmap
# ==========================
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()