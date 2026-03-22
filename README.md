import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1 : Load the dataset
file_path = "C:\\Users\\ARUL KUMARAN.T\\Downloads\\household_income_expense_dataset.csv"
data = pd.read_csv(file_path)

# Step 2 : Overview the Dataset
print("Dataset Overview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Convert dataset to NumPy array
data_array = data.to_numpy()

# Step 3 : Calculate average income and expenses
avg_income = np.mean(data["Monthly_Income"])
avg_expense = np.mean(data["Total_Expense"])

print("\nAverage Monthly Income:", avg_income)
print("Average Total Expense:", avg_expense)

# Step 4 : Identify household with highest and lowest savings
savings = data["Savings"].to_numpy()

highest_savings_house = np.argmax(savings)
lowest_savings_house = np.argmin(savings)

print("\nHousehold with Highest Savings:", highest_savings_house)
print("Household with Lowest Savings:", lowest_savings_house)

# Step 5 : Average expense per household
avg_household_expense = np.mean(data[[
    "Food_Expense",
    "Housing_Expense",
    "Transport_Expense",
    "Utilities",
    "Education",
    "Healthcare",
    "Entertainment",
    "Other_Expenses"
]], axis=1)

print("\nAverage Expense per Household:")
print(avg_household_expense)

# Step 6 : Determine savings rate (Savings > 0)
savings_rate = np.mean(savings > 0)
print("\nSavings Rate (Households with positive savings):", savings_rate)

# Step 7 : Calculate correlation matrix between expenses
expense_columns = [
    "Food_Expense",
    "Housing_Expense",
    "Transport_Expense",
    "Utilities",
    "Education",
    "Healthcare",
    "Entertainment",
    "Other_Expenses"
]

expense_data = data[expense_columns].to_numpy()

correlation_matrix = np.corrcoef(expense_data, rowvar=False)

print("\nCorrelation Matrix Between Expenses:")
print(correlation_matrix)

# Step 8 : Average expense per category
avg_expense_categories = np.mean(expense_data, axis=0)

print("\nAverage Expense per Category:")
print(avg_expense_categories)

# -------------------------------
# Plot 1 : Correlation Heatmap
# -------------------------------

# Convert expense data to DataFrame for labels
expense_df = pd.DataFrame(expense_data, columns=expense_columns)

# Calculate correlation
corr_matrix = expense_df.corr()

plt.figure(figsize=(8,6))

sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)

plt.title("Correlation Matrix of Household Expenses")

plt.show()

# -------------------------------
# Plot 2 : Bar Plot (Income vs Expense)
# -------------------------------

labels = ["Average Income", "Average Expense"]
values = [avg_income, avg_expense]

plt.figure(figsize=(6,5))
plt.bar(labels, values)
plt.title("Average Income vs Total Expense")
plt.ylabel("Amount")
plt.show()

# -------------------------------
# Plot 3 : Line Plot (Expense Trend)
# -------------------------------

plt.figure(figsize=(8,5))
plt.plot(expense_columns, avg_expense_categories, marker="o")
plt.title("Average Household Expenses by Category")
plt.xlabel("Expense Categories")
plt.ylabel("Average Expense")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------
# Plot 4: Savings and Regression Line (Dotted Actual Data)
# -------------------------------

savings = data["Savings"].to_numpy()

plt.figure(figsize=(12,6))

plt.hist(savings,
         bins=10,
         color="skyblue",
         edgecolor="black")

plt.title("Household Savings Distribution")
plt.xlabel("Savings Amount")
plt.ylabel("Frequency")

plt.grid(axis="y")

plt.show()
