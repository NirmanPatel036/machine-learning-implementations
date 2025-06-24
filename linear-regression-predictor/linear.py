import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("salary_data.csv")

# Ensure numeric and clean data
# The Salary column has comma separators that need to be removed
df['Salary'] = df['Salary'].str.replace(',', '').astype(float)

# Alternative approach if column names are different:
# Check actual column names first
print("Column names:", df.columns.tolist())
print("Data types:", df.dtypes)
print("First few rows:")
print(df.head())

# Visualize the initial dataset
plt.scatter(df.YearsExperience, df.Salary, color='blue')
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Define loss function
def loss_function(m, c, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m * x + c)) ** 2
    return total_error / float(len(points))

# Gradient descent step
def gradient_descent(m_old, c_old, points, alpha):
    m_gradient = 0
    c_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        m_gradient += -(2 / n) * x * (y - (m_old * x + c_old))
        c_gradient += -(2 / n) * (y - (m_old * x + c_old))

    m_new = m_old - alpha * m_gradient
    c_new = c_old - alpha * c_gradient

    return m_new, c_new

# Training loop with initialization and learning rate
m = 0
c = 0
alpha = 0.0001
epochs = 1000

# Print initial loss to monitor training
initial_loss = loss_function(m, c, df)
print(f"Initial loss: {initial_loss:.2f}")

for i in range(epochs):
    m, c = gradient_descent(m, c, df, alpha)
    if i % 200 == 0:
        current_loss = loss_function(m, c, df)
        print(f"Epoch {i}: m = {m:.4f}, c = {c:.2f}, loss = {current_loss:.2f}")

final_loss = loss_function(m, c, df)
print(f"\nFinal equation: y = {m:.2f}x + {c:.2f}")
print(f"Final loss: {final_loss:.2f}")

plt.scatter (df.YearsExperience, df.Salary, color="black")
plt.plot(list(range(1, 10)), [m * x + c for x in range(1, 10)], color="red")
plt.show()