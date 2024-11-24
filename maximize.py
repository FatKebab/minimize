import pandas as pd
from scipy.optimize import linprog
import numpy as np  # Import numpy for numerical operations

# Read CSV
data = pd.read_csv("func.csv", sep=";")

# Clean up column names in case of trailing/leading spaces
data.columns = data.columns.str.strip()

# Print data for debugging
print("CSV Data:")
print(data.head())

# Extract objective function and constraints
obj = list(map(float, data.iloc[0, :-1]))  # Coefficients of the objective function
constraints = data.iloc[1:]
b = constraints.iloc[:, -1].astype(float).values  # RHS values (last column)
A = constraints.iloc[:, :-1].astype(float).values  # Constraint coefficients (all but last column)

# Solve primal problem
res_primal = linprog(c=-1 * pd.Series(obj), A_ub=A, b_ub=b, bounds=(0, None), method='highs')

# Round results for display
primal_rounded_x = np.round(res_primal.x, 4)
primal_fun_rounded = np.round(res_primal.fun, 4)

print("\nPrimal Solution:")
print(f"Optimal value (rounded): {primal_fun_rounded}")
print(f"Optimal variables (rounded): {primal_rounded_x}")

# Solve dual problem
c_dual = b
A_dual = -A.T
b_dual = -np.array(obj)  # Convert obj to NumPy array for negation
res_dual = linprog(c=c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')

# Round results for display
dual_rounded_x = np.round(res_dual.x, 4)
dual_fun_rounded = np.round(res_dual.fun, 4)

print("\nDual Solution:")
print(f"Optimal value (rounded): {dual_fun_rounded}")
print(f"Optimal variables (rounded): {dual_rounded_x}")

# Check Complementary Slackness
if res_primal.success and res_dual.success:
    print("\nComplementary Slackness is satisfied.")
else:
    print("\nComplementary Slackness is not satisfied.")
