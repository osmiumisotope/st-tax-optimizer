import gurobipy as gp
from gurobipy import GRB
import io
import base64
import openpyxl



def generate_random_portfolio(num_stocks=10, num_bonds=5):
    portfolio = []
    tickers = [f"STOCK_{i}" for i in range(num_stocks)] + [f"BOND_{i}" for i in range(num_bonds)]

# Create a new model
model = gp.Model("production_optimization")

# Create variables
x = model.addVar(vtype=GRB.CONTINUOUS, name="chairs")
y = model.addVar(vtype=GRB.CONTINUOUS, name="tables")

# Set objective
model.setObjective(20 * x + 30 * y, GRB.MAXIMIZE)

# Add constraints
model.addConstr(5 * x + 10 * y <= 200, "wood")
model.addConstr(2 * x + 3 * y <= 60, "labor")

# Optimize model
model.optimize()

# Print solution
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found:")
    print(f"Chairs to produce: {x.x:.2f}")
    print(f"Tables to produce: {y.x:.2f}")
    print(f"Total profit: ${model.objVal:.2f}")
else:
    print("No optimal solution found")
