import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import streamlit as st
from datetime import datetime, timedelta
import random

# Function to generate random portfolio
def generate_random_portfolio(num_assets):
    assets = ['Stock' + str(i) for i in range(num_assets // 2)] + ['Bond' + str(i) for i in range(num_assets // 2)]
    quantities = np.random.randint(1, 100, size=num_assets)
    buy_dates = [datetime.today() - timedelta(days=random.randint(1, 365)) for _ in range(num_assets)]
    prices = np.random.uniform(10, 100, size=num_assets)
    return pd.DataFrame({'Asset': assets, 'Quantity': quantities, 'Buy Date': buy_dates, 'Buy Price': prices})


# Function to calculate portfolio value and asset allocation
def calculate_portfolio_stats(portfolio, current_prices):
    portfolio['Current Price'] = current_prices
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()
    allocation = portfolio.groupby(portfolio['Asset'].str.contains('Stock'))['Current Value'].sum() / total_value
    return total_value, allocation


# Function to calculate tax payment
def calculate_tax_payment(portfolio, current_prices, tax_rate):
    portfolio['Current Price'] = current_prices
    portfolio['Profit'] = (portfolio['Current Price'] - portfolio['Buy Price']) * portfolio['Quantity']
    return (portfolio[portfolio['Profit'] > 0]['Profit'] * tax_rate).sum()


# Optimization function
def optimize_portfolio(portfolio, current_prices, desired_allocation, tax_rate):
    total_value, current_allocation = calculate_portfolio_stats(portfolio, current_prices)

    m = gp.Model("Portfolio Optimization")

    # Decision variables
    quantities = m.addVars(len(portfolio), vtype=GRB.INTEGER, name="q")

    # Constraints
    m.addConstr(quantities.sum() == total_value, "budget")
    m.addConstr(gp.quicksum(quantities[i] * current_prices[i] for i in range(len(portfolio)) if
                            portfolio['Asset'][i].startswith("Stock")) == desired_allocation * total_value,
                "allocation")

    # Objective
    tax_payment = calculate_tax_payment(portfolio, current_prices, tax_rate)
    m.setObjective(tax_payment, GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        optimized_quantities = [int(var.x) for var in quantities.values()]
        optimized_portfolio = portfolio.copy()
        optimized_portfolio['Quantity'] = optimized_quantities
        trades = optimized_portfolio['Quantity'] - portfolio['Quantity']
        return optimized_portfolio, trades
    else:
        print('No optimal solution found.')
        return None, None


# Streamlit app
def main():
    st.title('Portfolio Optimizer')

    # Upload portfolio
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        portfolio = pd.read_excel(uploaded_file)
    else:
        num_assets = st.number_input('Number of assets', min_value=1, value=10, step=1)
        portfolio = generate_random_portfolio(num_assets)

    # Get current prices
    current_prices = np.random.uniform(10, 100, size=len(portfolio))

    # Get desired allocation
    desired_allocation = st.slider('Desired Stock Allocation', min_value=0.0, max_value=1.0, value=0.6)

    # Get tax rate
    tax_rate = st.slider('Tax Rate', min_value=0.0, max_value=1.0, value=0.2)

    # Optimize portfolio
    if st.button('Optimize Portfolio'):
        optimized_portfolio, trades = optimize_portfolio(portfolio, current_prices, desired_allocation, tax_rate)
        st.write('Optimized Portfolio')
        st.write(optimized_portfolio)
        st.write('Trades')
        st.write(trades)


if __name__ == '__main__':
    main()