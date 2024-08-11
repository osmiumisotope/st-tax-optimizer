import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import streamlit as st
from datetime import datetime, timedelta
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Function to generate random portfolio with stock lots
def generate_random_portfolio(num_assets):
    assets = ['Stock' + str(i) for i in range(num_assets // 2)] + ['Bond' + str(i) for i in range(num_assets // 2)]
    current_prices = {asset: round(random.uniform(10, 100), 2) for asset in assets}

    portfolio = pd.DataFrame(columns=['Asset', 'Quantity', 'Buy Date', 'Buy Price', 'Current Price'])
    for asset in assets:
        num_lots = random.randint(1, 5)
        lots = []
        for _ in range(num_lots):
            quantity = random.randint(1, 100)
            buy_date = datetime.today() - timedelta(days=random.randint(1, 1095))
            buy_price = round(random.uniform(0.5 * current_prices[asset], current_prices[asset]), 2)
            lots.append({'Asset': asset, 'Quantity': quantity, 'Buy Date': buy_date, 'Buy Price': buy_price,
                         'Current Price': current_prices[asset]})
        portfolio = pd.concat([portfolio, pd.DataFrame(lots)], ignore_index=True)
    return portfolio, current_prices


# Function to calculate portfolio value and asset allocation (same as before)
def calculate_portfolio_stats(portfolio, current_prices):
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()

    if total_value == 0:
        allocation = 0
    else:
        allocation = portfolio[portfolio['Asset'].str.contains('Stock')]['Current Value'].sum() / total_value

    return total_value, allocation


# Function to calculate tax payment considering long-term and short-term capital gains
def calculate_tax_payment(portfolio, current_prices, tax_rate_lt, tax_rate_st):
    portfolio['Current Price'] = portfolio['Asset'].map(current_prices)
    portfolio['Profit'] = (portfolio['Current Price'] - portfolio['Buy Price']) * portfolio['Quantity']
    portfolio['Days Held'] = (datetime.today() - portfolio['Buy Date']).dt.days
    tax_payment_lt = (
                portfolio[(portfolio['Profit'] > 0) & (portfolio['Days Held'] > 365)]['Profit'] * tax_rate_lt).sum()
    tax_payment_st = (
                portfolio[(portfolio['Profit'] > 0) & (portfolio['Days Held'] <= 365)]['Profit'] * tax_rate_st).sum()
    return tax_payment_lt + tax_payment_st


# Optimization function using Gurobi
def optimize_portfolio(portfolio, current_prices, desired_allocation, tax_rate_lt, tax_rate_st, max_tax_burden):
    total_value, current_allocation = calculate_portfolio_stats(portfolio, current_prices)

    if total_value == 0:
        print("The total portfolio value is zero. Cannot optimize.")
        return None, None

    m = gp.Model("Portfolio Optimization")

    # Decision variables
    sell_pct = m.addVars(len(portfolio), lb=0, ub=1, name="sell_pct")

    # Constraints
    m.addConstr(gp.quicksum(sell_pct[i] * portfolio['Quantity'][i] * portfolio['Current Price'][i] for i in
                            range(len(portfolio))) <= total_value, "budget")

    if desired_allocation[0] > 0:
        m.addConstr(gp.quicksum(
            sell_pct[i] * portfolio['Quantity'][i] * portfolio['Current Price'][i] for i in range(len(portfolio)) if
            portfolio['Asset'][i].startswith("Stock")) >= desired_allocation[0] * total_value, "stock_allocation_lower")

    if desired_allocation[1] < 1:
        m.addConstr(gp.quicksum(
            sell_pct[i] * portfolio['Quantity'][i] * portfolio['Current Price'][i] for i in range(len(portfolio)) if
            portfolio['Asset'][i].startswith("Stock")) <= desired_allocation[1] * total_value, "stock_allocation_upper")

    tax_payment = gp.quicksum(
        (portfolio['Current Price'][i] - portfolio['Buy Price'][i]) * sell_pct[i] * portfolio['Quantity'][i] * (
            tax_rate_lt if portfolio['Days Held'][i] > 365 else tax_rate_st) for i in range(len(portfolio)) if
        portfolio['Current Price'][i] > portfolio['Buy Price'][i])
    m.addConstr(tax_payment <= max_tax_burden, "max_tax_burden")

    # Objective
    allocation_deviation = gp.quicksum(
        sell_pct[i] * portfolio['Quantity'][i] * portfolio['Current Price'][i] for i in range(len(portfolio)) if
        portfolio['Asset'][i].startswith("Stock")) - desired_allocation[0] * total_value
    m.setObjective(allocation_deviation, GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        optimized_portfolio = portfolio.copy()
        optimized_portfolio['Sell Qty'] = [round(var.x * portfolio['Quantity'][i]) for i, var in
                                           enumerate(sell_pct.values())]
        trades = optimized_portfolio[optimized_portfolio['Sell Qty'] > 0]
        return optimized_portfolio, trades
    else:
        print('No optimal solution found.')
        return None, None


# Streamlit app
def main():
    st.title('Portfolio Optimizer')

    # Upload portfolio or generate random portfolio
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        portfolio = pd.read_excel(uploaded_file)
        current_prices = {asset: portfolio[portfolio['Asset'] == asset]['Current Price'].iloc[0] for asset in
                          portfolio['Asset'].unique()}
    else:
        num_assets = st.number_input('Number of assets', min_value=1, value=10, step=1)
        portfolio, current_prices = generate_random_portfolio(num_assets)

    # Display portfolio
    st.subheader('Portfolio')
    st.write(portfolio)

    # Calculate current allocation
    total_value, current_allocation = calculate_portfolio_stats(portfolio, current_prices)

    # Display current allocation
    st.subheader('Current Allocation')
    allocation_table = pd.DataFrame({
        'Asset Class': ['Stock', 'Bond'],
        'Allocation': [current_allocation, 1 - current_allocation]
    })
    allocation_table['Allocation'] = allocation_table['Allocation'].apply(lambda x: f'{x:.2%}')
    st.table(allocation_table)

    # Get desired allocation range
    desired_allocation_lower = st.slider('Desired Stock Allocation Lower Bound', min_value=0.0, max_value=1.0,
                                         value=0.55)
    desired_allocation_upper = st.slider('Desired Stock Allocation Upper Bound', min_value=desired_allocation_lower,
                                         max_value=1.0, value=0.65)
    desired_allocation = (desired_allocation_lower, desired_allocation_upper)

    # Get tax rates
    tax_rate_lt = st.slider('Long-term Capital Gains Tax Rate', min_value=0.0, max_value=1.0, value=0.15)
    tax_rate_st = st.slider('Short-term Capital Gains Tax Rate', min_value=0.0, max_value=1.0, value=0.35)

    # Get maximum tax burden
    max_tax_burden = st.number_input('Maximum Tax Burden ($)', min_value=0, value=1000, step=100)

    # Optimize portfolio
    if st.button('Optimize Portfolio'):
        optimized_portfolio, trades = optimize_portfolio(portfolio, current_prices, desired_allocation, tax_rate_lt,
                                                         tax_rate_st, max_tax_burden)
        if optimized_portfolio is not None:
            st.subheader('Optimized Portfolio')
            st.write(optimized_portfolio)

            # Calculate optimized allocation
            _, optimized_allocation = calculate_portfolio_stats(optimized_portfolio, current_prices)

            # Display optimized allocation
            st.subheader('Optimized Allocation')
            optimized_allocation_table = pd.DataFrame({
                'Asset Class': ['Stock', 'Bond'],
                'Allocation': [optimized_allocation, 1 - optimized_allocation]
            })
            optimized_allocation_table['Allocation'] = optimized_allocation_table['Allocation'].apply(
                lambda x: f'{x:.2%}')
            st.table(optimized_allocation_table)

            st.subheader('Trades')
            st.write(trades)


if __name__ == '__main__':
    main()

