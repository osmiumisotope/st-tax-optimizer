import streamlit as st
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import io
import base64
import openpyxl



def generate_random_portfolio(num_stocks=10, num_bonds=5):
    portfolio = []
    tickers = [f"STOCK_{i}" for i in range(num_stocks)] + [f"BOND_{i}" for i in range(num_bonds)]

    for ticker in tickers:
        num_lots = np.random.randint(1, 6)  # 1 to 5 lots per security
        for _ in range(num_lots):
            buy_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 1095))  # 30 days to 3 years ago
            quantity = np.random.randint(10, 1001)  # 10 to 1000 shares
            buy_price = np.random.uniform(10, 200)  # $10 to $200 per share
            current_price = buy_price * (1 + np.random.uniform(-0.5, 1.0))  # -50% to +100% change

            portfolio.append({
                "Ticker": ticker,
                "Buy Date": buy_date,
                "Quantity": quantity,
                "Buy Price": buy_price,
                "Current Price": current_price
            })

    return pd.DataFrame(portfolio)

def optimize_portfolio(portfolio, target_allocation, max_tax_burden, short_term_tax_rate, long_term_tax_rate):
    st.write("Starting portfolio optimization...")
    try:
        # Calculate initial portfolio value and allocation
        portfolio['Value'] = portfolio['Quantity'] * portfolio['Current Price']
        initial_value = portfolio['Value'].sum()
        initial_stock_allocation = portfolio[portfolio['Ticker'].str.contains('STOCK')]['Value'].sum() / initial_value
        initial_bond_allocation = 1 - initial_stock_allocation

        st.write(f"Initial portfolio value: ${initial_value:.2f}")
        st.write(f"Initial stock allocation: {initial_stock_allocation:.2%}")
        st.write(f"Initial bond allocation: {initial_bond_allocation:.2%}")

        # Create optimization model
        model = gp.Model("PortfolioOptimization")

        # Decision variables: amount to sell for each lot
        sell_amounts = model.addVars(portfolio.index, lb=0, name="sell_amounts")

        for i in portfolio.index:
            model.addConstr(sell_amounts[i] <= portfolio.loc[i, 'Value'])

        # Calculate capital gains and taxes
        long_term_mask = (pd.Timestamp.now() - portfolio['Buy Date']).dt.days > 365
        portfolio['Capital Gain'] = (portfolio['Current Price'] - portfolio['Buy Price']) * portfolio['Quantity']
        portfolio['Tax Rate'] = np.where(long_term_mask, long_term_tax_rate, short_term_tax_rate)
        portfolio['Tax'] = portfolio['Capital Gain'] * portfolio['Tax Rate']

        # Objective: Minimize the absolute difference from target allocation
        target_stock_value = target_allocation * initial_value
        current_stock_value = portfolio[portfolio['Ticker'].str.contains('STOCK')]['Value'].sum()

        st.write(f"Target stock value: ${target_stock_value:.2f}")
        st.write(f"Current stock value: ${current_stock_value:.2f}")

        abs_diff = model.addVar(name="abs_diff")
        model.addConstr(
            (current_stock_value - gp.quicksum(
                sell_amounts[i] for i in portfolio[portfolio['Ticker'].str.contains('STOCK')].index))
            - target_stock_value <= abs_diff
        )
        model.addConstr(
            target_stock_value - (current_stock_value - gp.quicksum(
                sell_amounts[i] for i in portfolio[portfolio['Ticker'].str.contains('STOCK')].index))
            <= abs_diff
        )

        model.setObjective(abs_diff, GRB.MINIMIZE)

        # Constraints
        # 1. Tax burden
        model.addConstr(gp.quicksum(sell_amounts[i] * portfolio.loc[i, 'Tax'] / portfolio.loc[i, 'Value'] for i in
                                    portfolio.index) <= max_tax_burden)

        # Optimize
        model.optimize()

        st.write(f"Optimization status: {model.status}")
        st.write(f"Objective value: {model.objVal}")

        if model.status == GRB.OPTIMAL:
            # Extract results
            optimized_portfolio = portfolio.copy()
            optimized_portfolio['Sell Amount'] = [sell_amounts[i].x for i in portfolio.index]
            optimized_portfolio['Sell Quantity'] = np.floor(
                optimized_portfolio['Sell Amount'] / optimized_portfolio['Current Price'])
            optimized_portfolio['New Quantity'] = optimized_portfolio['Quantity'] - optimized_portfolio['Sell Quantity']
            optimized_portfolio['New Value'] = optimized_portfolio['New Quantity'] * optimized_portfolio[
                'Current Price']
            optimized_portfolio['Tax Paid'] = optimized_portfolio['Sell Quantity'] * (
                        optimized_portfolio['Current Price'] - optimized_portfolio['Buy Price']) * optimized_portfolio[
                                                  'Tax Rate']

            new_stock_value = optimized_portfolio[optimized_portfolio['Ticker'].str.contains('STOCK')][
                'New Value'].sum()
            new_total_value = optimized_portfolio['New Value'].sum()
            new_stock_allocation = new_stock_value / new_total_value
            new_bond_allocation = 1 - new_stock_allocation

            st.write(f"New stock value: ${new_stock_value:.2f}")
            st.write(f"New total value: ${new_total_value:.2f}")
            st.write(f"New stock allocation: {new_stock_allocation:.2%}")
            st.write(f"New bond allocation: {new_bond_allocation:.2%}")

            return optimized_portfolio, new_stock_allocation, new_bond_allocation
        else:
            st.error("Optimization failed. The model is infeasible with the given constraints.")
            return None, None, None

    except gp.GurobiError as e:
        st.error(f"Gurobi error: {e}")
        return None, None, None


