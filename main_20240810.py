import streamlit as st
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import io
import base64
import openpyxl

# Set page config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")


# Function to generate a random portfolio
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


# Function to download dataframe as Excel
def download_excel(df, filename):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href


# Function to optimize portfolio
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


# Streamlit UI
st.title("Portfolio Optimizer")

# Sidebar for inputs
st.sidebar.header("Portfolio Input")
input_method = st.sidebar.radio("Choose input method:", ("Generate Random Portfolio", "Upload Excel File"))

if input_method == "Generate Random Portfolio":
    if st.sidebar.button("Generate Random Portfolio"):
        portfolio = generate_random_portfolio()
        st.session_state.portfolio = portfolio
elif input_method == "Upload Excel File":
    uploaded_file = st.sidebar.file_uploader("Upload your portfolio Excel file", type="xlsx")
    if uploaded_file is not None:
        portfolio = pd.read_excel(uploaded_file)
        st.session_state.portfolio = portfolio

    # Download template button
    template = pd.DataFrame(columns=["Ticker", "Buy Date", "Quantity", "Buy Price", "Current Price"])
    st.sidebar.markdown(download_excel(template, "portfolio_template.xlsx"), unsafe_allow_html=True)

# Display current portfolio
if 'portfolio' in st.session_state:
    st.subheader("Current Portfolio")
    st.write(st.session_state.portfolio)

    # Calculate current allocation
    total_value = (st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Current Price']).sum()
    stock_value = st.session_state.portfolio[st.session_state.portfolio['Ticker'].str.contains('STOCK')]['Quantity'] * \
                  st.session_state.portfolio[st.session_state.portfolio['Ticker'].str.contains('STOCK')][
                      'Current Price']
    current_stock_allocation = stock_value.sum() / total_value
    current_bond_allocation = 1 - current_stock_allocation

    st.write(f"Current Stock Allocation: {current_stock_allocation:.2%}")
    st.write(f"Current Bond Allocation: {current_bond_allocation:.2%}")

    # Input for desired allocation
    st.subheader("Desired Allocation")
    allocation_options = {
        "Most Conservative (10:90)": 0.1,
        "Conservative (20:80)": 0.2,
        "Moderately Conservative (30:70)": 0.3,
        "Moderate (40:60)": 0.4,
        "Balanced (50:50)": 0.5,
        "Moderately Aggressive (60:40)": 0.6,
        "Aggressive (70:30)": 0.7,
        "Very Aggressive (80:20)": 0.8,
        "Most Aggressive (90:10)": 0.9
    }
    selected_allocation = st.selectbox("Select desired stock:bond allocation:", list(allocation_options.keys()))
    target_allocation = allocation_options[selected_allocation]

    # Input for tax rates and max tax burden
    col1, col2, col3 = st.columns(3)
    with col1:
        short_term_tax_rate = st.number_input("Short-term Tax Rate (%)", min_value=0.0, max_value=100.0,
                                              value=35.0) / 100
    with col2:
        long_term_tax_rate = st.number_input("Long-term Tax Rate (%)", min_value=0.0, max_value=100.0, value=15.0) / 100
    with col3:
        max_tax_burden = st.number_input("Maximum Tax Burden ($)", min_value=0, value=3000)

    # Optimize button
    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            optimized_portfolio, new_stock_allocation, new_bond_allocation = optimize_portfolio(
                st.session_state.portfolio, target_allocation, max_tax_burden, short_term_tax_rate, long_term_tax_rate
            )

        if optimized_portfolio is not None:
            st.subheader("Optimized Portfolio")
            st.write(optimized_portfolio)

            st.subheader("Allocation Summary")
            summary_data = {
                "": ["Stocks", "Bonds"],
                "Current Allocation": [f"{current_stock_allocation:.2%}", f"{current_bond_allocation:.2%}"],
                "Target Allocation": [f"{target_allocation:.2%}", f"{1 - target_allocation:.2%}"],
                "New Allocation": [f"{new_stock_allocation:.2%}", f"{new_bond_allocation:.2%}"]
            }
            st.table(pd.DataFrame(summary_data).set_index(""))

            st.subheader("Trade Summary")
            trades = optimized_portfolio[optimized_portfolio['Sell Quantity'] > 0][
                ['Ticker', 'Sell Quantity', 'Current Price', 'Tax Paid']]
            st.write(trades)

            total_tax_paid = optimized_portfolio['Tax Paid'].sum()
            st.write(f"Total Tax Paid: ${total_tax_paid:.2f}")
        else:
            st.error("Failed to optimize the portfolio. Please adjust your constraints and try again.")
else:
    st.info("Please generate a random portfolio or upload an Excel file to begin.")