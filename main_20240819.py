import streamlit as st
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import io
import base64
import openpyxl
import plotly.graph_objects as go

# Set page config
# st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon=":abacus:")
st.markdown(f'<style>{open("style.css").read()}</style>', unsafe_allow_html=True)


# Function to generate a random portfolio
def generate_random_portfolio(num_stocks=10, num_bonds=5):
    portfolio = []
    tickers = [f"STOCK_{i}" for i in range(num_stocks)] + [f"BOND_{i}" for i in range(num_bonds)]

    for ticker in tickers:
        # Generate current price for this ticker
        base_price = np.random.uniform(10, 200)  # Base price between $10 and $200
        current_price = base_price * (1 + np.random.uniform(-0.3, 0.5))  # -30% to +50% change

        num_lots = np.random.randint(1, 6)  # 1 to 5 lots per security
        for _ in range(num_lots):
            buy_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 1095))  # 30 days to 3 years ago
            quantity = np.random.randint(10, 1001)  # 10 to 1000 shares
            buy_price = base_price * (1 + np.random.uniform(-0.2, 0.2))  # Buy price within ±20% of base price

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
    # st.write("Starting portfolio optimization...")
    try:
        # Calculate initial portfolio value and allocation
        portfolio['Value'] = portfolio['Quantity'] * portfolio['Current Price']
        initial_value = portfolio['Value'].sum()
        initial_stock_value = portfolio[portfolio['Ticker'].str.contains('STOCK')]['Value'].sum()
        initial_bond_value = initial_value - initial_stock_value
        initial_stock_allocation = initial_stock_value / initial_value
        initial_bond_allocation = 1 - initial_stock_allocation

        # st.write(f"Initial portfolio value: ${initial_value:.2f}")
        # st.write(f"Initial stock allocation: {initial_stock_allocation:.2%}")
        # st.write(f"Initial bond allocation: {initial_bond_allocation:.2%}")

        # Create optimization model
        model = gp.Model("PortfolioRebalancing")

        # Decision variables
        sell_fractions = model.addVars(portfolio.index, lb=0, ub=1, name="sell_fractions")
        buy_amounts = model.addVars(portfolio['Ticker'].unique(), lb=0, name="buy_amounts")

        # Calculate capital gains and taxes
        portfolio['Buy Date'] = pd.to_datetime(portfolio['Buy Date'])
        long_term_mask = (pd.Timestamp.now() - portfolio['Buy Date']).dt.days > 365
        portfolio['Tax Rate'] = np.where(long_term_mask, long_term_tax_rate, short_term_tax_rate)
        portfolio['Capital Gain Per Share'] = portfolio['Current Price'] - portfolio['Buy Price']

        # Variables for allocation deviation

        # Constraints
        # 1. Target allocation
        target_stock_value = target_allocation * initial_value
        current_stock_value = (
                gp.quicksum((1 - sell_fractions[i]) * portfolio.loc[i, 'Value'] for i in
                            portfolio[portfolio['Ticker'].str.contains('STOCK')].index) +
                gp.quicksum(buy_amounts[t] for t in portfolio['Ticker'].unique() if 'STOCK' in t)
        )
        abs_deviation = model.addVar(name="absolute_deviation")

        model.addConstr(current_stock_value - target_stock_value <= abs_deviation, "upper_bound_deviation")
        model.addConstr(target_stock_value - current_stock_value <= abs_deviation, "lower_bound_deviation")

        # 2. Tax calculation
        tax_paid = gp.quicksum(
            sell_fractions[i] * portfolio.loc[i, 'Quantity'] * portfolio.loc[i, 'Capital Gain Per Share'] *
            portfolio.loc[i, 'Tax Rate']
            for i in portfolio.index
        )
        model.addConstr(tax_paid <= max_tax_burden)

        # 3. Cash balance (money from selling = money for buying)
        cash_from_selling = gp.quicksum(sell_fractions[i] * portfolio.loc[i, 'Value'] for i in portfolio.index)
        cash_for_buying = gp.quicksum(buy_amounts[t] for t in portfolio['Ticker'].unique())
        model.addConstr(cash_from_selling - cash_for_buying <= initial_value * 0.02)
        model.addConstr(cash_from_selling >= cash_for_buying)

        # 4. Prevent unnecessary trading
        total_traded = cash_from_selling + cash_for_buying
        model.addConstr(total_traded <= 2 * abs(target_stock_value - initial_stock_value))

        model.addConstr(
            gp.quicksum((1 - sell_fractions[i]) * portfolio.loc[i, 'Value'] for i in portfolio.index) +
            gp.quicksum(buy_amounts[t] for t in portfolio['Ticker'].unique()) >= 0.97 * initial_value,
            "maintain_total_value")

        model.addConstr(
            gp.quicksum((1 - sell_fractions[i]) * portfolio.loc[i, 'Value'] for i in portfolio.index) +
            gp.quicksum(buy_amounts[t] for t in portfolio['Ticker'].unique()) <= initial_value,
            "maintain_total_value_2")

        # Objective: Balance allocation accuracy and tax minimization
        allocation_weight = 100  # Adjust this weight to prioritize allocation accuracy
        model.setObjective(allocation_weight * abs_deviation + tax_paid, GRB.MINIMIZE)

        # Optimize
        model.optimize()

        if model.status == GRB.OPTIMAL:

            actual_tax_paid = sum(
                sell_fractions[i].x * portfolio.loc[i, 'Quantity'] * portfolio.loc[i, 'Capital Gain Per Share'] *
                portfolio.loc[i, 'Tax Rate']
                for i in portfolio.index)

            # st.write(f"Tax paid: ${actual_tax_paid:.2f}")

            # Extract results
            optimized_portfolio = portfolio.copy()
            optimized_portfolio['Sell Fraction'] = [sell_fractions[i].x for i in portfolio.index]
            optimized_portfolio['Sell Quantity'] = np.floor(
                optimized_portfolio['Sell Fraction'] * optimized_portfolio['Quantity'])
            optimized_portfolio['New Quantity'] = optimized_portfolio['Quantity'] - optimized_portfolio['Sell Quantity']
            optimized_portfolio['Sell Value'] = optimized_portfolio['Sell Quantity'] * optimized_portfolio[
                'Current Price']
            optimized_portfolio['New Value'] = optimized_portfolio['New Quantity'] * optimized_portfolio[
                'Current Price']
            optimized_portfolio['Tax Paid'] = optimized_portfolio['Sell Quantity'] * optimized_portfolio[
                'Capital Gain Per Share'] * optimized_portfolio['Tax Rate']

            ticker_prices = portfolio.groupby('Ticker')['Current Price'].first()

            # Handle buying
            buy_results = pd.DataFrame({
                'Ticker': portfolio['Ticker'].unique(),
                'Buy Amount': [buy_amounts[t].x for t in portfolio['Ticker'].unique()]
            })

            ticker_prices = portfolio.groupby('Ticker')['Current Price'].first()
            buy_results['Buy Quantity'] = np.floor(buy_results['Buy Amount'] / buy_results['Ticker'].map(ticker_prices))

            # Merge buy results into optimized portfolio
            optimized_portfolio = optimized_portfolio.merge(
                buy_results[['Ticker', 'Buy Quantity']],
                on='Ticker',
                how='left'
            )

            # Distribute Buy Quantity across lots
            def distribute_buy_quantity(group):
                buy_quantity = group['Buy Quantity'].iloc[0]  # Total buy quantity for this ticker
                if buy_quantity > 0:
                    # Distribute proportionally based on the original quantity of each lot
                    total_original_quantity = group['Quantity'].sum()
                    group['Buy Quantity'] = np.floor(group['Quantity'] / total_original_quantity * buy_quantity)
                    # Adjust for any rounding errors by adding remainder to the largest lot
                    remainder = buy_quantity - group['Buy Quantity'].sum()
                    if remainder > 0:
                        group.loc[group['Quantity'].idxmax(), 'Buy Quantity'] += remainder
                return group

            optimized_portfolio = optimized_portfolio.groupby('Ticker').apply(distribute_buy_quantity)

            optimized_portfolio['Buy Quantity'] = optimized_portfolio['Buy Quantity'].fillna(0)
            optimized_portfolio['Final Quantity'] = optimized_portfolio['New Quantity'] + optimized_portfolio[
                'Buy Quantity']
            optimized_portfolio['Final Value'] = optimized_portfolio['Final Quantity'] * optimized_portfolio[
                'Current Price']

            # optimized_portfolio['Buy Quantity'] = optimized_portfolio['Buy Quantity'].fillna(0)
            # optimized_portfolio['Final Quantity'] = optimized_portfolio['New Quantity'] + optimized_portfolio[
            #     'Buy Quantity']
            # optimized_portfolio['Final Value'] = optimized_portfolio['Final Quantity'] * optimized_portfolio[
            #     'Current Price']

            new_stock_value = optimized_portfolio[optimized_portfolio['Ticker'].str.contains('STOCK')][
                'Final Value'].sum()
            new_total_value = optimized_portfolio['Final Value'].sum()
            new_stock_allocation = new_stock_value / new_total_value
            new_bond_allocation = 1 - new_stock_allocation

            # st.write(f"New stock value: ${new_stock_value:.2f}")
            # st.write(f"New total value: ${new_total_value:.2f}")
            # st.write(f"New stock allocation: {new_stock_allocation:.2%}")
            # st.write(f"New bond allocation: {new_bond_allocation:.2%}")

            total_tax_paid = optimized_portfolio['Tax Paid'].sum()
            # st.write(f"Total tax paid: ${total_tax_paid:.2f}")

            # st.write("Buy Summary")
            # st.write(buy_results[buy_results['Buy Quantity'] > 0])
            buy_summary = buy_results[buy_results['Buy Quantity'] > 0]
            sell_summary = optimized_portfolio[optimized_portfolio['Sell Quantity'] > 0][['Ticker', 'Sell Quantity', 'Sell Value', 'Tax Paid']]
            # st.write("Sell Summary")
            # st.write(optimized_portfolio[optimized_portfolio['Sell Quantity'] > 0][
            #           ['Ticker', 'Sell Quantity', 'Sell Value', 'Tax Paid']])
            return optimized_portfolio, new_stock_allocation, new_bond_allocation, buy_summary, sell_summary
        elif model.status == GRB.INFEASIBLE:
            st.error("The model is infeasible. This could be due to conflicting constraints.")
            model.computeIIS()
            st.write("Conflicting constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    st.write(c.ConstrName)
            return None, None, None, None, None
        else:
            st.error(f"Optimization failed with status code {model.status}.")
            return None, None, None, None, None

    except gp.GurobiError as e:
        st.error(f"Gurobi error: {e}")
        return None, None, None


# Streamlit UI
st.title("Portfolio Optimizer :abacus:")

container = st.container()
with container:
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
                      st.session_state.portfolio[st.session_state.portfolio['Ticker'].str.contains('STOCK')]['Current Price']
        current_stock_allocation = stock_value.sum() / total_value
        current_bond_allocation = 1 - current_stock_allocation

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
        selected_allocation = st.radio("Select desired stock:bond allocation:", list(allocation_options.keys()), index=4)
        target_allocation = allocation_options[selected_allocation]

        # Input for tax rates and max tax burden
        short_term_tax_rate = st.slider("Short-term Tax Rate (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.1) / 100
        long_term_tax_rate = st.slider("Long-term Tax Rate (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1) / 100
        max_tax_burden = st.number_input("Maximum Tax Burden ($)", min_value=0, value=3000)

        # Optimize button
        if st.button("Optimize Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                optimized_portfolio, new_stock_allocation, new_bond_allocation, buy_summary, sell_summary = optimize_portfolio(
                    st.session_state.portfolio, target_allocation, max_tax_burden, short_term_tax_rate, long_term_tax_rate
                )

            if optimized_portfolio is not None:
                col1, col2 = st.columns(2)

                with col1:
                    # Display current allocation as a pie chart
                    fig_current = go.Figure(data=[go.Pie(labels=['Stocks', 'Bonds'], values=[current_stock_allocation, current_bond_allocation])])
                    fig_current.update_layout(title='Current Allocation')
                    st.plotly_chart(fig_current)

                with col2:
                    # Display new allocation as a pie chart
                    fig_new = go.Figure(data=[go.Pie(labels=['Stocks', 'Bonds'], values=[new_stock_allocation, new_bond_allocation])])
                    fig_new.update_layout(title='New Allocation')
                    st.plotly_chart(fig_new)

                st.success("Optimized Portfolio")
                st.write(optimized_portfolio)

                # st.info("Trade Summary :handshake:")
                # trades = optimized_portfolio[optimized_portfolio['Sell Quantity'] > 0][
                #     ['Ticker', 'Sell Quantity', 'Current Price', 'Tax Paid']]
                # st.write(trades)

                st.write("Buy Summary")
                st.write(buy_summary)
                st.write("Sell Summary")
                st.write(sell_summary)
                total_tax_paid = optimized_portfolio['Tax Paid'].sum()
                st.write(f"Total Tax Paid: ${total_tax_paid:,.2f}")

            else:
                st.error("Failed to optimize the portfolio. Please adjust your constraints and try again.")
    else:
        st.info("Please generate a random portfolio or upload an Excel file to begin.")

