# app.py
"""
An interactive web dashboard for the End-to-End Fixed Income Analysis project.
Powered by Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the backend logic from our 'src' folder
from src import yield_curve_model
from src import analysis
from src import forecasting

# =================================================================
# App Configuration
# =================================================================
st.set_page_config(
    page_title="Fixed Income Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =================================================================
# Data Caching and Loading
# =================================================================
@st.cache_data  # This powerful command caches the data so it doesn't re-run on every interaction
def load_and_calibrate():
    """
    Fetches the latest Treasury data and calibrates the Nelson-Siegel model.
    This function will only run once and its result will be stored.
    """
    # This is a simplified version of our data fetching
    latest_yields_data = {
        'Maturity': np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 5, 10, 30]),
        'Yield': np.array([4.2, 4.02, 3.83, 3.68, 3.6, 3.74, 4.16, 4.73])
    }

    maturities = latest_yields_data['Maturity']
    market_yields = latest_yields_data['Yield']

    final_params, rmse = yield_curve_model.calibrate_yield_curve(maturities, market_yields)

    # Create sample historical data for GARCH demonstration
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    n_days = len(dates)

    # Simulate realistic 10-year yield with volatility
    initial_yield = 4.0
    daily_returns = np.random.normal(0, 0.5, n_days)  # 0.5% daily vol
    price_path = initial_yield + np.cumsum(daily_returns * 0.01)  # Scale down for yield movement
    price_path = np.clip(price_path, 1.0, 8.0)  # Keep yields in realistic range

    # Create DataFrame for GARCH model
    sample_df = pd.DataFrame({
        'DGS10': price_path
    }, index=dates)

    # Run GARCH model on sample data
    conditional_vol, forecast_vol, garch_params = forecasting.run_garch_model(sample_df)

    return final_params, rmse, maturities, market_yields, conditional_vol, forecast_vol, garch_params


# =================================================================
# Main App UI
# =================================================================

st.title("📈 Fixed Income Analysis Dashboard")
st.markdown("An interactive dashboard showcasing the results of the yield curve modeling and risk analysis project.")

# --- Load data and run the main calibration ---
# This is called only once thanks to the cache
final_params, rmse, maturities, market_yields, conditional_vol, forecast_vol, garch_params = load_and_calibrate()

# --- Sidebar for user inputs ---
st.sidebar.header("Risk Scenario Controls")
rate_shock_bps = st.sidebar.slider(
    "Interest Rate Shock (in basis points)",
    min_value=-200,
    max_value=200,
    value=100,  # Default value
    step=10
)

# --- Display Results ---

# Create two columns for a cleaner layout
col1, col2 = st.columns((2, 1.5))

with col1:
    st.header("Yield Curve Calibration")
    st.metric("Final Model RMSE", f"{rmse:.2f} bps",
              help="This is the best achievable RMSE for the Nelson-Siegel model on this data.")

    # Generate and display the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_smooth = np.linspace(0.01, 30, 500)
    yields_smooth = yield_curve_model.nelson_siegel(tau_smooth, *final_params)
    ax.scatter(maturities, market_yields, color='red', s=80, zorder=5, label='Market Data')
    ax.plot(tau_smooth, yields_smooth, 'b-', linewidth=2, label='Fitted Nelson-Siegel Curve')
    ax.set_title(f'Calibrated Yield Curve (RMSE = {rmse:.2f} bps)')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.header("Relative Value Analysis")
    st.markdown("Pricing a sample portfolio to find rich/cheap bonds.")
    # Define our sample portfolio
    portfolio_data = {
        'Bond Name': ['T 2.25 08/15/27', 'T 1.75 11/15/29', 'T 3.00 02/15/34', 'T 4.50 05/15/38'],
        'Coupon Rate (%)': [2.25, 1.75, 3.00, 4.50],
        'Years to Maturity': [1.87, 4.12, 8.37, 12.62],
        'Market Price': [96.50, 91.25, 93.00, 101.50]
    }
    portfolio_df = pd.DataFrame(portfolio_data)

    # Run RV analysis
    rv_df = portfolio_df.copy()
    model_prices = rv_df.apply(
        lambda row: analysis._calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], final_params),
        axis=1
    )
    rv_df['Model Price'] = model_prices
    rv_df['Rich/Cheap ($)'] = rv_df['Model Price'] - rv_df['Market Price']
    st.dataframe(rv_df.round(4))

    # --- Duration Scenario Analysis (Interactive Part) ---
    st.header("Duration & Risk Scenario")
    st.markdown(f"Simulating the impact of a **{rate_shock_bps} bps** parallel rate shock.")

    # Run duration analysis based on the slider value
    total_market_value = portfolio_df['Market Price'].sum()
    mod_durations = portfolio_df.apply(
        lambda row: analysis._calculate_modified_duration(row['Coupon Rate (%)'], row['Years to Maturity'],
                                                          final_params),
        axis=1
    )
    weighted_duration = (mod_durations * portfolio_df['Market Price']).sum() / total_market_value

    # 1. Estimated change
    estimated_change_pct = -weighted_duration * (rate_shock_bps / 10000)
    estimated_new_value = total_market_value * (1 + estimated_change_pct)

    # 2. Actual change
    shocked_params = final_params.copy()
    shocked_params[0] += (rate_shock_bps / 100)
    actual_new_value = portfolio_df.apply(
        lambda row: analysis._calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], shocked_params),
        axis=1
    ).sum()

    # Display the results in columns
    scen_col1, scen_col2 = st.columns(2)
    scen_col1.metric("Estimated New Value", f"${estimated_new_value:,.2f}",
                     f"{(estimated_change_pct * 100):.2f}% change")
    scen_col2.metric("Actual New Value (Re-Priced)", f"${actual_new_value:,.2f}",
                     f"{(actual_new_value / total_market_value - 1) * 100:.2f}% change")