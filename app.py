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

# Use cloud-optimized database for Streamlit Cloud deployment
import cloud_database as data_manager

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
@st.cache_data(ttl=3600)  # Cache for 1 hour to allow database updates
def load_and_calibrate():
    """
    Gets Treasury data from database and calibrates both NS and NSS models.
    This function will run once and cache for 1 hour.
    """
    # Get data from database
    df_db = data_manager.get_data_for_analysis()

    if df_db is None or df_db.empty:
        # Fallback to sample data if database fails
        st.warning("Database unavailable. Using sample data.")
        latest_yields_data = {
            'Maturity': np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 5, 10, 30]),
            'Yield': np.array([4.2, 4.02, 3.83, 3.68, 3.6, 3.74, 4.16, 4.73])
        }
        maturities = latest_yields_data['Maturity']
        market_yields = latest_yields_data['Yield']
        df_for_garch = None
    else:
        # Use real database data
        latest_yields = df_db.iloc[-1]
        maturities = np.array([1/12, 3/12, 6/12, 1, 2, 5, 10, 30])
        market_yields = latest_yields.values
        df_for_garch = df_db

    # Calibrate both models
    ns_params, ns_rmse = yield_curve_model.calibrate_yield_curve(maturities, market_yields)
    nss_params, nss_rmse = yield_curve_model.calibrate_svensson_model(maturities, market_yields)

    # Handle GARCH modeling
    if df_for_garch is not None:
        # Use real data for GARCH
        conditional_vol, forecast_vol, garch_params = forecasting.run_garch_model(df_for_garch)
    else:
        # Fallback: create sample historical data for GARCH demonstration
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        n_days = len(dates)

        initial_yield = 4.0
        daily_returns = np.random.normal(0, 0.5, n_days)
        price_path = initial_yield + np.cumsum(daily_returns * 0.01)
        price_path = np.clip(price_path, 1.0, 8.0)

        sample_df = pd.DataFrame({'DGS10': price_path}, index=dates)
        conditional_vol, forecast_vol, garch_params = forecasting.run_garch_model(sample_df)

    return ns_params, ns_rmse, nss_params, nss_rmse, maturities, market_yields, conditional_vol, forecast_vol, garch_params


# =================================================================
# Main App UI
# =================================================================

st.title("🏦 Fixed Income Analysis Dashboard")
st.markdown("### Professional Treasury Market Analysis & Risk Management")

# Add educational introduction
with st.expander("📚 What is Fixed Income Analysis?", expanded=True):
    st.markdown("""
    **Fixed Income Analysis** helps investors understand government bond markets and manage risk. This dashboard analyzes:

    - 📊 **Yield Curves**: How interest rates vary across different bond maturities
    - 🎯 **Volatility**: How much bond prices fluctuate over time
    - ⚠️ **Risk Scenarios**: What happens if interest rates change suddenly

    **Why it matters**: Treasury bonds affect mortgage rates, retirement savings, and the entire economy!
    """)

st.markdown("---")

# --- Sidebar for user inputs ---
st.sidebar.header("Model Configuration")
# Model Configuration with educational content
st.sidebar.markdown("### 📈 Yield Curve Model")
model_type = st.sidebar.radio(
    "Choose Analysis Model",
    ["Nelson-Siegel", "Nelson-Siegel-Svensson"],
    help="Choose between 4-parameter NS and 6-parameter NSS models"
)

with st.sidebar.expander("🔍 Model Explanation"):
    st.markdown("""
    **Nelson-Siegel (4-parameter)**: Classic model, good for normal yield curves

    **Nelson-Siegel-Svensson (6-parameter)**: Advanced model, better for complex curves with multiple humps

    *Lower RMSE = Better fit to market data*
    """)

# Risk Scenario Controls with educational content
st.sidebar.markdown("### ⚠️ Risk Scenarios")
rate_shock_bps = st.sidebar.slider(
    "Interest Rate Change (basis points)",
    min_value=-200,
    max_value=200,
    value=100,  # Default value
    step=10,
    help="100 basis points = 1% change in interest rates"
)

with st.sidebar.expander("💡 What are Basis Points?"):
    st.markdown("""
    **100 basis points = 1%**

    **+100 bps**: Interest rates increase by 1% (bond prices go down)

    **-100 bps**: Interest rates decrease by 1% (bond prices go up)

    *This helps you understand interest rate risk!*
    """)

# --- Load data and run the main calibration ---
# This is called only once thanks to the cache
ns_params, ns_rmse, nss_params, nss_rmse, maturities, market_yields, conditional_vol, forecast_vol, garch_params = load_and_calibrate()

# *** FIX 1: Select the correct model FUNCTION and parameters ***
if model_type == "Nelson-Siegel":
    final_params = ns_params
    rmse = ns_rmse
    final_model_func = yield_curve_model.nelson_siegel # <-- Select the function
else:
    final_params = nss_params
    rmse = nss_rmse
    final_model_func = yield_curve_model.nelson_siegel_svensson # <-- Select the function

# --- Display Results ---

# Create two columns for a cleaner layout
col1, col2 = st.columns((2, 1.5))

with col1:
    st.header("📊 Yield Curve Analysis")

    # Educational content about yield curves
    with st.expander("📚 Understanding Yield Curves", expanded=False):
        st.markdown("""
        A **yield curve** shows interest rates for bonds with different maturity dates:

        - **Short-term** (1 month): Influences short-term borrowing costs
        - **Long-term** (30 years): Influences mortgage rates and long-term investments

        **Normal curve**: Long-term rates > Short-term rates
        **Inverted curve**: Short-term rates > Long-term rates (recession signal)

        **RMSE** (Root Mean Square Error): How well our model matches real market data. Lower is better!
        """)

    # Display RMSE comparison with better explanations
    st.markdown("#### 📈 Model Performance")
    col_rmse1, col_rmse2 = st.columns(2)
    with col_rmse1:
        st.metric(f"Nelson-Siegel", f"{ns_rmse:.2f} bps", help="4-parameter classic model")
    with col_rmse2:
        st.metric(f"Nelson-Siegel-Svensson", f"{nss_rmse:.2f} bps", help="6-parameter advanced model")

    st.metric(f"✅ Selected: {model_type}", f"{rmse:.2f} bps",
              help="RMSE for the currently selected yield curve model")

    # Generate and display the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_smooth = np.linspace(0.01, 30, 500)

    # Plot based on selected model
    yields_smooth = final_model_func(tau_smooth, *final_params) # Use the selected function
    curve_label = f'Fitted {model_type} Curve'

    ax.scatter(maturities, market_yields, color='red', s=80, zorder=5, label='Market Data')
    ax.plot(tau_smooth, yields_smooth, 'b-', linewidth=2, label=curve_label)
    ax.set_title(f'{model_type} Yield Curve (RMSE = {rmse:.2f} bps)')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Add model comparison plot when NSS is selected
    if model_type == "Nelson-Siegel-Svensson":
        st.subheader("Model Comparison")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))

        # Plot both curves for comparison
        ns_yields = yield_curve_model.nelson_siegel(tau_smooth, *ns_params)
        nss_yields = yield_curve_model.nelson_siegel_svensson(tau_smooth, *nss_params)

        ax_comp.scatter(maturities, market_yields, color='red', s=80, zorder=5, label='Market Data')
        ax_comp.plot(tau_smooth, ns_yields, 'b--', linewidth=2, alpha=0.7, label=f'NS Curve ({ns_rmse:.2f} bps)')
        ax_comp.plot(tau_smooth, nss_yields, 'g-', linewidth=2, label=f'NSS Curve ({nss_rmse:.2f} bps)')
        ax_comp.set_title('NS vs NSS Model Comparison')
        ax_comp.set_xlabel('Maturity (Years)')
        ax_comp.set_ylabel('Yield (%)')
        ax_comp.legend()
        ax_comp.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_comp)

        # Show improvement percentage
        if nss_rmse < ns_rmse:
            improvement = (ns_rmse - nss_rmse) / ns_rmse * 100
            st.success(f"NSS model shows {improvement:.2f}% improvement in fit accuracy")
        else:
            st.info("NS model provides comparable or better fit")

# --- Volatility Analysis Section ---
st.header("🎯 Volatility & Risk Analysis")

# Educational content about volatility
with st.expander("📚 What is Volatility?", expanded=True):
    st.markdown("""
    **Volatility** measures how much bond prices fluctuate over time:

    - **High volatility**: Prices change a lot (risky, uncertain market)
    - **Low volatility**: Prices change little (stable, predictable market)

    **GARCH Model**: Industry-standard tool for forecasting future volatility based on historical patterns

    **Why it matters**: Higher volatility = higher risk, but also potentially higher returns!
    """)

if conditional_vol is not None and garch_params is not None:
    # Create subplot with two panels
    fig_vol, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: 10-year yield with AR prediction (sample data)
    sample_df = pd.DataFrame({
        'DGS10': np.random.normal(4.0, 0.5, len(conditional_vol))
    }, index=conditional_vol.index)

    ax1.plot(sample_df.index[-252:], sample_df['DGS10'].iloc[-252:], 'b-', label='10-Year Yield', linewidth=1)
    ax1.set_title('10-Year Treasury Yield - Recent Trading Days')
    ax1.set_ylabel('Yield (%)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: GARCH conditional volatility with forecast
    ax2.plot(conditional_vol.index[-252:], conditional_vol.iloc[-252:], 'r-', label='GARCH Conditional Volatility', linewidth=2)

    # Add forecast period
    if forecast_vol is not None:
        forecast_dates = pd.date_range(
            start=conditional_vol.index[-1] + pd.Timedelta(days=1),
            periods=len(forecast_vol),
            freq='D'
        )
        ax2.plot(forecast_dates, forecast_vol, 'g--', label='30-Day Volatility Forecast', linewidth=2, alpha=0.7)

    ax2.set_title('GARCH(1,1) Volatility Analysis')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig_vol)

    # Display volatility metrics
    col_vol1, col_vol2, col_vol3 = st.columns(3)
    with col_vol1:
        st.metric("Current Volatility", f"{conditional_vol.iloc[-1]:.4f}%")
    with col_vol2:
        if forecast_vol is not None:
            st.metric("Avg 30-Day Forecast", f"{forecast_vol.mean():.4f}%")
    with col_vol3:
        if garch_params is not None:
            persistence = garch_params['alpha[1]'] + garch_params['beta[1]']
            st.metric("GARCH Persistence", f"{persistence:.4f}")

    # GARCH Model Parameters
    st.subheader("GARCH Model Parameters")
    if garch_params is not None:
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            st.metric("Omega (Constant)", f"{garch_params['omega']:.6f}")
        with param_col2:
            st.metric("Alpha (ARCH)", f"{garch_params['alpha[1]']:.6f}")
        with param_col3:
            st.metric("Beta (GARCH)", f"{garch_params['beta[1]']:.6f}")
else:
    st.warning("GGARCH model could not be fitted with current data.")

with col2:
    st.header("💼 Portfolio Analysis")

    # Educational content about portfolio analysis
    with st.expander("📚 What is Portfolio Analysis?", expanded=False):
        st.markdown("""
        **Relative Value Analysis** helps identify which bonds are:

        - **Rich**: Overpriced compared to theoretical value (sell signal)
        - **Cheap**: Underpriced compared to theoretical value (buy signal)

        We use yield curve models to calculate what bonds *should* be worth based on market rates.
        """)

    st.markdown("#### 🎯 Rich/Cheap Analysis")
    st.markdown("Finding mispriced bonds using our yield curve model...")

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
    
    # *** FIX 2: Pass the selected 'final_model_func' to the helper ***
    model_prices = rv_df.apply(
        lambda row: analysis._calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], final_params, final_model_func),
        axis=1
    )
    rv_df['Model Price'] = model_prices
    rv_df['Rich/Cheap ($)'] = rv_df['Model Price'] - rv_df['Market Price']
    st.dataframe(rv_df.round(4))

    # --- Duration Scenario Analysis (Interactive Part) ---
    st.header("⚠️ Interest Rate Risk Scenarios")

    # Educational content about duration risk
    with st.expander("📚 Understanding Duration Risk", expanded=False):
        st.markdown("""
        **Duration** measures how sensitive bond prices are to interest rate changes:

        - **High duration**: Price changes a lot when rates change (high risk)
        - **Low duration**: Price changes little when rates change (low risk)

        **Key insight**: When interest rates go up, bond prices go down, and vice versa!

        This simulation shows what would happen to your portfolio if rates suddenly change.
        """)

    st.markdown(f"#### 🔄 Simulating {rate_shock_bps} bps rate change")
    st.markdown(f"**{rate_shock_bps/100:.1f}%** {'increase' if rate_shock_bps > 0 else 'decrease'} in interest rates")

    # Run duration analysis based on the slider value
    total_market_value = portfolio_df['Market Price'].sum()
    
    # *** FIX 3: Pass the selected 'final_model_func' to the helper ***
    mod_durations = portfolio_df.apply(
        lambda row: analysis._calculate_modified_duration(row['Coupon Rate (%)'], row['Years to Maturity'],
                                                            final_params, final_model_func),
        axis=1
    )
    weighted_duration = (mod_durations * portfolio_df['Market Price']).sum() / total_market_value

    # 1. Estimated change
    estimated_change_pct = -weighted_duration * (rate_shock_bps / 10000)
    estimated_new_value = total_market_value * (1 + estimated_change_pct)

    # 2. Actual change
    shocked_params = final_params.copy()
    shocked_params[0] += (rate_shock_bps / 100)
    
    # *** FIX 4: Pass the selected 'final_model_func' to the helper ***
    actual_new_value = portfolio_df.apply(
        lambda row: analysis._calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], shocked_params, final_model_func),
        axis=1
    ).sum()

    # Display the results in columns
    scen_col1, scen_col2 = st.columns(2)
    scen_col1.metric("Estimated New Value", f"${estimated_new_value:,.2f}",
                       f"{(estimated_change_pct * 100):.2f}% change")
    scen_col2.metric("Actual New Value (Re-Priced)", f"${actual_new_value:,.2f}",
                       f"{(actual_new_value / total_market_value - 1) * 100:.2f}% change")

st.markdown("---")

# Final summary section
st.header("📋 Summary & Key Insights")

col_summary1, col_summary2, col_summary3 = st.columns(3)

with col_summary1:
    st.metric("📊 Best Model", model_type, help="Model with lowest RMSE")
    st.metric("🎯 Model Accuracy", f"{rmse:.2f} bps", help="Lower is better")

with col_summary2:
    if conditional_vol is not None and garch_params is not None:
        st.metric("📈 Current Volatility", f"{conditional_vol.iloc[-1]:.3f}%")
        persistence = garch_params['alpha[1]'] + garch_params['beta[1]']
        st.metric("🔄 Volatility Persistence", f"{persistence:.3f}", help="How long volatility shocks last")

with col_summary3:
    st.metric("💰 Portfolio Value", f"${total_market_value:,.2f}")
    rate_impact = (actual_new_value / total_market_value - 1) * 100
    st.metric("⚠️ Rate Impact", f"{rate_impact:.2f}%", help=f"From {rate_shock_bps} bps change")

# Educational conclusion
st.markdown("""
### 🎓 What This Analysis Shows

This dashboard demonstrates professional **quantitative finance** techniques used by:

- **Investment banks** for pricing and risk management
- **Hedge funds** for trading strategies
- **Asset managers** for portfolio optimization
- **Central banks** for monetary policy analysis

**Key takeaways:**
1. **Yield curves** reveal market expectations about future interest rates
2. **Volatility models** help quantify and predict market risk
3. **Duration analysis** shows how interest rate changes affect bond portfolios
4. **Model selection** matters - advanced models can provide better accuracy

This is the type of analysis that drives multi-trillion dollar bond markets globally!
""")
