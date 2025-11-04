

import pandas as pd
import numpy as np
import datetime

# Import our custom modules
from src import yield_curve_model
from src import analysis
from src import forecasting
import data_manager

def get_data_from_database():
    """
    Get Treasury data from SQLite database with automatic updates.
    """
    print("Accessing Treasury data database...")

    # Check database status
    db_info = data_manager.get_database_info()
    print(f"Database status: {db_info['status']}")

    if db_info['status'] == 'Database not found':
        print("Database not found. Initializing with historical data...")
        df_cleaned = data_manager.initialize_database()
    else:
        print(f"Database found: {db_info['record_count']} records ({db_info['date_range']})")
        print("Checking for updates...")
        df_cleaned = data_manager.refresh_database()

    if df_cleaned is None or df_cleaned.empty:
        raise ValueError("Failed to obtain data from database")

    print(f"Data successfully loaded from database ({len(df_cleaned)} records)")
    return df_cleaned

def main():
    """Main project pipeline."""
    print("="*50)
    print("Starting Fixed Income Analysis Project")
    print("="*50)

    # --- Phase 1: Data ---
    df_cleaned = fetch_and_clean_data()
    
    # Prepare data for calibration (using the most recent day)
    latest_yields = df_cleaned.iloc[-1]
    maturities = np.array([1/12, 3/12, 6/12, 1, 2, 5, 10, 30])
    market_yields = latest_yields.values

    # --- Phase 2: Yield Curve Calibration ---
    print("\n" + "="*50)
    print("Phase 2: Yield Curve Calibration: NS and NSS Models")
    print("="*50)

    # Nelson-Siegel calibration
    print("Calibrating Nelson-Siegel (4-parameter) model...")
    ns_params, ns_rmse = yield_curve_model.calibrate_yield_curve(maturities, market_yields)

    # Nelson-Siegel-Svensson calibration
    print("\nCalibrating Nelson-Siegel-Svensson (6-parameter) model...")
    nss_params, nss_rmse = yield_curve_model.calibrate_svensson_model(maturities, market_yields)

    # Print comparison summary
    print("\n--- Model Comparison Summary ---")
    print(f"Nelson-Siegel RMSE: {ns_rmse:.4f} bps")
    print(f"Nelson-Siegel-Svensson RMSE: {nss_rmse:.4f} bps")

    if nss_rmse < ns_rmse:
        improvement = (ns_rmse - nss_rmse) / ns_rmse * 100
        print(f"NSS model improvement: {improvement:.2f}% better fit")
        final_params = nss_params
        rmse = nss_rmse
        preferred_model = "NSS"
    else:
        print("NS model preferred (better or equal fit)")
        final_params = ns_params
        rmse = ns_rmse
        preferred_model = "NS"

    print(f"Using {preferred_model} model for subsequent analysis")

    # --- Phase 3: Portfolio Analysis ---
    print("\n" + "="*50)
    print("Phase 3: Running Portfolio Analysis")
    print("="*50)
    # Define our sample portfolio
    portfolio_data = {
        'Bond Name': ['T 2.25 08/15/27', 'T 1.75 11/15/29', 'T 3.00 02/15/34', 'T 4.50 05/15/38'],
        'Coupon Rate (%)': [2.25, 1.75, 3.00, 4.50],
        'Years to Maturity': [1.87, 4.12, 8.37, 12.62],
        'Market Price': [96.50, 91.25, 93.00, 101.50]
    }
    portfolio_df = pd.DataFrame(portfolio_data)
    analysis.run_all_analysis(final_params, portfolio_df)

    # --- Phase 4: Forecasting ---
    print("\n" + "="*50)
    print("Phase 4: Building AR and GARCH Forecasting Models")
    print("="*50)
    forecasting.run_autoregressive_forecast(df_cleaned)
    forecasting.run_garch_model(df_cleaned)
    
    print("\n" + "="*50)
    print("Project Pipeline Complete.")
    print("="*50)

if __name__ == "__main__":
    main()
