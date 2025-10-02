
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web

# Import our custom modules
from src import yield_curve_model
from src import analysis
from src import forecasting

def fetch_and_clean_data():
    """Fetches and prepares the Treasury and macro data from FRED."""
    print("Fetching data from FRED... (This may take a moment)")
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.today()
    
    series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
    
    df = web.DataReader(series_ids, 'fred', start_date, end_date)
    df_cleaned = df.ffill().dropna()
    print("Data successfully fetched and cleaned.")
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
    print("Phase 2: Calibrating Nelson-Siegel Model")
    print("="*50)
    final_params, rmse = yield_curve_model.calibrate_yield_curve(maturities, market_yields)

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
    print("Phase 4: Building Forecasting Model")
    print("="*50)
    forecasting.run_autoregressive_forecast(df_cleaned)
    
    print("\n" + "="*50)
    print("Project Pipeline Complete.")
    print("="*50)

if __name__ == "__main__":
    main()
