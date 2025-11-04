
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from arch import arch_model

def run_autoregressive_forecast(df_cleaned):
    """
    Builds and evaluates an AR model on the 10-year yield data.
    """
    df_ar = df_cleaned[['DGS10']].copy()
    df_ar['DGS10_lag1'] = df_ar['DGS10'].shift(1)
    df_ar = df_ar.dropna()

    X = df_ar[['DGS10_lag1']]# pylint: disable=invalid-name
    y = df_ar['DGS10']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    ar_model = LinearRegression()
    ar_model.fit(X_train, y_train)
    
    y_pred = ar_model.predict(X_test)
    r_squared_ar = r2_score(y_test, y_pred)

    print("\n--- Autoregressive Model Evaluation ---")
    print(f"R-squared (R²): {r_squared_ar:.4f}")
    print(f"This means the simple AR model explains {r_squared_ar*100:.2f}% of the variance.")

def run_garch_model(df_cleaned):
    """
    Fits a GARCH(1,1) model to the daily returns of the 10-year Treasury yield.
    """
    if len(df_cleaned) < 100:
        print("Warning: Insufficient data for GARCH modeling (need at least 100 observations)")
        return None, None, None

    # Calculate daily log returns in percentage
    returns = np.log(df_cleaned['DGS10'] / df_cleaned['DGS10'].shift(1)) * 100
    returns = returns.dropna()

    if len(returns) < 100:
        print("Warning: Insufficient return data for GARCH modeling")
        return None, None, None

    try:
        # Fit GARCH(1,1) model with normal distribution
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')

        # Extract conditional volatility
        conditional_vol = res.conditional_volatility

        # Generate 30-day volatility forecast
        forecast = res.forecast(horizon=30, start=conditional_vol.index[-1])
        forecast_vol = np.sqrt(forecast.variance.iloc[-1].values)

        # Calculate volatility statistics
        vol_stats = {
            'mean': conditional_vol.mean(),
            'std': conditional_vol.std(),
            'min': conditional_vol.min(),
            'max': conditional_vol.max(),
            'current': conditional_vol.iloc[-1]
        }

        print("\n--- GARCH Volatility Model Results ---")
        print(f"Model Parameters:")
        print(f"  Omega (constant): {res.params['omega']:.6f}")
        print(f"  Alpha (ARCH): {res.params['alpha[1]']:.6f}")
        print(f"  Beta (GARCH): {res.params['beta[1]']:.6f}")
        print(f"  Persistence (α + β): {res.params['alpha[1]'] + res.params['beta[1]']:.6f}")
        print(f"\nVolatility Statistics (%):")
        print(f"  Current Volatility: {vol_stats['current']:.4f}")
        print(f"  Average Volatility: {vol_stats['mean']:.4f}")
        print(f"  Volatility Range: {vol_stats['min']:.4f} - {vol_stats['max']:.4f}")
        print(f"\n30-Day Volatility Forecast Range: {forecast_vol.min():.4f} - {forecast_vol.max():.4f}")

        return conditional_vol, forecast_vol, res.params

    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        print("Falling back to exponential weighted moving average volatility")

        # Fallback: EWMA volatility
        ewm_vol = returns.ewm(span=30).std()
        forecast_vol = np.full(30, ewm_vol.iloc[-1])

        vol_stats = {
            'mean': ewm_vol.mean(),
            'std': ewm_vol.std(),
            'min': ewm_vol.min(),
            'max': ewm_vol.max(),
            'current': ewm_vol.iloc[-1]
        }

        print(f"\nEWMA Volatility Statistics (%):")
        print(f"  Current Volatility: {vol_stats['current']:.4f}")
        print(f"  Average Volatility: {vol_stats['mean']:.4f}")
        print(f"  Volatility Range: {vol_stats['min']:.4f} - {vol_stats['max']:.4f}")

        return ewm_vol, forecast_vol, None
