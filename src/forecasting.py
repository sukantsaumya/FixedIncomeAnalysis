
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
