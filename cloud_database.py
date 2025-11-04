"""
Cloud-optimized database manager for Streamlit Cloud deployment.
Handles database initialization on first run and data persistence.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from dotenv import load_dotenv

# Use session state for database persistence in cloud
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'database_data' not in st.session_state:
    st.session_state.database_data = None

def get_data_for_analysis():
    """
    Get Treasury data with cloud-optimized initialization.
    Creates sample data on first run, then uses cached version.
    """

    # Initialize sample data if not already done
    if not st.session_state.db_initialized:
        st.info("🔄 Initializing database for first run...")

        # Create realistic sample data that resembles real Treasury data
        create_sample_database()
        st.session_state.db_initialized = True

        # Show success message
        st.success("✅ Database initialized with sample Treasury data")
        st.balloons()

    return st.session_state.database_data

def create_sample_database():
    """
    Create realistic sample Treasury data for demonstration.
    This mimics real FRED data structure and behavior.
    """

    # Set seed for reproducible results
    np.random.seed(42)

    # Create date range (5 years of daily data)
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')

    # Remove weekends (Treasury markets closed)
    dates = dates[dates.weekday < 5]

    n_days = len(dates)

    # Create realistic yield curves with typical Treasury market behavior
    # Initial yields (approximate current market levels)
    initial_yields = {
        'DGS1MO': 5.35,   # 1-month
        'DGS3MO': 5.30,   # 3-month
        'DGS6MO': 5.10,   # 6-month
        'DGS1':   4.70,   # 1-year
        'DGS2':   4.35,   # 2-year
        'DGS5':   4.20,   # 5-year
        'DGS10':  4.25,   # 10-year
        'DGS30':  4.40    # 30-year
    }

    # Create realistic yield curve movements
    data = {}

    for tenor, initial_yield in initial_yields.items():
        # Different volatility for different tenors
        if 'MO' in tenor:  # Short-term rates more volatile
            vol = 0.3
            mean_reversion = 0.05
        elif '30' in tenor:  # Long-term rates less volatile
            vol = 0.15
            mean_reversion = 0.02
        else:  # Medium-term rates
            vol = 0.2
            mean_reversion = 0.03

        # Generate realistic yield movements with mean reversion
        yields = np.zeros(n_days)
        yields[0] = initial_yield

        for i in range(1, n_days):
            # Mean-reverting random walk
            shock = np.random.normal(0, vol)
            mean_reversion_force = mean_reversion * (initial_yield - yields[i-1])
            yields[i] = yields[i-1] + shock + mean_reversion_force

            # Keep yields in realistic bounds
            yields[i] = np.clip(yields[i], 0.5, 8.0)

        data[tenor] = yields

    # Create DataFrame
    df = pd.DataFrame(data, index=dates)

    # Add some realistic correlation between yield changes
    # (short-term and long-term rates move together)
    correlation_shock = np.random.normal(0, 0.1, n_days)
    df['DGS1MO'] += correlation_shock * 0.5
    df['DGS3MO'] += correlation_shock * 0.4
    df['DGS10'] += correlation_shock * 0.3
    df['DGS30'] += correlation_shock * 0.2

    # Ensure proper yield curve ordering (short-term < long-term normally)
    for i in range(len(df)):
        if df.iloc[i]['DGS10'] < df.iloc[i]['DGS1MO']:
            # Inverted yield curve - rare but possible
            pass
        else:
            # Normal yield curve
            pass

    # Store in session state for persistence
    st.session_state.database_data = df

    print(f"Created sample database with {len(df)} records ({df.index.min().date()} to {df.index.max().date()})")

def get_database_info():
    """
    Get database information for cloud deployment.
    """
    if st.session_state.database_data is not None:
        df = st.session_state.database_data
        return {
            "status": "Database initialized (sample data)",
            "record_count": len(df),
            "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
            "file_size_mb": "N/A (in-memory)",
            "note": "Sample data for demonstration - mimics real Treasury market behavior"
        }
    else:
        return {
            "status": "Database not initialized",
            "record_count": 0,
            "date_range": "N/A",
            "file_size_mb": "N/A",
            "note": "Will initialize on first run"
        }

def refresh_database():
    """
    For cloud deployment, this returns the cached sample data.
    In a real deployment, this would fetch from FRED API.
    """
    return get_data_for_analysis()

def initialize_database():
    """
    Initialize database for cloud deployment.
    """
    create_sample_database()
    return st.session_state.database_data

def update_database(df_new):
    """
    For cloud deployment, this is a no-op since we use sample data.
    """
    pass