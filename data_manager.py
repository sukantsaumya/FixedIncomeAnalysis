"""
Database manager for Fixed Income Analysis project.
Handles SQLite database operations and incremental data updates from FRED.
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import os
from dotenv import load_dotenv


def create_database():
    """Initialize SQLite database with treasury_data table."""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    db_path = 'data/treasury_data.db'

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with proper schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS treasury_data (
        date DATE PRIMARY KEY,
        DGS1MO REAL,
        DGS3MO REAL,
        DGS6MO REAL,
        DGS1 REAL,
        DGS2 REAL,
        DGS5 REAL,
        DGS10 REAL,
        DGS30 REAL
    )
    ''')

    conn.commit()
    conn.close()
    print(f"Database created at {db_path}")


def get_latest_date():
    """Query database for most recent data date."""
    db_path = 'data/treasury_data.db'

    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(date) FROM treasury_data")
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        return datetime.datetime.strptime(result[0], '%Y-%m-%d').date()
    return None


def fetch_new_data(start_date, end_date):
    """Fetch Treasury data from FRED for specified date range."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError("FRED_API_KEY not found. Please create a .env file with your key.")

    series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']

    print(f"Fetching new data from FRED ({start_date} to {end_date})...")
    df = web.DataReader(series_ids, 'fred', start_date, end_date, api_key=api_key)

    return df


def update_database(df_new):
    """Append new data to existing database."""
    db_path = 'data/treasury_data.db'

    conn = sqlite3.connect(db_path)

    # Use INSERT OR REPLACE to handle duplicates
    df_new.to_sql('treasury_data', conn, if_exists='append', index_label='date',
                  method='multi', chunksize=1000)

    conn.commit()
    conn.close()
    print(f"Database updated with {len(df_new)} new records")


def get_all_data():
    """Retrieve complete dataset from database for analysis."""
    db_path = 'data/treasury_data.db'

    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)

    # Read data with date as index
    df = pd.read_sql_query("SELECT * FROM treasury_data ORDER BY date", conn,
                         index_col='date', parse_dates=['date'])

    conn.close()
    return df


def initialize_database():
    """Create database structure and fetch historical data (2010-start)."""
    print("Initializing database with historical data...")

    # Create database structure
    create_database()

    # Fetch historical data from 2010 to present
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.today()

    try:
        df_historical = fetch_new_data(start_date, end_date)
        df_cleaned = df_historical.ffill().dropna()

        # Update database with historical data
        update_database(df_cleaned)

        return df_cleaned

    except Exception as e:
        print(f"Error initializing database: {e}")
        return None


def refresh_database():
    """Incremental update with new data only."""
    latest_date = get_latest_date()

    if latest_date is None:
        print("No existing data found. Initializing database...")
        return initialize_database()

    # Check if we need to update (if latest data is older than today)
    today = datetime.datetime.today().date()
    update_threshold = latest_date + datetime.timedelta(days=1)

    if today <= update_threshold:
        print("Database is up to date.")
        return get_all_data()

    # Fetch new data from latest_date + 1 to today
    start_date = latest_date + datetime.timedelta(days=1)
    end_date = today

    try:
        df_new = fetch_new_data(start_date, end_date)

        if df_new.empty:
            print("No new data available from FRED")
            return get_all_data()

        df_cleaned = df_new.ffill().dropna()

        if not df_cleaned.empty:
            update_database(df_cleaned)

        return get_all_data()

    except Exception as e:
        print(f"Error refreshing database: {e}")
        print("Using existing database data")
        return get_all_data()


def get_data_for_analysis():
    """Return pandas DataFrame with datetime index for analysis."""
    df = get_all_data()

    if df is None:
        print("No database data available. Please initialize database first.")
        return None

    if len(df) < 260:  # At least 1 year of trading data
        print("Warning: Database has less than 1 year of data")

    print(f"Loaded {len(df)} records from database (Date range: {df.index.min().date()} to {df.index.max().date()})")
    return df


def get_database_info():
    """Return metadata about the database."""
    db_path = 'data/treasury_data.db'

    if not os.path.exists(db_path):
        return {"status": "Database not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get record count and date range
    cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM treasury_data")
    result = cursor.fetchone()

    conn.close()

    return {
        "status": "Database exists",
        "record_count": result[0],
        "date_range": f"{result[1]} to {result[2]}",
        "file_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2)
    }


def backup_database():
    """Create a backup of the database."""
    import shutil
    from datetime import datetime

    db_path = 'data/treasury_data.db'

    if not os.path.exists(db_path):
        print("No database to backup")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'data/treasury_data_backup_{timestamp}.db'

    try:
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to {backup_path}")
        return True
    except Exception as e:
        print(f"Error backing up database: {e}")
        return False