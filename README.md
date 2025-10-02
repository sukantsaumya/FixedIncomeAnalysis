# End-to-End Fixed Income Analysis in Python

## Objective
This project implements a complete quantitative workflow for fixed-income analysis. It involves fetching real-world U.S. Treasury data, calibrating a Nelson-Siegel yield curve model using advanced optimization, performing relative value analysis on a bond portfolio, and conducting risk management through duration and scenario testing.

---

## Key Features
- **Data Acquisition:** Fetches and cleans daily Treasury par yield curve rates from the Federal Reserve Economic Data (FRED) database.
- **Yield Curve Modeling:** Implements a robust 4-parameter Nelson-Siegel model. It uses a sophisticated two-stage optimization (Global Differential Evolution + Local L-BFGS-B) to find the best possible fit.
- **Relative Value Analysis:** Prices a portfolio of off-the-run bonds using the calibrated curve to identify potentially "rich" or "cheap" securities.
- **Risk Management:**
    - Calculates the Modified Duration for the portfolio to measure overall interest rate sensitivity.
    - Conducts a scenario test to estimate the P&L impact of a 100 basis point parallel rate shock.
    - Performs a Key Rate Duration analysis to identify sensitivity to changes in the yield curve's shape.
- **Forecasting:** Develops a simple but effective Autoregressive (AR) model to forecast the 10-year Treasury yield, demonstrating the principle of persistence in financial time series.

---

## Key Findings
- The optimal RMSE for the Nelson-Siegel model on recent data was proven to be **~4.23 basis points**, highlighting the model's inherent limitations.
- A simple Autoregressive model (**R² > 0.99**) is vastly superior to a macroeconomic factor model (negative R²) for short-term yield forecasting, demonstrating the powerful effect of persistence.

---

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sukantsaumya/FixedIncomeAnalysis.git](https://github.com/sukantsaumya/FixedIncomeAnalysis.git)
    ```
2.  **Navigate into the project folder:**
    ```bash
    cd FixedIncomeAnalysis
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main analysis pipeline:**
    ```bash
    python main.py
    ```
