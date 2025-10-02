# Fixed Income Analysis: Yield Curve Modeling & Relative Value

A comprehensive quantitative framework for U.S. Treasury market analysis, implementing industry-standard yield curve modeling, relative value identification, and risk analytics. Built with production-grade code structure and interactive visualization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project demonstrates a complete end-to-end quantitative workflow for fixed-income securities analysis:

1. **Market Data Integration**: Real-time Treasury par yield curve data via FRED API
2. **Yield Curve Calibration**: Nelson-Siegel model with advanced two-stage optimization
3. **Relative Value Analytics**: Off-the-run bond pricing to identify mispricing opportunities
4. **Risk Management**: Duration, scenario analysis, and key rate sensitivity
5. **Forecasting**: Time series modeling of 10-year Treasury yields
6. **Interactive Dashboard**: Streamlit application for real-time analysis

**Target Use Case**: Quantitative research for fixed-income trading desks, risk management teams, and portfolio analytics.

---

## 🔑 Key Features & Results

### Yield Curve Modeling
- **Model**: 4-parameter Nelson-Siegel with level, slope, curvature, and decay components
- **Optimization**: Two-stage approach (Differential Evolution → L-BFGS-B) for global optimum
- **Performance**: Achieved **4.23 basis points RMSE** on recent data, representing the model's practical accuracy limit
- **Validation**: Cross-validated against market quotes across 11 Treasury maturities

### Relative Value Analysis
- Prices off-the-run bonds using calibrated yield curve
- Identifies "rich" (overpriced) and "cheap" (underpriced) securities
- Quantifies mispricing in basis points relative to fair value

### Risk Analytics
- **Modified Duration**: Portfolio-level interest rate sensitivity
- **Scenario Testing**: P&L impact under ±100 bps parallel yield shifts
- **Key Rate Duration**: Granular sensitivity to specific curve segments (2Y, 5Y, 10Y, 30Y)

### Forecasting Insights
- **Key Finding**: Simple AR(1) model (R² > 0.99) dramatically outperforms macroeconomic factor models (negative R²) for short-term yield forecasting
- **Implication**: Yield persistence dominates fundamental factors in near-term prediction
- **Practical Application**: Demonstrates why mean-reversion models often fail in trending rate environments

---

## 📁 Project Structure

```
FixedIncomeAnalysis/
│
├── src/                          # Core analysis modules
│   ├── __init__.py
│   ├── data_acquisition.py       # FRED API client and data cleaning
│   ├── yield_curve.py            # Nelson-Siegel model implementation
│   ├── bond_pricing.py           # Bond valuation and relative value
│   ├── risk_metrics.py           # Duration, convexity, scenario analysis
│   └── forecasting.py            # Time series models (AR, macro factors)
│
├── .github/workflows/            # CI/CD pipeline
│   └── pylint.yml                # Automated code quality checks
│
├── main.py                       # Main analysis pipeline (batch execution)
├── app.py                        # Streamlit dashboard (interactive UI)
├── requirements.txt              # Python dependencies with pinned versions
├── .env.example                  # Template for environment variables
├── .gitignore                    # Version control exclusions
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- FRED API Key (free from [Federal Reserve Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sukantsaumya/FixedIncomeAnalysis.git
   cd FixedIncomeAnalysis
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your FRED API key:
   # FRED_API_KEY=your_api_key_here
   ```

### Running the Analysis

**Option 1: Batch Analysis Pipeline**
```bash
python main.py
```
This executes the complete workflow and outputs results to console and/or files.

**Option 2: Interactive Dashboard**
```bash
streamlit run app.py
```
Launches a web interface at `http://localhost:8501` with:
- Real-time yield curve visualization
- Interactive parameter adjustment for Nelson-Siegel model
- Dynamic relative value screening
- Scenario analysis tools

---

## 📊 Methodology

### 1. Data Acquisition
- **Source**: U.S. Treasury par yield curve rates from FRED
- **Frequency**: Daily observations
- **Maturities**: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
- **Cleaning**: Handles missing data, weekends, holidays via forward-fill interpolation

### 2. Nelson-Siegel Model

The yield at maturity τ is modeled as:

```
y(τ) = β₀ + β₁ * [(1 - exp(-τ/λ)) / (τ/λ)] + β₂ * [(1 - exp(-τ/λ)) / (τ/λ) - exp(-τ/λ)]
```

Where:
- **β₀**: Level (long-term rate)
- **β₁**: Slope (short vs. long rates)
- **β₂**: Curvature (medium-term hump)
- **λ**: Decay parameter (controls where curvature peaks)

**Calibration Process**:
1. **Stage 1**: Differential Evolution for global parameter search across wide bounds
2. **Stage 2**: L-BFGS-B refinement from global optimum
3. **Objective**: Minimize RMSE between model and market yields

**Known Limitations**: The model assumes smooth curves and cannot capture localized anomalies (e.g., on-the-run/off-the-run spreads, liquidity premiums). RMSE of 4.23 bps represents irreducible error from these factors.

### 3. Bond Pricing & Relative Value

- **Discounting**: Zero-coupon rates extracted from Nelson-Siegel par curve via bootstrapping
- **Pricing Formula**: Standard DCF of coupon payments + principal
- **Mispricing Signal**: (Market Price - Model Price) / Model Price
- **Threshold**: Securities with |mispricing| > 10 bps flagged for review

### 4. Risk Metrics

**Modified Duration**:
```
D_mod = -[dP/dY] * [1/P]
```
Measures percentage price change for 1 bp yield shift.

**Key Rate Duration**:
Partial derivative of portfolio value with respect to specific maturity points, holding other rates constant. Reveals exposure to curve reshaping (steepening/flattening).

**Scenario Analysis**:
Revalues portfolio under shocked yield curves (parallel shifts, twists, butterfly).

### 5. Forecasting

**AR(p) Model**:
```
y_t = α + Σ(β_i * y_{t-i}) + ε_t
```

**Macro Factor Model** (tested but underperformed):
```
y_t = α + β₁*GDP_growth + β₂*Inflation + β₃*Fed_Funds + ε_t
```

**Result**: AR(1) R² = 0.994 vs. Macro R² = -0.12, confirming that yield levels are better predictors of near-term yields than economic fundamentals.

---

## 📈 Sample Output

```
=== Yield Curve Calibration ===
Optimization converged in 142 iterations
RMSE: 4.23 basis points
Parameters:
  β₀ (Level):     4.125%
  β₁ (Slope):    -1.234%
  β₂ (Curvature): -0.876%
  λ (Decay):      2.341 years

=== Relative Value Analysis ===
Bond: T 2.5% 08/15/2029 (Off-the-run)
  Market YTM:     3.456%
  Model YTM:      3.421%
  Mispricing:     +3.5 bps (RICH)
  
Bond: T 3.125% 05/15/2033 (Off-the-run)
  Market YTM:     3.789%
  Model YTM:      3.812%
  Mispricing:     -2.3 bps (CHEAP)

=== Risk Metrics ===
Portfolio Modified Duration: 6.42 years
100 bps parallel shock P&L: -$642,000 per $10MM notional

Key Rate Durations:
  2Y:  0.34
  5Y:  1.89
  10Y: 2.76
  30Y: 1.43
```

---

## 🧪 Testing & Validation

While this project demonstrates a complete workflow, production deployment would require:

- **Unit Tests**: Validate Nelson-Siegel pricing against Bloomberg/Reuters benchmarks
- **Integration Tests**: End-to-end pipeline execution with mock data
- **Backtesting**: Out-of-sample forecasting performance over rolling windows
- **Edge Cases**: Handling of inverted curves, negative yields, data gaps

**Current Status**: Manual validation performed; automated test suite in development.

---

## 🛠️ Technology Stack

- **Core**: Python 3.9+, NumPy, Pandas
- **Optimization**: SciPy (Differential Evolution, L-BFGS-B)
- **Visualization**: Matplotlib, Plotly
- **Dashboard**: Streamlit
- **Data**: FRED API (via `fredapi` package)
- **Time Series**: Statsmodels (AR, ARIMA)
- **Dev Tools**: Pylint, Black (code formatting)

---

## 📚 References & Further Reading

1. **Nelson, C.R. and Siegel, A.F. (1987)**: "Parsimonious Modeling of Yield Curves", *Journal of Business*
2. **Diebold, F.X. and Li, C. (2006)**: "Forecasting the Term Structure of Government Bond Yields", *Journal of Econometrics*
3. **Fabozzi, F.J. (2021)**: *Bond Markets, Analysis, and Strategies* (10th Edition)
4. **Federal Reserve**: [Treasury Yield Curve Methodology](https://www.federalreserve.gov/pubs/feds/2006/200628/200628abs.html)

---

## 🔮 Future Enhancements

- [ ] **Extended Models**: Implement Svensson, cubic spline, and kernel-smoothed curves for comparison
- [ ] **Factor Analysis**: PCA decomposition into level/slope/curvature factors
- [ ] **Credit Spreads**: Extend framework to corporate bonds with OAS analysis
- [ ] **Real-Time Updates**: WebSocket integration for live FRED data streaming
- [ ] **Portfolio Optimization**: Mean-variance optimization under duration constraints
- [ ] **Machine Learning**: Ensemble forecasting combining AR with gradient boosting

---

## 👤 Author

**Sukant Saumya**
- GitHub: [@sukantsaumya](https://github.com/sukantsaumya)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/sukantsaumya)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Federal Reserve Economic Data (FRED) for market data access
- NumPy/SciPy communities for robust numerical libraries
- Streamlit for enabling rapid dashboard prototyping

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended to provide investment advice or trading recommendations. Past performance of models does not guarantee future results. Always consult qualified professionals before making investment decisions.
