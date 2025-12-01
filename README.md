# Fixed Income Analysis Dashboard

A professional quantitative finance application featuring advanced yield curve modeling, volatility analysis, and production-grade data management.

## 🚀 Live Demo
[View Live Dashboard](https://fixedincomeanalysis.streamlit.app/)

## ✨ Key Features

### 📊 Advanced Financial Models
- **GARCH(1,1) Volatility Modeling**: Industry-standard risk analysis with 30-day forecasts
- **Nelson-Siegel-Svensson**: 6-parameter yield curve model with direct NS comparison
- **Autoregressive Forecasting**: Time-series prediction for Treasury yields

### 🗄️ Production Data Pipeline
- **SQLite Database Integration**: Incremental updates from FRED API
- **Automated Data Management**: Smart caching and error handling
- **Real-time Treasury Data**: Live market data integration

### 📈 Interactive Dashboard
- **Model Selection**: Toggle between NS and NSS yield curve models
- **Volatility Analysis**: Dual-panel charts with forecasting
- **Risk Scenarios**: Interactive rate shock simulations
- **Performance Metrics**: RMSE comparisons and improvement tracking

## 🛠️ Technologies Used

- **Quantitative Finance**: GARCH models, Nelson-Siegel-Svensson, AR forecasting
- **Data Engineering**: SQLite, pandas-datareader, data pipelines
- **Machine Learning**: scipy optimization, scikit-learn, arch library
- **Web Development**: Streamlit, matplotlib, interactive visualizations
- **API Integration**: FRED (Federal Reserve Economic Data)

## 📋 Installation

### Local Development
```bash
git clone https://github.com/yourusername/FixedIncomeAnalysis.git
cd FixedIncomeAnalysis
pip install -r requirements.txt
streamlit run app.py
```

### Main Pipeline
```bash
python main.py
```

## 🔧 Configuration

1. Create a `.env` file with your FRED API key:
```
FRED_API_KEY=your_api_key_here
```

2. Get your free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)

## 📊 Project Structure

```
FixedIncomeAnalysis/
├── app.py                    # Streamlit dashboard
├── main.py                   # Main analysis pipeline
├── data_manager.py           # Database operations
├── requirements.txt          # Python dependencies
├── src/
│   ├── forecasting.py        # GARCH and AR models
│   ├── yield_curve_model.py  # NS and NSS models
│   └── analysis.py           # Portfolio analysis
└── data/                     # SQLite database (auto-created)
```

## 🎯 Key Achievements

- **Risk Analysis**: Implemented GARCH(1,1) volatility modeling with forecasting
- **Model Improvement**: NSS model shows measurable RMSE improvement over NS
- **Data Engineering**: Built production-grade database with incremental updates
- **Professional UI**: Interactive dashboard with model comparison features
- **Code Quality**: Comprehensive error handling and fallback mechanisms

## 📈 Model Performance

- **Nelson-Siegel**: ~4.23 bps RMSE (baseline)
- **Nelson-Siegel-Svensson**: Improved fit with additional hump factor
- **GARCH Volatility**: 30-day forecasting with confidence intervals

## 🤝 Contributing

This project demonstrates professional quantitative finance and data engineering skills suitable for financial analyst and quantitative researcher roles.

## 📞 Contact

- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]
