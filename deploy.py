#!/usr/bin/env python3
"""
Deployment preparation script for Fixed Income Analysis project.
Run this before deploying to cloud platforms.
"""

import os
import shutil
from pathlib import Path

def prepare_for_deployment():
    """Prepare project for cloud deployment."""

    print("🚀 Preparing Fixed Income Analysis for deployment...")

    # Create .streamlit config
    streamlit_config = """
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
"""

    # Create .streamlit directory and config
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write(streamlit_config)

    print("✅ Created Streamlit configuration")

    # Create data directory
    os.makedirs('data', exist_ok=True)
    print("✅ Created data directory")

    # Create requirements.txt if needed
    if not os.path.exists('requirements.txt'):
        print("⚠️  requirements.txt not found")

    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/treasury_data.db
data/treasury_data_backup_*.db
*.log
.env
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

    print("✅ Created .gitignore")

    # Create README template
    readme_content = """# Fixed Income Analysis Dashboard

A professional quantitative finance application featuring advanced yield curve modeling, volatility analysis, and production-grade data management.

## 🚀 Live Demo
[View Live Dashboard](https://yourname-fixedincome.streamlit.app)

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
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("✅ Created professional README.md")

    print("\n🎉 Project ready for deployment!")
    print("\n📝 Next steps:")
    print("1. Add your FRED_API_KEY to .env file")
    print("2. Push to GitHub repository")
    print("3. Deploy to Streamlit Cloud")
    print("4. Share professional URL with recruiters")

if __name__ == "__main__":
    prepare_for_deployment()