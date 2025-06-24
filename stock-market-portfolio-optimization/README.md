# Stock Market Portfolio Optimization

![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-purple.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-red.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-yellow.svg)

A comprehensive Python implementation of Modern Portfolio Theory for optimal portfolio construction using Indian stock market data. This project analyzes historical stock performance, calculates risk-return metrics, and identifies the optimal portfolio allocation using the Sharpe ratio maximization approach.

## 📊 Project Overview

This portfolio optimization tool analyzes four major Indian stocks:
- **RELIANCE.NS** - Reliance Industries Limited
- **TCS.NS** - Tata Consultancy Services
- **INFY.NS** - Infosys Limited
- **HDFCBANK.NS** - HDFC Bank Limited

The analysis covers 365 days of historical data and implements Monte Carlo simulation to generate 10,000 random portfolio combinations to identify the efficient frontier.

## 🎯 Key Features

- **Data Acquisition**: Automated download of stock data using yfinance
- **Technical Analysis**: 50-day and 200-day moving averages calculation
- **Risk Analysis**: Daily returns distribution and correlation analysis
- **Portfolio Optimization**: Sharpe ratio maximization using Modern Portfolio Theory
- **Visualization**: Comprehensive plots including efficient frontier, price trends, and correlation heatmaps

## 📈 Analysis Results

### Stock Performance Metrics (Annualized)

| Stock | Expected Return | Volatility |
|-------|----------------|------------|
| HDFCBANK.NS | 17.40% | 19.37% |
| INFY.NS | 9.54% | 25.55% |
| RELIANCE.NS | 3.76% | 21.69% |
| TCS.NS | -6.58% | 21.90% |

### Optimal Portfolio Allocation

**Maximum Sharpe Ratio Portfolio:**
- **Expected Return**: 15.31%
- **Volatility**: 16.92%
- **Sharpe Ratio**: 0.905

**Optimal Weights:**
| Stock | Allocation |
|-------|------------|
| INFY.NS | 37.39% |
| TCS.NS | 25.05% |
| HDFCBANK.NS | 19.37% |
| RELIANCE.NS | 18.19% |

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization
```

2. **Install required dependencies:**
```bash
pip install pandas numpy yfinance matplotlib seaborn
```

3. **Create necessary directories:**
```bash
mkdir -p plots/adj_close plots/volume_traded
```

## 🚀 Usage

Run the main analysis script:

```bash
python portfolio.py
```

The script will:
1. Download historical stock data for the past 365 days
2. Generate technical analysis plots
3. Calculate portfolio optimization metrics
4. Save visualization plots to the `plots/` directory
5. Display optimal portfolio allocation results

## 📊 Generated Visualizations

The script generates several plots saved in the `plots/` directory:

- `adj_close_over_time.png` - Stock price trends over time
- `adj_close/{ticker}.png` - Individual stock analysis with moving averages
- `volume_traded/{ticker}.png` - Trading volume analysis
- `distribution_daily_returns.png` - Daily returns distribution
- `correlation_matrix.png` - Stock correlation heatmap
- `efficient_frontier.png` - Portfolio optimization frontier

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- yfinance >= 0.2.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🔍 Methodology

### Modern Portfolio Theory Implementation

1. **Data Collection**: Historical stock prices from Yahoo Finance
2. **Return Calculation**: Daily returns computed using percentage change
3. **Risk Metrics**: Annualized volatility and expected returns
4. **Monte Carlo Simulation**: 10,000 random portfolio weight combinations
5. **Optimization**: Sharpe ratio maximization for optimal risk-adjusted returns

### Key Formulas

- **Expected Return**: μ = (Σ daily returns / n) × 252
- **Volatility**: σ = √(Σ(return - μ)² / n) × √252
- **Sharpe Ratio**: SR = (Portfolio Return - Risk-free Rate) / Portfolio Volatility

## 📋 Project Structure

```
portfolio-optimization/
│
├── portfolio.py                 # Main analysis script
├── README.md                   # Project documentation
├── plots/                      # Generated visualizations
│   ├── adj_close/             # Individual stock analysis
│   ├── volume_traded/         # Volume analysis
│   └── *.png                  # Summary plots
└── requirements.txt           # Dependencies
```

## ⚠️ Important Notes

- Stock data is downloaded fresh each time the script runs
- The analysis uses 365 days of historical data ending on the current date
- Risk-free rate is assumed to be 0 in Sharpe ratio calculations
- Results are based on historical performance and do not guarantee future returns

## 📊 Performance Insights

The optimal portfolio demonstrates:
- **Diversification**: Balanced allocation across all four stocks
- **Risk Management**: 16.92% volatility versus individual stock volatilities of 19-26%
- **Return Optimization**: 15.31% expected return with strong risk-adjusted performance
- **Sharpe Efficiency**: 0.905 Sharpe ratio indicating excellent risk-adjusted returns

## 🔮 Future Enhancements

- Integration of risk-free rate data
- Implementation of additional optimization objectives (minimum variance, maximum return)
- Real-time data updates and portfolio rebalancing alerts
- Extended backtesting with out-of-sample validation
- Integration with more sophisticated risk models (VaR, CVaR)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Disclaimer: This tool is for educational and research purposes only. Investment decisions should be made with proper financial advice and risk assessment.*
