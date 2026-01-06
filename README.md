# Stock Swing Trader

Stock Swing Trader is a comprehensive technical and fundamental analysis tool designed to assist traders in making informed decisions. Built with Python and Streamlit, this application provides multi-timeframe analysis, consensus-based trading signals, and machine learning predictions for stock price movements.

## Features

### 1. Multi-Timeframe Analysis
The application analyzes stock data across three distinct timeframes to identify trend alignment:
- **1 Hour (1h)**: For short-term momentum and entry precision.
- **Daily (1d)**: For the primary trend and swing trading setup.
- **Weekly (1wk)**: For long-term trend confirmation.

### 2. Technical Indicators & Voting System
A consensus-based algorithm aggregates signals from multiple technical indicators to generate a final BUY, SELL, or NEUTRAL recommendation. The indicators include:
- **RSI (Relative Strength Index)**: Identifies overbought and oversold conditions.
- **MACD (Moving Average Convergence Divergence)**: Detects trend potential and reversals.
- **EMA (Exponential Moving Average)**: Analyzes trend direction using EMA 20 and EMA 50 crossovers.
- **Bollinger Bands**: Measures volatility and potential price breakouts.
- **Stochastic Oscillator**: Identifies momentum shifts.

### 3. Fundamental Analysis
Integrates key fundamental metrics to validate technical setups:
- Valuation Ratios (P/E, PEG, Price to Book).
- Profitability Metrics (ROE, Profit Margin).
- Financial Health (Debt-to-Equity, Current Ratio).
- Analyst Ratings and Price Targets.

### 4. Backtesting Engine
Includes a historical simulation feature allowing users to test strategy performance with customizable risk management parameters:
- Adjustable Take Profit and Stop Loss percentages.
- Win Rate, Total Return, and Profit/Loss ratio calculation.
- Detailed trade log generation.

### 5. Machine Learning Prediction
Leverages historical data to predict future price direction using ensemble learning algorithms:
- **Random Forest Classifier**: Robust non-linear classification.
- **XGBoost**: Gradient boosting for high-performance predictions.
- **Feature Importance**: Visualizes which indicators influence the model's decision the most.

## Technology Stack

- **Core**: Python
- **Interface**: Streamlit
- **Data**: yfinance
- **Analysis**: pandas, numpy, pandas-ta
- **Visualization**: Plotly
- **Machine Learning**: scikit-learn, xgboost

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kei2511/app-analisis-saham.git
   cd app-analisis-saham
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a valid stock ticker in the sidebar (e.g., BBCA.JK for Indonesian stocks or AAPL for US stocks).
2. View the Multi-Timeframe Analysis table for an immediate signal summary.
3. Analyze the interactive charts to visualize price action and indicators.
4. Run the Backtest module to verify strategy performance on historical data.
5. Check the Machine Learning section for predictive insights on the next market session.

## License

This project is open source and available under the MIT License.
