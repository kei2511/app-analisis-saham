import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page Configuration
st.set_page_config(layout="wide", page_title="Stock Swing Trader 🚀")

# Sidebar
st.sidebar.header("⚙️ Configuration")
ticker_input = st.sidebar.text_input("Stock Ticker (e.g., BBCA.JK, AAPL)", value="BBCA.JK").upper()

# Timeframe configurations
TIMEFRAMES = {
    "1h": {"interval": "1h", "period": "1mo", "label": "1 Hour"},
    "1d": {"interval": "1d", "period": "1y", "label": "Daily"},
    "1wk": {"interval": "1wk", "period": "2y", "label": "Weekly"}
}

# Helper Function: Fetch Data for specific timeframe
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data(ticker, interval, period):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # If empty and doesn't have .JK, try adding .JK
        if data.empty and not ticker.endswith(".JK"):
            ticker_jk = ticker + ".JK"
            data = yf.download(ticker_jk, period=period, interval=interval, progress=False)
            if not data.empty:
                ticker = ticker_jk

        if data.empty:
            return None, None
            
        # Drop MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        return data, ticker
    except Exception as e:
        return None, None

# Helper Function: Calculate indicators and get signal
def analyze_timeframe(data):
    if data is None or len(data) < 50:
        return None
    
    # Calculate indicators
    data = data.copy()
    data['EMA20'] = ta.ema(data['Close'], length=20)
    data['EMA50'] = ta.ema(data['Close'], length=50)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_SIGNAL'] = macd['MACDs_12_26_9']
    
    bb = ta.bbands(data['Close'], length=20, std=2)
    data['BB_LOWER'] = bb.iloc[:, 0]
    data['BB_UPPER'] = bb.iloc[:, 2]
    
    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
    data['STOCH_K'] = stoch['STOCHk_14_3_3']
    data['STOCH_D'] = stoch['STOCHd_14_3_3']
    
    latest = data.iloc[-1]
    current_price = latest['Close']
    
    # Voting
    signals = {}
    votes = 0
    
    # RSI
    if latest['RSI'] < 30:
        signals['RSI'] = ("🟢 BUY", 1)
        votes += 1
    elif latest['RSI'] > 70:
        signals['RSI'] = ("🔴 SELL", -1)
        votes -= 1
    else:
        signals['RSI'] = ("⚪ NEUTRAL", 0)
    
    # MACD
    if latest['MACD'] > latest['MACD_SIGNAL']:
        signals['MACD'] = ("🟢 BUY", 1)
        votes += 1
    else:
        signals['MACD'] = ("🔴 SELL", -1)
        votes -= 1
    
    # EMA
    if latest['EMA20'] > latest['EMA50']:
        signals['EMA'] = ("🟢 BUY", 1)
        votes += 1
    else:
        signals['EMA'] = ("🔴 SELL", -1)
        votes -= 1
    
    # Bollinger
    if current_price < latest['BB_LOWER']:
        signals['BB'] = ("🟢 BUY", 1)
        votes += 1
    elif current_price > latest['BB_UPPER']:
        signals['BB'] = ("🔴 SELL", -1)
        votes -= 1
    else:
        signals['BB'] = ("⚪ NEUTRAL", 0)
    
    # Stochastic
    if latest['STOCH_K'] < 20 and latest['STOCH_K'] > latest['STOCH_D']:
        signals['Stoch'] = ("🟢 BUY", 1)
        votes += 1
    elif latest['STOCH_K'] > 80 and latest['STOCH_K'] < latest['STOCH_D']:
        signals['Stoch'] = ("🔴 SELL", -1)
        votes -= 1
    else:
        signals['Stoch'] = ("⚪ NEUTRAL", 0)
    
    # Overall signal
    if votes >= 2:
        overall = "🟢 BUY"
    elif votes <= -2:
        overall = "🔴 SELL"
    else:
        overall = "⚪ NEUTRAL"
    
    return {
        'data': data,
        'signals': signals,
        'votes': votes,
        'overall': overall,
        'price': current_price,
        'latest': latest
    }

# Helper Function: Get Fundamentals & Analyst Ratings
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            # Basic
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            # Valuation
            "pe": info.get("trailingPE", "N/A"),
            "peg": info.get("pegRatio", "N/A"),
            "pb": info.get("priceToBook", "N/A"),
            # Profitability
            "roe": info.get("returnOnEquity", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            # Financial Health
            "debt_equity": info.get("debtToEquity", "N/A"),
            "current_ratio": info.get("currentRatio", "N/A"),
            # Dividends
            "dividend_yield": info.get("dividendYield", "N/A"),
            # Analyst Ratings
            "recommendation": info.get("recommendationKey", "N/A"),
            "num_analysts": info.get("numberOfAnalystOpinions", "N/A"),
            "target_mean": info.get("targetMeanPrice", "N/A"),
            "target_high": info.get("targetHighPrice", "N/A"),
            "target_low": info.get("targetLowPrice", "N/A"),
        }
    except:
        return None

# =============== MAIN APP ===============
st.title("📈 Multi-Timeframe Stock Analyzer")

# Fetch data for all timeframes
results = {}
resolved_ticker = None

with st.spinner("Fetching data for multiple timeframes..."):
    for tf_key, tf_config in TIMEFRAMES.items():
        data, ticker = get_data(ticker_input, tf_config['interval'], tf_config['period'])
        if data is not None:
            resolved_ticker = ticker
            analysis = analyze_timeframe(data)
            if analysis:
                results[tf_key] = {
                    'label': tf_config['label'],
                    'analysis': analysis
                }

if resolved_ticker and results:
    st.subheader(f"Ticker: **{resolved_ticker}**")
    
    # ===== MULTI-TIMEFRAME COMPARISON TABLE =====
    st.markdown("## 🕐 Multi-Timeframe Analysis")
    
    # Build comparison table
    comparison_data = []
    tf_votes = []
    
    for tf_key in ['1h', '1d', '1wk']:
        if tf_key in results:
            r = results[tf_key]
            analysis = r['analysis']
            row = {
                'Timeframe': r['label'],
                'RSI': analysis['signals']['RSI'][0],
                'MACD': analysis['signals']['MACD'][0],
                'EMA': analysis['signals']['EMA'][0],
                'Bollinger': analysis['signals']['BB'][0],
                'Stochastic': analysis['signals']['Stoch'][0],
                'Net Score': analysis['votes'],
                'Overall': analysis['overall']
            }
            comparison_data.append(row)
            tf_votes.append(analysis['votes'])
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # ===== OVERALL CONSENSUS =====
    total_tf_votes = sum(tf_votes)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Final Consensus")
        if total_tf_votes >= 3:
            final = "STRONG BUY 🚀"
            color = "green"
        elif total_tf_votes >= 1:
            final = "BUY 🟢"
            color = "green"
        elif total_tf_votes <= -3:
            final = "STRONG SELL 📉"
            color = "red"
        elif total_tf_votes <= -1:
            final = "SELL 🔴"
            color = "red"
        else:
            final = "NEUTRAL 😐"
            color = "gray"
        st.markdown(f"<h1 style='color: {color};'>{final}</h1>", unsafe_allow_html=True)
        st.write(f"Combined Score: **{total_tf_votes}** (from {len(tf_votes)} timeframes)")
    
    with col2:
        st.markdown("### 💰 Current Price")
        if '1d' in results:
            daily = results['1d']['analysis']
            current_price = daily['price']
            st.metric("Price", f"{current_price:,.2f}")
            
            # Risk Management
            atr = ta.atr(daily['data']['High'], daily['data']['Low'], daily['data']['Close'], length=14).iloc[-1]
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
            st.write(f"**Stop Loss:** {stop_loss:,.2f}")
            st.write(f"**Take Profit:** {take_profit:,.2f}")
    
    with col3:
        st.markdown("### 📊 Fundamentals")
        fund = get_fundamentals(resolved_ticker)
        if fund:
            st.write(f"**{fund['name']}**")
            st.write(f"Sector: {fund['sector']}")
            
            # Format ROE as percentage
            roe_display = f"{fund['roe']*100:.1f}%" if isinstance(fund['roe'], (int, float)) else "N/A"
            div_display = f"{fund['dividend_yield']*100:.2f}%" if isinstance(fund['dividend_yield'], (int, float)) else "N/A"
            
            st.write(f"P/E: {fund['pe']} | PEG: {fund['peg']}")
            st.write(f"ROE: {roe_display} | Div Yield: {div_display}")
    
    # ===== ANALYST RATINGS SECTION =====
    st.markdown("---")
    st.markdown("## 🏦 Analyst Ratings")
    
    if fund and fund['recommendation'] != "N/A":
        col_a, col_b, col_c, col_d = st.columns(4)
        
        # Recommendation with color
        rec = fund['recommendation'].upper()
        if rec in ['STRONG_BUY', 'BUY']:
            rec_color = "green"
            rec_emoji = "🟢"
        elif rec in ['SELL', 'STRONG_SELL']:
            rec_color = "red"
            rec_emoji = "🔴"
        else:
            rec_color = "orange"
            rec_emoji = "🟡"
        
        with col_a:
            st.markdown(f"**Consensus:**")
            st.markdown(f"<span style='color:{rec_color}; font-size:1.5em;'>{rec_emoji} {rec.replace('_', ' ')}</span>", unsafe_allow_html=True)
            st.write(f"Based on {fund['num_analysts']} analysts")
        
        with col_b:
            st.metric("Target (Mean)", f"{fund['target_mean']:,.0f}" if isinstance(fund['target_mean'], (int, float)) else "N/A")
        
        with col_c:
            st.metric("Target (High)", f"{fund['target_high']:,.0f}" if isinstance(fund['target_high'], (int, float)) else "N/A")
        
        with col_d:
            st.metric("Target (Low)", f"{fund['target_low']:,.0f}" if isinstance(fund['target_low'], (int, float)) else "N/A")
    else:
        st.info("Analyst ratings not available for this ticker (common for Indonesian stocks).")
    
    # ===== DETAILED STRATEGY BREAKDOWN =====
    with st.expander("📊 Detailed Strategy Breakdown per Timeframe"):
        for tf_key in ['1h', '1d', '1wk']:
            if tf_key in results:
                r = results[tf_key]
                analysis = r['analysis']
                st.markdown(f"### {r['label']}")
                
                detail_data = []
                for strat, (signal, vote) in analysis['signals'].items():
                    detail_data.append({
                        'Strategy': strat,
                        'Signal': signal,
                        'Vote': vote
                    })
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
                st.markdown("---")
    
    # ===== CHART (Daily by default) =====
    st.markdown("## 📈 Price Chart (Daily)")
    if '1d' in results:
        daily_data = results['1d']['analysis']['data']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=daily_data.index,
                        open=daily_data['Open'], high=daily_data['High'],
                        low=daily_data['Low'], close=daily_data['Close'], name="Price"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['EMA20'], line=dict(color='yellow', width=1), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['EMA50'], line=dict(color='orange', width=1), name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['BB_UPPER'], line=dict(color='blue', width=1, dash='dot'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['BB_LOWER'], line=dict(color='blue', width=1, dash='dot'), name="BB Lower"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🔬 Backtest (Historical Signal Simulation)")
    
    if '1d' in results:
        with st.expander("📊 Run Backtest on Daily Data", expanded=False):
            st.info("BACKTEST WITH TP/SL: Matches real swing trading.\n"
                    "We buy on signal, but SELL automatically if we hit Target Profit or Stop Loss.")
            
            # User Inputs for Backtest
            col_b1, col_b2 = st.columns(2)
            tp_pct = col_b1.slider("Take Profit (%)", 1.0, 20.0, 5.0, 0.5) / 100
            sl_pct = col_b2.slider("Stop Loss (%)", 1.0, 10.0, 3.0, 0.5) / 100
            
            daily_data = results['1d']['analysis']['data'].copy()
            
            # 1. Calculate Signals
            daily_data['Signal'] = 0  
            for i in range(50, len(daily_data)):
                row = daily_data.iloc[i]
                votes = 0
                
                if row['RSI'] < 30: votes += 1
                elif row['RSI'] > 70: votes -= 1
                
                if row['MACD'] > row['MACD_SIGNAL']: votes += 1
                else: votes -= 1
                
                if row['EMA20'] > row['EMA50']: votes += 1
                else: votes -= 1
                
                if row['Close'] < row['BB_LOWER']: votes += 1
                elif row['Close'] > row['BB_UPPER']: votes -= 1
                
                if row['STOCH_K'] < 20 and row['STOCH_K'] > row['STOCH_D']: votes += 1
                elif row['STOCH_K'] > 80 and row['STOCH_K'] < row['STOCH_D']: votes -= 1
                
                if votes >= 2: daily_data.iloc[i, daily_data.columns.get_loc('Signal')] = 1
                elif votes <= -2: daily_data.iloc[i, daily_data.columns.get_loc('Signal')] = -1
            
            # 2. Simulate Trading with TP/SL
            trades = []
            position = None 
            
            for i in range(50, len(daily_data)):
                row = daily_data.iloc[i]
                date = daily_data.index[i]
                # High/Low are important for TP/SL checks
                price_open = row['Open']
                price_high = row['High']
                price_low = row['Low']
                price_close = row['Close']
                signal = row['Signal']
                
                # Check OPEN Position (TP/SL)
                if position is not None:
                    entry = position['entry_price']
                    target_price = entry * (1 + tp_pct)
                    stop_price = entry * (1 - sl_pct)
                    
                    # 1. Check STOP LOSS (Hit Low?)
                    if price_low <= stop_price:
                        # Executed at SL price
                        exit_price = stop_price
                        pnl_pct = -sl_pct * 100
                        trades.append({
                            'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                            'Entry Price': entry,
                            'Exit Date': date.strftime('%Y-%m-%d'),
                            'Exit Price': exit_price,
                            'P/L (%)': pnl_pct,
                            'Result': '❌ SL Hit'
                        })
                        position = None
                        continue # Trade closed, move to next day
                    
                    # 2. Check TAKE PROFIT (Hit High?)
                    elif price_high >= target_price:
                        # Executed at TP price
                        exit_price = target_price
                        pnl_pct = tp_pct * 100
                        trades.append({
                            'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                            'Entry Price': entry,
                            'Exit Date': date.strftime('%Y-%m-%d'),
                            'Exit Price': exit_price,
                            'P/L (%)': pnl_pct,
                            'Result': '✅ TP Hit'
                        })
                        position = None
                        continue
                    
                    # 3. Indicator-based Exit (Signal Sell)
                    elif signal == -1:
                        pnl_pct = ((price_close - entry) / entry) * 100
                        result = '✅ Signal Win' if pnl_pct > 0 else '❌ Signal Loss'
                        trades.append({
                            'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                            'Entry Price': entry,
                            'Exit Date': date.strftime('%Y-%m-%d'),
                            'Exit Price': price_close,
                            'P/L (%)': pnl_pct,
                            'Result': result
                        })
                        position = None
                        continue

                # Check NEW Entry
                if position is None and signal == 1:
                    # Buy at Close (simplify)
                    position = {
                        'entry_date': date, 
                        'entry_price': price_close
                    }
            
            # Close pending position
            if position is not None:
                last_price = daily_data.iloc[-1]['Close']
                pnl = ((last_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'Entry Date': position['entry_date'].strftime('%Y-%m-%d'),
                    'Entry Price': position['entry_price'],
                    'Exit Date': "Open",
                    'Exit Price': last_price,
                    'P/L (%)': pnl,
                    'Result': '⏳ OPEN'
                })
            
            # 3. Stats Display
            if trades:
                df_trades = pd.DataFrame(trades)
                
                closed_trades = [t for t in trades if t['Result'] != '⏳ OPEN']
                wins = len([t for t in closed_trades if float(t['P/L (%)']) > 0])
                losses = len([t for t in closed_trades if float(t['P/L (%)']) <= 0])
                win_rate = (wins / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
                total_return = sum([float(t['P/L (%)']) for t in closed_trades])
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Total Trades", len(closed_trades))
                col_s2.metric("Win Rate", f"{win_rate:.1f}%")
                col_s3.metric("Total Return", f"{total_return:+.2f}%")
                col_s4.metric("Wins/Losses", f"{wins} / {losses}")
                
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.warning("No trades triggered.")
    
    # ===== ML PREDICTION SECTION =====
    st.markdown("---")
    st.markdown("## 🤖 Machine Learning Prediction")
    
    if '1d' in results:
        with st.expander("🧠 ML Prediction (Random Forest & XGBoost)", expanded=False):
            st.info("ML models are trained on historical data to predict if tomorrow's price will go UP or DOWN.")
            
            daily_data = results['1d']['analysis']['data'].copy()
            
            # Feature Engineering
            daily_data['Return'] = daily_data['Close'].pct_change()
            daily_data['Target'] = (daily_data['Close'].shift(-1) > daily_data['Close']).astype(int)  # 1=UP, 0=DOWN
            
            # Features: Technical Indicators
            features = ['RSI', 'MACD', 'MACD_SIGNAL', 'EMA20', 'EMA50', 'BB_UPPER', 'BB_LOWER', 'STOCH_K', 'STOCH_D']
            
            # Add more features
            daily_data['Price_vs_EMA20'] = daily_data['Close'] / daily_data['EMA20']
            daily_data['Price_vs_EMA50'] = daily_data['Close'] / daily_data['EMA50']
            daily_data['EMA_Cross'] = (daily_data['EMA20'] > daily_data['EMA50']).astype(int)
            daily_data['BB_Position'] = (daily_data['Close'] - daily_data['BB_LOWER']) / (daily_data['BB_UPPER'] - daily_data['BB_LOWER'])
            daily_data['Volume_SMA'] = daily_data['Volume'].rolling(20).mean()
            daily_data['Volume_Ratio'] = daily_data['Volume'] / daily_data['Volume_SMA']
            
            features += ['Price_vs_EMA20', 'Price_vs_EMA50', 'EMA_Cross', 'BB_Position', 'Volume_Ratio', 'Return']
            
            # Clean data
            ml_data = daily_data.dropna()
            ml_data = ml_data[:-1]  # Remove last row (no target)
            
            if len(ml_data) < 100:
                st.warning("Not enough data for ML training (need at least 100 days).")
            else:
                X = ml_data[features]
                y = ml_data['Target']
                
                # Train/Test Split (80/20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                # ===== RANDOM FOREST =====
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_acc = accuracy_score(y_test, rf_pred) * 100
                
                # ===== XGBOOST =====
                xgb_acc = None
                xgb_pred = None
                if XGB_AVAILABLE:
                    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    xgb_acc = accuracy_score(y_test, xgb_pred) * 100
                
                # Display Results
                st.markdown("### Model Accuracy Comparison")
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.metric("🌲 Random Forest", f"{rf_acc:.1f}%")
                
                with col_m2:
                    if XGB_AVAILABLE:
                        st.metric("🚀 XGBoost", f"{xgb_acc:.1f}%")
                    else:
                        st.warning("XGBoost not installed")
                
                # Tomorrow's Prediction
                st.markdown("### 🔮 Tomorrow's Prediction")
                
                # Get latest features
                latest_features = daily_data[features].iloc[-1:]
                
                if not latest_features.isnull().values.any():
                    rf_tomorrow = rf_model.predict(latest_features)[0]
                    rf_prob = rf_model.predict_proba(latest_features)[0]
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        rf_direction = "📈 UP" if rf_tomorrow == 1 else "📉 DOWN"
                        rf_confidence = max(rf_prob) * 100
                        st.markdown(f"**Random Forest:** {rf_direction}")
                        st.write(f"Confidence: {rf_confidence:.1f}%")
                    
                    with col_p2:
                        if XGB_AVAILABLE:
                            xgb_tomorrow = xgb_model.predict(latest_features)[0]
                            xgb_prob = xgb_model.predict_proba(latest_features)[0]
                            xgb_direction = "📈 UP" if xgb_tomorrow == 1 else "📉 DOWN"
                            xgb_confidence = max(xgb_prob) * 100
                            st.markdown(f"**XGBoost:** {xgb_direction}")
                            st.write(f"Confidence: {xgb_confidence:.1f}%")
                    
                    # Consensus
                    st.markdown("---")
                    if XGB_AVAILABLE:
                        if rf_tomorrow == xgb_tomorrow:
                            consensus_dir = "📈 UP" if rf_tomorrow == 1 else "📉 DOWN"
                            st.success(f"✅ **Both models agree:** {consensus_dir}")
                        else:
                            st.warning("⚠️ **Models disagree.** Be cautious.")
                    else:
                        st.info(f"Single model prediction: {rf_direction}")
                else:
                    st.warning("Could not generate prediction (missing features).")
                
                # Feature Importance
                with st.expander("📊 Feature Importance (Random Forest)"):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)

else:
    st.warning("Ticker not found or insufficient data. Try 'BBCA.JK', 'BUMI', or 'AAPL'.")
