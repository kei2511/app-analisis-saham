# Stock Swing Trader

Stock Swing Trader adalah alat analisis teknikal dan fundamental komprehensif yang dirancang untuk membantu trader dalam mengambil keputusan yang tepat. Dibangun menggunakan Python dan Streamlit, aplikasi ini menyediakan analisis multi-timeframe, sinyal perdagangan berbasis konsensus, dan prediksi machine learning untuk pergerakan harga saham.

## Fitur

### 1. Analisis Multi-Timeframe
Aplikasi ini menganalisis data saham di tiga kerangka waktu (timeframe) berbeda untuk mengidentifikasi keselarasan tren:
- **1 Jam (1h)**: Untuk momentum jangka pendek dan presisi entri.
- **Harian (1d)**: Untuk tren utama dan pengaturan swing trading.
- **Mingguan (1wk)**: Untuk konfirmasi tren jangka panjang.

### 2. Indikator Teknikal & Sistem Voting
Algoritma berbasis konsensus mengagregasi sinyal dari berbagai indikator teknikal untuk menghasilkan rekomendasi akhir BELI (BUY), JUAL (SELL), atau NETRAL. Indikator yang digunakan meliputi:
- **RSI (Relative Strength Index)**: Mengidentifikasi kondisi jenuh beli dan jenuh jual.
- **MACD (Moving Average Convergence Divergence)**: Mendeteksi potensi tren dan pembalikan arah.
- **EMA (Exponential Moving Average)**: Menganalisis arah tren menggunakan persilangan EMA 20 dan EMA 50.
- **Bollinger Bands**: Mengukur volatilitas dan potensi penembusan harga (breakout).
- **Stochastic Oscillator**: Mengidentifikasi pergeseran momentum.

### 3. Analisis Fundamental
Mengintegrasikan metrik fundamental utama untuk memvalidasi pengaturan teknikal:
- Rasio Valuasi (P/E, PEG, Price to Book).
- Metrik Profitabilitas (ROE, Margin Keuntungan).
- Kesehatan Finansial (Debt-to-Equity, Current Ratio).
- Rating Analis dan Target Harga.

### 4. Mesin Backtesting
Mencakup fitur simulasi historis yang memungkinkan pengguna untuk menguji kinerja strategi dengan parameter manajemen risiko yang dapat disesuaikan:
- Persentase Take Profit dan Stop Loss yang dapat diatur.
- Perhitungan Win Rate, Total Return, dan rasio Laba/Rugi.
- Pembuatan log perdagangan terperinci.

### 5. Prediksi Machine Learning
Memanfaatkan data historis untuk memprediksi arah harga di masa depan menggunakan algoritma ensemble learning:
- **Random Forest Classifier**: Klasifikasi non-linear yang kuat.
- **XGBoost**: Gradient boosting untuk prediksi berkinerja tinggi.
- **Feature Importance**: Memvisualisasikan indikator mana yang paling mempengaruhi keputusan model.

## Teknologi yang Digunakan

- **Inti**: Python
- **Antarmuka**: Streamlit
- **Data**: yfinance
- **Analisis**: pandas, numpy, pandas-ta
- **Visualisasi**: Plotly
- **Machine Learning**: scikit-learn, xgboost

## Instalasi

1. Clone repositori:
   ```bash
   git clone https://github.com/kei2511/app-analisis-saham.git
   cd app-analisis-saham
   ```

2. Install dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   python -m streamlit run app.py
   ```
   *Catatan: Jika perintah `streamlit` tidak dikenali, gunakan `python -m streamlit run app.py`.*

## Penggunaan

1. Masukkan kode saham yang valid di sidebar (contoh: BBCA.JK untuk saham Indonesia atau AAPL untuk saham AS).
2. Lihat tabel Analisis Multi-Timeframe untuk ringkasan sinyal langsung.
3. Analisis grafik interaktif untuk memvisualisasikan aksi harga dan indikator.
4. Jalankan modul Backtest untuk memverifikasi kinerja strategi pada data historis.
5. Periksa bagian Machine Learning untuk wawasan prediktif pada sesi pasar berikutnya.

## Lisensi

Proyek ini bersifat open source dan tersedia di bawah Lisensi MIT.
