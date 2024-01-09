import json
import streamlit as st
import requests
import pandas as pd
import locale
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import textwrap
import numpy as np


# Mengatur lokal ke Indonesia
locale.setlocale(locale.LC_ALL, 'id_ID')

# Fungsi untuk menampilkan animasi error
def show_error_animation():
    error_message = """
        <div style="text-align: center; color: red;">
            <style>
                @keyframes shake {
                    0% { transform: translate(0, 0); }
                    10%, 90% { transform: translate(-10px, 0); }
                    20%, 80% { transform: translate(10px, 0); }
                    30%, 50%, 70% { transform: translate(-10px, 0); }
                    40%, 60% { transform: translate(10px, 0); }
                    100% { transform: translate(0, 0); }
                }
                p {
                    animation: shake 0.5s;
                }
            </style>
        </div>
    """
    st.markdown(error_message, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .sidebar {
                background-color: #f8f9fa;
                padding: 20px;
                height: 100vh;
                width: 200px;
            }
            .content {
                padding: 20px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.title("Menu")

#==============Start==================#

# Menampilkan pilihan menu
menu_options = ["Dashboard", "Prediksi"]
selected_menu = st.sidebar.selectbox("Pilih Menu", menu_options)

# Menampilkan konten sesuai dengan menu yang dipilih
if selected_menu == "Dashboard":
    # Konten untuk Dashboard
    st.title("Dashboard")
    
    # Mengambil data transaksi dari CoinGecko API
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "idr",
        "order": "market_cap_desc",
        "per_page": 21,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }

    response = requests.get(url, params=params)
    coins_data = response.json()


    # Membuat tabel untuk menampilkan data transaksi
    table_data = []
    for coin in coins_data:
        current_price = locale.currency(coin["current_price"], grouping=True)
        total_volume = locale.currency(coin["total_volume"], grouping=True)
        market_cap = locale.currency(coin["market_cap"], grouping=True)
        row = {
            "Nama": coin["name"],
            "Simbol": coin["symbol"],
            "Harga Hari ini": current_price,
            "Total Transaksi 24 Jam": total_volume,
            "Total Pasar": market_cap
        }
        table_data.append(row)

    # Menampilkan tabel data transaksi
    st.table(table_data)

    #============End================#
    # Mengambil data historis dari CoinGecko API
    url = "https://api.coingecko.com/api/v3/coins/binance/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": 1514764800,  # Timestamp untuk 1 Januari 2018
        "to": 1672396799  # Timestamp untuk 31 Desember 2023
    }

    response = requests.get(url, params=params)
    chart_data = response.json()

    # Membuat DataFrame dari data historis
    df = pd.DataFrame(chart_data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Menampilkan judul halaman
    st.title("Area Chart - Historis Koin Binance")

    # Menampilkan area chart menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(df.index, df["price"], color="skyblue")

    # Mengatur label sumbu x dan y
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (USD)")

    # Memutar label sumbu x
    plt.xticks(rotation=45)

    # Menampilkan grafik
    st.pyplot(fig)

else:
    # Load the dataset
    data_path = "C:/project-akhir/data_binance.csv"
    df = pd.read_csv(data_path)

    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Pra-pemrosesan data: Mengubah format 'Vol.' menjadi float
    df['Vol.'] = df['Vol.'].replace('-', float('nan'))  # Ganti '-' dengan NaN
    df['Vol.'] = df['Vol.'].fillna(0)  # Isi nilai NaN dengan 0
    df['Vol.'] = df['Vol.'].astype(float)  # Konversi ke float

    # Pra-pemrosesan data: Mengubah format 'Vol.' dan 'Change %' menjadi float
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

    # Select features and target variable
    features = ['Open', 'High', 'Low', 'Vol.', 'Change %']
    target = 'Price'

    # Split the data into train and test sets
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

    # Calculate MAPE
    def calculate_mape(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

    train_mape = calculate_mape(y_train, y_pred_train)
    test_mape = calculate_mape(y_test, y_pred_test)
    
    # Function to convert Binance price to Rupiah
    def convert_to_rupiah(price):
        response = requests.get(f"https://open.er-api.com/v6/latest/USD")
        data = response.json()
        exchange_rate = data["rates"]["IDR"]
        rupiah_price = price * exchange_rate
        return rupiah_price

    # Streamlit app
    st.title('Prediksi Harga Binance')

    # Input features for prediction
    st.warning('Harap isi semua input untuk melakukan prediksi')
col1, col2 = st.columns(2)
with col1:
    open_price = st.number_input('Harga Pembukaan')
    high_price = st.number_input('Harga Tertinggi')
    low_price = st.number_input('Harga Terendah')
with col2:
    volume = st.number_input('Volume')
    change_percentage = st.number_input('Perubahan Persentase')
    prediction_date = st.date_input('Tanggal Prediksi')

    # Predict function
    def predict_price(open_price, high_price, low_price, volume, change_percentage, prediction_date):
        features = [[open_price, high_price, low_price, volume, change_percentage]]
        prediction = model.predict(features)
        prediction_df = pd.DataFrame({'Date': [prediction_date], 'Price': [prediction[0]]})
        return prediction_df

    # Button to trigger prediction
col1, col2 = st.columns(2)
with col1:
    if st.button('Prediksi'):
        if open_price and high_price and low_price and volume and change_percentage and prediction_date:
            prediction_df = predict_price(open_price, high_price, low_price, volume, change_percentage, prediction_date)
            st.success(f'Prediksi Harga Binance : $ {prediction_df["Price"].values[0]:.2f} USD')
            # Convert price to Rupiah           
            rupiah_price = convert_to_rupiah(prediction_df["Price"].values[0])
            st.success(f'Hasil Prediksi BNB Ke IDR : Rp. {locale.format_string("%.2f", rupiah_price, grouping=True)}')                
            st.info(f'Prediksi Harga Binance (BNB) pada tanggal : {prediction_date}')
                # Calculate MAPE
        else:
            st.warning("Mohon isi semua inputan untuk melakukan prediksi!")
            show_error_animation()           
with col2:
    if st.button('Reset'):
        open_price = None
        high_price = None
        low_price = None
        volume = None
        change_percentage = None 
        
    # Display model evaluation metrics
    # col5, col6 = st.columns(2)
    # with col5:
    #     st.markdown(f'<span title="Train RMSE (Root Mean Squared Error) adalah metrik evaluasi yang digunakan untuk mengukur sejauh mana model regresi cocok dengan data pelatihan. Nilai Train RMSE menggambarkan sejauh mana perbedaan antara nilai aktual dan nilai prediksi oleh model. Semakin rendah nilai Train RMSE, semakin baik model dapat mempelajari pola dalam data pelatihan \n Rumus untuk menghitung Train RMSE adalah sebagai berikut: \n RMSE = sqrt(mean((y_actual - y_pred)^2)) \n Di mana: \n -y_actual adalah nilai aktual dari target variabel dalam data pelatihan. \n -y_pred adalah nilai yang diprediksi oleh model untuk target variabel dalam data pelatihan. \n -mean adalah rata-rata dari seluruh perbedaan kuadrat antara nilai aktual dan nilai prediksi.\n -sqrt adalah operasi akar kuadrat." style="background-color: #54d2d2; color:#072448; padding: 10px; border-radius: 5px; width: 25%;">Train RMSE : {train_rmse:.2f} </span>', unsafe_allow_html=True)
    #     "\n"
    #     st.markdown(f'<span title="RMSE (Root Mean Squared Error) adalah metrik evaluasi yang umum digunakan untuk mengukur seberapa baik model regresi memprediksi nilai target. Nilai RMSE mengindikasikan sejauh mana perbedaan antara nilai yang diprediksi oleh model dan nilai aktual. \n Rumus untuk menghitung RMSE adalah sebagai berikut: \n RMSE = sqrt(mean((y_true - y_pred)^2)) \n Di mana: \n -y_true adalah nilai target aktual (ground truth) \n -y_pred adalah nilai target yang diprediksi oleh model \n -mean() mengacu pada rata-rata (mean) dari seluruh elemen dalam dataset \n -sqrt() adalah operasi akar kuadrat (square root)" style="background-color: #54d2d2; color:#072448; padding: 10px; border-radius: 5px;">Test RMSE: {test_rmse:.2f}</span>', unsafe_allow_html=True)
    # with col6:
    #     st.markdown(f'<span title="Train MAPE (Mean Absolute Percentage Error) adalah metrik evaluasi yang digunakan untuk mengukur akurasi relatif dari model regresi. MAPE mengukur persentase rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya, dibagi dengan nilai sebenarnya, dan kemudian diambil rata-ratanya." style="background-color: #54d2d2; color:#072448; padding: 10px; border-radius: 5px; width: 25%;">Train MAPE: {train_mape:.2f} % </span>', unsafe_allow_html=True)
    #     "\n"
    #     st.markdown(f'<span title="Test MAPE adalah singkatan dari Mean Absolute Percentage Error pada tahap pengujian (testing) model prediksi. Ini adalah salah satu metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan relatif dari prediksi model dalam bentuk persentase. \n Test MAPE menghitung rata-rata dari selisih absolut antara nilai sebenarnya (y_true) dan nilai prediksi (y_pred), kemudian membaginya dengan nilai sebenarnya (y_true), dan dikalikan dengan 100 untuk mendapatkan hasil dalam bentuk persentase. \n Rumus Test MAPE: \n Test MAPE = (1 / n) * Σ(|(y_true - y_pred) / y_true|) * 100 \n -Test MAPE: Nilai Test MAPE (Mean Absolute Percentage Error) \n -y_true: Nilai sebenarnya (true value) \n -y_pred: Nilai prediksi (predicted value) \n -n: Jumlah sampel data" style="background-color: #54d2d2; color:#072448; padding: 10px; border-radius: 5px; width: 25%;">Test MAPE: {test_mape:.2f} % </span>', unsafe_allow_html=True)
    # "\n"