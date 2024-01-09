import json
import streamlit as st
import requests
import pandas as pd
import locale
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Mengatur lokal ke Indonesia
locale.setlocale(locale.LC_ALL, 'id_ID')

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
   # Load data
    data_path = 'C:/project-akhir/binance.csv'
    data = pd.read_csv(data_path)

    # Preprocessing data
    data['Date'] = pd.to_datetime(data['Date'])  # Konversi kolom 'Date' ke tipe datetime
    data.set_index('Date', inplace=True)  # Set kolom 'Date' sebagai indeks
    data.dropna(inplace=True)  # Hapus baris dengan nilai yang hilang

    # Pra-pemrosesan data: Mengubah format 'Vol.' menjadi float
    data['Vol.'] = data['Vol.'].replace('-', float('nan'))  # Ganti '-' dengan NaN
    data['Vol.'] = data['Vol.'].fillna(0)  # Isi nilai NaN dengan 0
    data['Vol.'] = data['Vol.'].astype(float)  # Konversi ke float

    # Pra-pemrosesan data: Mengubah format 'Vol.' dan 'Change %' menjadi float
    data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
    
    # Split data menjadi fitur (X) dan target (y)
    X = data[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    y = data['Price']

    # Menggunakan sidebar untuk memilih parameter
    st.sidebar.header('Parameter Random Forest')
    n_estimators = st.sidebar.slider('Jumlah Estimator', min_value=1, max_value=100, value=10, step=1)
    max_features = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2'])
    max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=5, step=1)

    # Membuat model Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)

    # Membagi data menjadi set pelatihan dan set pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model
    rf.fit(X_train, y_train)

    # Evaluasi model
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)

    # Prediksi harga
    prediction = rf.predict(X_test)

    # Menampilkan hasil
    st.title('Prediksi Harga Binance dengan Random Forest')
    st.write('Jumlah Estimator:', n_estimators)
    st.write('Max Features:', max_features)
    st.write('Max Depth:', max_depth)
    st.write('Skor Pelatihan:', train_score)
    st.write('Skor Pengujian:', test_score)

    st.subheader('Hasil Prediksi Harga Binance')
    st.write(pd.DataFrame({'Tanggal': X_test.index, 'Harga Aktual': y_test, 'Harga Prediksi': prediction}))
