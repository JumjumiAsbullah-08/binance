import json
import streamlit as st
import requests
import pandas as pd
import locale
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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
    # Konten untuk Prediksi
    st.title("Prediksi")
    # Langkah 1: Persiapan Data
    data = pd.read_csv('C:/project-akhir/data_binance.csv')
    data['Date'] = pd.to_datetime(data['Date'])  # Mengonversi kolom 'Date' menjadi tipe datetime
    data = data.set_index('Date')  # Mengatur kolom 'Date' sebagai indeks

    # Pra-pemrosesan data: Mengubah format 'Vol.' menjadi float
    #data['Vol.'] = data['Vol.'].str.replace('K', '').astype(float) * 1000
    #data['Vol.'] = data['Vol.'].str.replace('M', '').astype(float) * 1000000

    # Pra-pemrosesan data: Mengubah format 'Vol.' menjadi float
    data['Vol.'] = data['Vol.'].replace('-', float('nan'))  # Ganti '-' dengan NaN
    data['Vol.'] = data['Vol.'].fillna(0)  # Isi nilai NaN dengan 0
    data['Vol.'] = data['Vol.'].astype(float)  # Konversi ke float

    # Pra-pemrosesan data: Mengubah format 'Vol.' dan 'Change %' menjadi float
    data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
    
    # Langkah 2: Pelatihan Model
    X = data[['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']]
    y = data['Price']

    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Langkah 4: Pembuatan Aplikasi Streamlit
    st.title('Prediksi Harga Binance dengan Random Forest')

    # Sidebar untuk memasukkan fitur-fitur
    st.sidebar.header('Masukkan Fitur')
    price = st.sidebar.number_input('Price')
    open_price = st.sidebar.number_input('Open Price')
    high_price = st.sidebar.number_input('High Price')
    low_price = st.sidebar.number_input('Low Price')
    volume = st.sidebar.number_input('Volume')
    change_percentage = st.sidebar.number_input('Change Percentage')

    # Membuat prediksi berdasarkan input pengguna
    input_features = [price, open_price, high_price, low_price, volume, change_percentage]
    prediction = rf.predict([input_features])

    # Menampilkan hasil prediksi
    st.write('Prediksi Harga Binance:', prediction)