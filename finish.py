import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import pytz
import locale
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta, date

favicon_path = 'assets/binance.ico'
st.set_page_config(page_title="Prediksi Harga Binance", page_icon=favicon_path)

# untuk halaman css
hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
with st.sidebar:
    # Menampilkan gambar di sidebar
    st.image("assets/binance.png")

    # Opsi menu
    menu_options = ["Dashboard", "Data", "Prediksi", "Tentang"]
    selected_menu = st.sidebar.selectbox("Pilih Menu", menu_options)
if selected_menu == "Dashboard":
    # Konten untuk Dashboard
    # Membuat judul
    st.title("Tabel Live Top Cryptocurrency")

    # Fungsi untuk mengambil data top cryptocurrency dari API CoinGecko
    def get_top_cryptos(limit):
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False,
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return []

    # Mengambil data top 10 cryptocurrency dari API CoinGecko
    cryptos = get_top_cryptos(10)

    # Membuat DataFrame dari data cryptocurrency
    crypto_df = pd.DataFrame(cryptos)
    crypto_df.index = crypto_df.index + 1  # Penomoran dimulai dari 1
    columns = ["name", "symbol", "current_price", "market_cap", "total_volume", "price_change_percentage_24h"]
    crypto_df = crypto_df[columns]

    # Mengganti nama kolom dengan nama yang lebih pendek
    new_column_names = {
        "name": "Nama",
        "symbol": "Simbol",
        "current_price": "Harga (USD)",
        "market_cap": "Market Cap",
        "total_volume": "Vol (24H)",
        "price_change_percentage_24h": "Chg (24H)"
    }
    crypto_df.rename(columns=new_column_names, inplace=True)

    # Menambahkan simbol dollar dan persentase pada kolom tertentu
    crypto_df["Harga (USD)"] = crypto_df["Harga (USD)"].apply(lambda x: f"${x:.2f}")
    crypto_df["Market Cap"] = crypto_df["Market Cap"].apply(lambda x: f"${x:,}")
    crypto_df["Vol (24H)"] = crypto_df["Vol (24H)"].apply(lambda x: f"${x:,}")
    crypto_df["Chg (24H)"] = crypto_df["Chg (24H)"].apply(lambda x: f"{x:.2f}%")

    # Menampilkan tabel live top cryptocurrency
    if not crypto_df.empty:
        st.write("Top 10 Cryptocurrency:")
        st.dataframe(crypto_df)
    else:
        st.write("Gagal mengambil data top cryptocurrency.")
        
    # Background and Problem Research
    st.title("Background and Problem Research")
    st.write("Latar Belakang :")
    st.markdown(
        """
        <div style="text-align: justify;">
            Dalam beberapa tahun terakhir, pasar mata uang kripto telah berkembang pesat. Binance adalah salah satu platform pertukaran aset kripto terbesar di dunia, yang memungkinkan pengguna untuk membeli, menjual, dan menukar berbagai jenis mata uang kripto. Harga aset kripto seperti Bitcoin, Ethereum, dan lainnya sangat fluktuatif, dan prediksi harga yang akurat memiliki potensi untuk memberikan keuntungan besar bagi para trader dan investor. Blockchain adalah teknologi dasar di balik mata uang kripto yang mencatat transaksi secara terdesentralisasi dan aman. Informasi yang tersimpan dalam blockchain mencakup jejak transaksi, data pasar, dan banyak detail lainnya. Keterkaitan antara informasi blockchain dengan pergerakan harga aset kripto menawarkan peluang bagi pengembangan model prediksi harga yang lebih baik. <br>
            Dibawah ini merupakan grafik pergerakan nilai harga Binance dari Januari 2018 hingga 15 Juni 2023. Grafik tersebut menunjukkan bahwa nilai harga Binance sangat tidak stabil, dengan fluktuasi yang tinggi dan perubahan yang cepat, baik naik maupun turun. Terlihat pada grafik bahwa beberapa tahun nilai harga binance mengalami kenaikan yang sangat drastis, seperti pada tahun 2021. Kemudian mengalami penurun yang signifikan di bulan juli tahun 2021 dan mengalami kenaikan lagi di awal tahun 2022.
        </div>
        """, unsafe_allow_html=True
    )
    st.image("assets/data bnb.jpg")
    st.markdown(
        """
            <hr> 
        """, unsafe_allow_html=True
    )
    st.write("Problem Research :")
    st.markdown (
        """
        <div style="text-align: justify;">
            Tujuan dari penelitian ini adalah untuk mengembangkan model prediksi harga aset kripto Binance (BNB) berdasarkan informasi dari blockchain. Model ini akan menggunakan algoritma Random Forest, yang merupakan salah satu teknik dalam machine learning yang kuat untuk masalah regresi dan klasifikasi. Fokus utama adalah pada regresi, yaitu memprediksi nilai numerik (harga) berdasarkan berbagai fitur atau variabel. </br>
            Penelitian ini akan berfokus pada membangun model prediksi harga menggunakan algoritma Random Forest berdasarkan informasi dari blockchain Binance. Diharapkan bahwa hasil penelitian ini akan memberikan wawasan berharga tentang kemungkinan keterkaitan antara informasi blockchain dan pergerakan harga aset kripto, meskipun masih perlu diingat bahwa pasar kripto memiliki sifat yang sangat dinamis dan sulit diprediksi sepenuhnya.
        </div>
        """, unsafe_allow_html=True
    )
elif selected_menu == "Data":
    # konten prediksi
    st.title("Data Set dan Algoritma Random Forest")
    st.markdown(
        """
            <h3>Data Set</h3>
            <p style="text-align:justify;">
            Dataset diterima dari <a href="https://depository.id/">PT. Tennet Depository Indonesia</a> mencakup berbagai informasi penting mengenai harga aset kripto Binance (BNB). Data ini mencakup periode waktu tertentu dan berisi Tujuh kolom yang relevan:
            </p>
            <ol style="text-align:justify;">
                <li>Date (Tanggal): Kolom ini mencatat tanggal observasi atau periode waktu tertentu dalam format tanggal. Informasi tanggal ini digunakan untuk mengatur data dalam urutan kronologis.</li>
                <li>Price (Harga): Kolom ini mencatat harga penutupan aset kripto pada tanggal tertentu. Harga penutupan adalah harga terakhir pada periode perdagangan tertentu dan digunakan sebagai acuan untuk analisis.</li>
                <li>Open (Harga Pembukaan): Kolom ini mencatat harga pembukaan aset kripto pada tanggal tertentu. Harga pembukaan adalah harga pertama pada periode perdagangan tersebut.</li>
                <li>High (Harga Tertinggi): Kolom ini mencatat harga tertinggi yang dicapai oleh aset kripto selama periode perdagangan pada tanggal tertentu. Informasi ini memberikan gambaran tentang volatilitas aset selama periode tersebut.</li>
                <li>Low (Harga Terendah): Harga terendah adalah harga minimum yang dicapai oleh instrumen keuangan selama periode waktu tertentu. Ini menggambarkan tingkat harga terendah yang dicapai oleh instrumen itu selama periode itu.</li>
                <li>Vol. (Volume Perdagangan): Kolom ini mencatat volume perdagangan aset kripto pada tanggal tertentu. Volume perdagangan mengindikasikan seberapa besar aktivitas perdagangan yang terjadi pada periode tersebut.</li>
                <li>Change % (Persentase Perubahan): Kolom ini mencatat persentase perubahan harga aset kripto dari harga pembukaan hingga harga penutupan pada tanggal tertentu. Informasi ini memberikan gambaran tentang seberapa besar pergerakan harga selama periode perdagangan tersebut.</li>            
            </ol>
            <p style="text-align:justify;">
            Jumlah data yang akan diambil yaitu 1992 data sample dan jumlah total data histori binance yaitu sebanyak 2045 data. Persentase data sampling adalah sekitar 97.38% dari total data yang tersedia dalam populasi sebagai sampel. Berikut Tabel sampling dataset :
            """, unsafe_allow_html=True
    )
    # Membaca dataset dari file CSV
    dataset_path = r"C:\project-akhir\BNB_LIVE.xlsx"
    df = pd.read_excel(dataset_path)
    
    # Menggantikan indeks DataFrame dengan nomor dari 1 hingga 1992
    df.index = range(1, len(df) + 1)
    
    # Menampilkan dataset
    st.write(df)
    st.markdown(
        """
            <p style="text-align:justify;">
            Dataset ini berharga untuk analisis dan prediksi harga aset kripto. Dengan memanfaatkan informasi tanggal, harga pembukaan, harga penutupan, harga tertinggi, volume perdagangan, dan persentase perubahan, dapat dilakukan berbagai analisis, termasuk analisis tren harga, volatilitas, dan pola perdagangan. Selain itu, dataset ini juga dapat digunakan untuk melatih model prediksi harga menggunakan teknik seperti algoritma Random Forest. Dengan memahami dan menganalisis data ini, dapat diperoleh wawasan berharga tentang dinamika pasar aset kripto Binance (BNB).
            </p>
        """, unsafe_allow_html=True
    )
    def get_coingecko_price():
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "binancecoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data["binancecoin"]["usd"]
        else:
            return None

    def get_coingecko_historical_data(days):
        url = f"https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {"vs_currency": "usd", "days": days}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            return prices
        else:
            return []

    # Mengambil data harga koin saat ini dari API CoinGecko
    price = get_coingecko_price()

    # Mengambil data historis pergerakan harga koin dalam X hari terakhir
    historical_data = get_coingecko_historical_data(30)  # Ambil data 30 hari terakhir

    # Menampilkan harga koin Binance (BNB)
    if price is not None:
        st.markdown(f"<h5 style='background-color:#f8bc2c; width:400px; color:black; border-radius:5px; height:25px;'>Harga BNB/USD saat ini: {price} USD</h5>", unsafe_allow_html=True)

        # Membuat grafik area dengan Plotly
        x = [datetime.utcfromtimestamp(data[0] / 1000) for data in historical_data]
        y = [data[1] for data in historical_data]
        fig = go.Figure(data=[go.Scatter(x=x, y=y, fill='tozeroy')])
        fig.update_layout(title='Grafik Harga Binance Coin (BNB)',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga (USD)')
        st.plotly_chart(fig)
    else:
        st.write("Gagal mengambil data harga koin.")
    def get_historical_bnb_data(limit=10):
        url = "https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            
            if len(prices) >= limit:
                prices = prices[-limit:]
            
            return prices
        else:
            return []

    # Mengambil data historis Binance Coin (BNB) dari API CoinGecko
    bnb_prices = get_historical_bnb_data(100)  # Misalnya, ambil 100 data terakhir

    # Menyiapkan data untuk grafik
    dates = [pd.to_datetime(price[0], unit='ms', utc=True) for price in bnb_prices]
    prices_usd = [price[1] for price in bnb_prices]

    # Konversi zona waktu UTC ke WIB
    wib_timezone = pytz.timezone("Asia/Jakarta")
    dates_wib = [date.astimezone(wib_timezone) for date in dates]

    # Membuat grafik live
    fig = go.Figure(data=go.Scatter(x=dates_wib, y=prices_usd))
    fig.update_layout(title="Grafik Live Harga Binance Coin (BNB) dalam Waktu (WIB)",
                    xaxis_title="Waktu (WIB)",
                    yaxis_title="Harga (USD)")

    # Menampilkan grafik live menggunakan Streamlit
    st.plotly_chart(fig)
    st.markdown(
        """
            <hr>
            <h3>Algoritma Random Forest</h3>
            <p style="text-align:justify;">
            Random Forest adalah sebuah algoritma pembelajaran mesin yang termasuk dalam kategori Ensemble Learning. Algoritma ini menggunakan konsep penggabungan beberapa pohon keputusan (decision tree) independen untuk melakukan proses klasifikasi atau regresi. Setiap pohon keputusan dalam Random Forest dilatih menggunakan subset acak dari data pelatihan, dan hasil akhir dari algoritma ini didapatkan melalui penggabungan atau voting dari keputusan yang dihasilkan oleh setiap pohon keputusan. Dengan memanfaatkan ansambel dari pohon-pohon keputusan, Random Forest dapat menghasilkan prediksi yang lebih akurat dan memiliki kemampuan untuk menangani kompleksitas data yang tinggi.
            </p>
            <p>Dibawah ini merupakan Pohon Keputusan (Decision Tree) dari Algoritma Random Forest</p>
            """, unsafe_allow_html=True
    )  
    st.image("assets/random forest.jpg")
    st.markdown(
        """
            <p style="text-align:justify;">Pohon keputusan individu memberikan suara untuk hasil kelas dalam contoh mainan random forest. (A) Dataset masukan ini menggambarkan tiga sampel, di mana lima fitur (x1, x2, x3, x4, dan x5) menjelaskan setiap sampel. (B) Pohon keputusan terdiri dari cabang yang bercabang pada titik keputusan. Setiap titik keputusan memiliki aturan yang menentukan apakah sampel akan masuk ke cabang satu atau cabang lain tergantung pada nilai fitur. Cabang-cabang tersebut berakhir pada daun yang termasuk dalam kelas merah atau kelas kuning. Pohon keputusan ini mengklasifikasikan sampel 1 ke kelas merah. (C) Pohon keputusan lainnya, dengan aturan yang berbeda pada setiap titik keputusan. Pohon ini juga mengklasifikasikan sampel 1 ke kelas merah. (D) Random forest menggabungkan suara dari pohon keputusan konstituennya, yang menghasilkan prediksi kelas akhir. (E) Prediksi output akhirnya juga adalah kelas merah (Denisko and Hoffman, 2018).</p>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <hr>
        <h3>Pengujian dan Pengukuran Sistem</h3>
        <p style="text-align:justify;">Dalam setiap pengujian dan pengukuran sistem dapat menggunakan metrik evaluasi, yaitu Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE). Root Mean Squared Error (RMSE) merupakan ukuran besarnya kesalahan hasil prediksi, di mana semakin mendekati 0, nilai RMSE menunjukkan bahwa hasil prediksi semakin akurat. Dengan menggunakan persamaan tersebut, dapat menghitung nilai RMSE untuk mengevaluasi akurasi model prediksi. Semakin kecil nilai RMSE, semakin dekat prediksi model dengan nilai aktual, sehingga menunjukkan performa model yang lebih baik.</p>
        <span>Rumus RMSE (Root Mean Square Error) :</span>
        """, unsafe_allow_html=True
    )
    st.markdown(r'$$\text{RMSE} = \sqrt{ \sum \frac{(X_i - \hat{Y}_i)}{n}^2}$$')
    st.markdown(
        """
        <p style="text-align:justify;">Mean Absolute Percentage Error (MAPE) adalah metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan relatif dari prediksi model dalam bentuk persentase. MAPE menghitung persentase rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya, dibagi dengan nilai sebenarnya, dan kemudian diambil rata-ratanya.</p>
        <span>Rumus MAPE (Mean Absolute Percentage Error) :</span>
        """, unsafe_allow_html=True
    )
    st.markdown(r'$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_i -  ŷ_i}{ŷ_i} \right| x 100\%$$')
elif selected_menu == "Prediksi":
    st.title("Data Live Binance (BNB)")
    def get_latest_bnb_data():
        url = "https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            if len(prices) > 0 and len(volumes) > 0:
                latest_price_bnb = prices[-1][1]
                open_price_bnb = prices[0][1]
                high_price_bnb = max(price[1] for price in prices)
                low_price_bnb = min(price[1] for price in prices)
                volume_bnb = volumes[-1][1] / 1000  # Mengubah volume ke format ribu (K)
                change_percentage = ((latest_price_bnb - open_price_bnb) / open_price_bnb) * 100
                return latest_price_bnb, open_price_bnb, high_price_bnb, low_price_bnb, volume_bnb, change_percentage
            else:
                return None, None, None, None, None, None
        else:
            return None, None, None, None, None, None

    # Mengambil data harga Binance Coin (BNB) dari API CoinGecko
    latest_price_bnb, open_price_bnb, high_price_bnb, low_price_bnb, volume_bnb, change_percentage = get_latest_bnb_data()

    # Mendapatkan tanggal saat ini
    current_date = date.today().strftime("%Y-%m-%d")

    # Menampilkan harga Binance Coin (BNB), harga pembuka, harga tertinggi, harga terendah, volume, persentase perubahan, dan tanggal dalam tabel
    if latest_price_bnb is not None:
        # Format angka desimal menjadi dua digit di kolom "Price (USD)" dan "Vol."
        latest_price_bnb = "{:.2f}".format(latest_price_bnb)
        open_price_bnb = "{:.2f}".format(open_price_bnb)
        high_price_bnb = "{:.2f}".format(high_price_bnb)
        low_price_bnb = "{:.2f}".format(low_price_bnb)
        
        # Menambahkan koma di antara setiap tiga digit pada volume
        locale.setlocale(locale.LC_ALL, '')  # Menggunakan pengaturan regional default
        volume_bnb = locale.format_string("%d", int(volume_bnb), grouping=True)
        
        df = pd.DataFrame({
            "Tanggal": [current_date],
            "Harga (USD)": [latest_price_bnb],
            "Harga Pembuka (USD)": [open_price_bnb],
            "Harga Tertinggi(USD)": [high_price_bnb],
            "Harga Terendah(USD)": [low_price_bnb],
            "Vol (24H)": [volume_bnb],
            "Persentase Perubahan %": [change_percentage]
        })
        st.write("Live Data Harga Binance (BNB) dalam USD:")
        st.dataframe(df)
    else:
        st.write("Gagal mengambil data harga Binance Coin (BNB) dari API CoinGecko.")
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

    # Predict function
    def predict_price(open_price, high_price, low_price, volume, change_percentage, prediction_date):
        features = [[open_price, high_price, low_price, volume, change_percentage]]
        prediction = model.predict(features)
        return prediction[0]

    # Function to get live BNB price from Coingecko API
    def get_live_bnb_price_previous_day(date):
        # Calculate the previous day's date
        previous_date = (date - pd.Timedelta(days=1))
        endpoint = "https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {
            "vs_currency": "usd",
            "from": previous_date,
            "to": date,
            "days": "1",
        }

        try:
            response = requests.get(endpoint, params=params)
            data = response.json()
            if "prices" in data and len(data["prices"]) > 0:
                live_bnb_price = data["prices"][0][1]
                return live_bnb_price
            else:
                st.warning("Data harga Binance Coin tidak tersedia untuk tanggal ini.")
                return None
        except Exception as e:
            st.warning("Gagal mengambil data harga Binance Coin.")
            return None

    # Function to convert Binance price to Rupiah
    def convert_to_rupiah(price):
        response = requests.get(f"https://open.er-api.com/v6/latest/USD")
        data = response.json()
        exchange_rate = data["rates"]["IDR"]
        rupiah_price = price * exchange_rate
        return rupiah_price

    # Function to get historical data for 30 days
    def get_coingecko_historical_data(days):
        url = f"https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {"vs_currency": "usd", "days": days}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            return prices
        else:
            return []
    # Streamlit app
    st.title('Prediksi Harga Binance')
    st.warning('Harap isi semua input untuk melakukan prediksi')
    
    # Create two columns for input features
    col1, col2 = st.columns(2)
    
    # Input features for prediction in the first column
    with col1:
        open_price = st.number_input('Harga Pembukaan (USD) *')
        high_price = st.number_input('Harga Tertinggi (USD) *')
        low_price = st.number_input('Harga Terendah (USD) *')

    # Input features for prediction in the second column
    with col2:
        volume = st.number_input('Volume (24H) *')
        change_percentage = st.number_input('Perubahan Persentase (%) *')
        prediction_date = st.date_input('Tanggal Prediksi *')
    # Button to trigger prediction
    if st.button('Prediksi'):
        if open_price and high_price and low_price and volume and change_percentage and prediction_date:
            # Predict the price
            prediction = predict_price(open_price, high_price, low_price, volume, change_percentage, prediction_date)
            # Calculate MAPE
            actual_price = get_live_bnb_price_previous_day(prediction_date)  # Get the actual price from the previous day
            if actual_price is not None:
                mape = (abs(actual_price - prediction) / actual_price) * 100
                # Calculate MAE
                # mae = abs(actual_price - prediction)
            # Convert price to Rupiah
            rupiah_prediction = convert_to_rupiah(prediction)
            st.success(f'Prediksi Harga Binance pada tanggal [{prediction_date}]: $ {prediction:.2f} USD | Rp. {locale.format_string("%.2f", rupiah_prediction, grouping=True)}', icon="✅")

            # Get live BNB price for the previous day
            live_bnb_price_previous_day = get_live_bnb_price_previous_day(prediction_date)

            if live_bnb_price_previous_day is not None:
                # Convert the live price to Rupiah
                live_rupiah_price_previous_day = convert_to_rupiah(live_bnb_price_previous_day)
                # st.info(f'Harga BNB Live pada tanggal sebelumnya [{prediction_date - pd.Timedelta(days=1)}]: $ {live_bnb_price_previous_day:.2f} USD | Rp. {locale.format_string("%.2f", live_rupiah_price_previous_day, grouping=True)}', icon="ℹ️")
                # Tampilkan harga BNB terkini menggunakan st.info
                st.info(f'Harga BNB Live pada tanggal sebelumnya: [{prediction_date - pd.Timedelta(days=1)}]: $ {latest_price_bnb} USD | Rp. {locale.format_string("%.2f", convert_to_rupiah(float(latest_price_bnb)), grouping=True)}', icon="ℹ️")
                # Get historical data for the last 30 days
                historical_data = get_coingecko_historical_data(30)
                if historical_data:
                    x_live = [datetime.utcfromtimestamp(data[0] / 1000).date() for data in historical_data]
                    y_live = [data[1] for data in historical_data]

                    # Create a DataFrame for live data
                    live_data = {
                        'Tanggal': x_live,
                        'Harga': y_live,
                        'Tipe Data': ['Harga Live BNB'] * len(x_live)
                    }

                    # Create a DataFrame for the prediction
                    prediction_data = {
                        'Tanggal': [prediction_date],
                        'Harga': [prediction],
                        'Tipe Data': ['Prediksi']
                    }

                    # Combine the live data and prediction data into one DataFrame
                    combined_data = {
                        'Tanggal': x_live + [prediction_date],
                        'Harga': y_live + [prediction],
                        'Tipe Data': ['Harga Live BNB'] * len(x_live) + ['Prediksi']
                    }

                    df_combined = pd.DataFrame(combined_data)

                    # Create a Plotly Express line chart for combined data
                    fig = px.line(df_combined, x='Tanggal', y='Harga', color='Tipe Data', markers=True, title='Harga Live Binance (BNB) vs Hasil Prediksi')
                    fig.update_xaxes(title_text='Tanggal')
                    fig.update_yaxes(title_text='Harga (USD)')

                    # Show the combined data chart in Streamlit
                    st.plotly_chart(fig)

                    # Calculate RMSE
                    y_pred = model.predict(X_test)
                    errors = y_pred - y_test
                    squared_errors = np.square(errors)
                    mse = np.mean(squared_errors)
                    rmse = np.sqrt(mse)
                    # Calculate the range of actual data
                    range_actual = y_test.max() - y_test.min()

                    # Calculate RMSE in percentage
                    rmse_percentage = (rmse / range_actual) * 100

                    with st.expander(f'Mean Absolute Percentage Error (MAPE): {mape:.2f} %'):
                        st.write("MAPE (Mean Absolute Percentage Error) adalah metrik untuk mengukur tingkat kesalahan dalam prediksi perbandingan persentase antara prediksi dan data aktual. MAPE mengukur akurasi model dalam meramalkan data dalam bentuk persentase, di mana nilai lebih rendah menunjukkan prediksi yang lebih akurat. <br> Hasil Nilai Mape:"f' {mape:.2f}' " %", unsafe_allow_html=True)
                    
                    # st.info(f'Mean Absolute Error (MAE): $ {mae:.2f} USD', icon="ℹ️")
                    with st.expander(f'Root Mean Square Error (RMSE): {rmse:.2f} | Persentase RMSE: {rmse_percentage:.2f}%'):
                        st.write("RMSE (Root Mean Square Error) adalah metrik untuk mengukur tingkat kesalahan rata-rata antara prediksi dan data aktual. RMSE mengukur akurasi model dalam memprediksi data dalam satuan yang sama dengan data aktual. Nilai RMSE yang lebih rendah menunjukkan prediksi yang lebih akurat. <br> Hasil Nilai RMSE:"f' {rmse:.2f}'" <hr> Persentase RMSE (Percentage RMSE) adalah nilai RMSE yang telah diubah menjadi persentase dari rentang data aktual. Ini membantu dalam memahami sejauh mana prediksi model mendekati variasi data aktual dalam bentuk persentase. <br> Hasil Persentase RMSE:"f' {rmse_percentage:.2f} %', unsafe_allow_html=True)
                    
                    # # Create data for the pie chart
                    labels = ['MAPE', 'RMSE', 'RMSE Percentage']
                    values = [mape, rmse, rmse_percentage]
                    
                    # Membuat grafik area untuk MAPE, RMSE, dan Persentase RMSE
                    fig_bar = go.Figure()

                    # Menambahkan trace untuk MAPE
                    fig_bar.add_trace(go.Scatter(x=['MAPE', 'MAPE'], y=[0, mape], fill='tozeroy', mode='lines', name='MAPE', line=dict(color='#c4a808')))

                    # Menambahkan trace untuk RMSE
                    fig_bar.add_trace(go.Scatter(x=['RMSE', 'RMSE'], y=[0, rmse], fill='tozeroy', mode='lines', name='RMSE', line=dict(color='#db146a')))

                    # Menambahkan trace untuk Persentase RMSE
                    fig_bar.add_trace(go.Scatter(x=['Persentase RMSE', 'Persentase RMSE'], y=[0, rmse_percentage], fill='tozeroy', mode='lines', name='Persentase RMSE', line=dict(color='#05BFDB')))

                    # Membuat grafik batang untuk MAPE, RMSE, dan Persentase RMSE
                    fig_bar = px.bar(x=['MAPE', 'RMSE', 'Persentase RMSE'],
                                    y=[mape, rmse, rmse_percentage],
                                    color=['MAPE', 'RMSE', 'Persentase RMSE'],
                                    labels={'y': 'Nilai'},
                                    title='Grafik Batang MAPE, RMSE, Persentase RMSE',
                                    barmode='group',  # Ini mengontrol lebar batang
                                    width=400)

                    # Menambahkan nilai pada batang grafik
                    for i in range(len(fig_bar.data)):
                        for j in range(len(fig_bar.data[i].y)):
                            fig_bar.add_annotation(
                                x=fig_bar.data[i].x[j],
                                y=fig_bar.data[i].y[j],
                                text=f'{fig_bar.data[i].y[j]:.2f}',  # Menambahkan nilai dengan dua desimal
                                showarrow=True,
                                arrowhead=4,
                                ax=0,
                                ay=-40  # Penyesuaian posisi teks
                            )

                    # Menampilkan grafik batang pada Streamlit
                    st.plotly_chart(fig_bar)
                    
                    # # Create a pie chart using Plotly Express
                    fig = px.pie(
                        names=labels, values=values,
                        title='Grafik Lingkaran Hasil Perhitungan MAPE, RMSE, dan Persentase RMSE',
                        color_discrete_sequence=['#c4a808', '#db146a', '#05BFDB'],
                        hole=0.4  # Set the hole size to create a doughnut chart

                    )
                    
                    # # Show the pie chart in Streamlit
                    st.plotly_chart(fig)
                    # Plot the scatterplot
                    plt.figure(figsize=(8, 6))
                    plt.scatter(y_test, y_pred, c='blue', label='Prediksi vs. Aktual')
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Garis Referensi')
                    plt.xlabel('Aktual')
                    plt.ylabel('Prediksi')
                    plt.title('Scatterplot Prediksi vs. Aktual')
                    plt.legend()
                    plt.show()
                    
                else:
                    st.warning("Gagal mengambil data historis harga Binance Coin.")
        else:
            st.warning("Mohon isi semua inputan untuk melakukan prediksi!", icon="⚠️")

    # Button to refresh the page
    if st.button('Reset'):
        st.experimental_rerun()
elif selected_menu == "Tentang":
   # Membagi layout menjadi 2 kolom
    col1, col2 = st.columns(2)

    # Gambar dan Biodata untuk Pembimbing Artikel Jurnal
    gambar_pembimbing_path = "assets/pak sam.jpg"
    biodata_pembimbing = """
    **Nama :** Samsudin, ST,M.Kom

    **Dosen :** Pembimbing Artikel Jurnal 

    **Publikasi :**
    - ID Sinta : 6003868
    - ID Scopus : 57209425430
    - Google Scholar : https://scholar.google.co.id/citations?user=_QmOWZ4AAAAJ&hl=id
    - ORCID : https://orcid.org/0000-0003-2219-2747 
    """
    col1.image(gambar_pembimbing_path, caption="Pembimbing Artikel Jurnal", use_column_width=True, width=150)
    col1.markdown("---")  # Garis pembatas
    col2.markdown(biodata_pembimbing)

    col1, col2 = st.columns(2)
    # Gambar dan Biodata untuk Penguji Artikel Jurnal
    gambar_penguji_path = "assets/buk ase.jpg"
    biodata_penguji = """
    **Nama :** Triase, ST, M.Kom

    **Dosen :** Penguji Artikel Jurnal

    **Publikasi :**
    - ID Sinta : 6003623
    - ID Scopus : -
    - Google Scholar : https://scholar.google.com/citations?hl=id&user=oif4ZPgAAAAJ
    - ORCID : -
    """
    col1.image(gambar_penguji_path, caption="Penguji Artikel Jurnal", use_column_width=True, width=150)
    col1.markdown("---")  # Garis pembatas
    col2.markdown(biodata_penguji)
    
    col1, col2 = st.columns(2)
    # Gambar Profil Sendiri
    gambar_profil_path = "assets/jumi.png"
    col1.image(gambar_profil_path, caption="Mahasiswa", use_column_width=True, width=150)

    # Biodata Anda
    biodata_anda = """
    **Nama :** Jumjumi Asbullah

    **Prodi :** Sistem Informasi-2

    **Publikasi :**
    - ID Sinta : -
    - ID Scopus : -
    - Google Scholar : -
    - ORCID : https://orcid.org/0009-0000-0952-9312
    """
    col2.markdown(biodata_anda)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)