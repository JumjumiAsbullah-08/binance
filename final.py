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
    crypto_df["No"] = crypto_df.index + 1
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
    dataset_path = r"C:\project-akhir\BNB.xlsx"
    df = pd.read_excel(dataset_path)
    
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
    # Function to get live BNB data from CoinGecko API
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
                volume_bnb = volumes[-1][1] / 1000  # Convert volume to thousands
                change_percentage = ((latest_price_bnb - open_price_bnb) / open_price_bnb) * 100
                return latest_price_bnb, open_price_bnb, high_price_bnb, low_price_bnb, volume_bnb, change_percentage
            else:
                return None, None, None, None, None, None
        else:
            return None, None, None, None, None, None

    # Fungsi untuk mendapatkan data historis dari Coingecko untuk beberapa bulan ke belakang
    def get_coingecko_historical_data_months(months):
        today = date.today()
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - pd.DateOffset(months=months)).strftime("%Y-%m-%d")

        url = f"https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {"vs_currency": "usd", "from": start_date, "to": end_date, "days": 1}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            return prices
        else:
            return []
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

    with col2:
        volume = st.number_input('Volume (24H) *')
        change_percentage = st.number_input('Perubahan Persentase (%) *')
        prediction_month = st.number_input('Jumlah Bulan untuk Prediksi *', min_value=1, max_value=12)

    # Button to trigger prediction for multiple months
    if st.button('Prediksi untuk Beberapa Bulan'):
        if open_price and high_price and low_price and volume and change_percentage and prediction_month:
            # Predict the price for multiple months
            prediction_dates = [date.today() + pd.DateOffset(months=i) for i in range(1, prediction_month + 1)]
            predictions = [predict_price(open_price, high_price, low_price, volume, change_percentage, date) for date in
                        prediction_dates]

            # Get historical data for the last N months
            historical_data = get_coingecko_historical_data_months(prediction_month)

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
                    'Tanggal': prediction_dates,
                    'Harga': predictions,
                    'Tipe Data': ['Prediksi'] * prediction_month
                }

                # Combine the live data and prediction data into one DataFrame
                combined_data = {
                    'Tanggal': x_live + prediction_dates,
                    'Harga': y_live + predictions,
                    'Tipe Data': ['Harga Live BNB'] * len(x_live) + ['Prediksi'] * prediction_month
                }

                df_combined = pd.DataFrame(combined_data)

                # Create a Plotly Express line chart for combined data
                fig = px.line(df_combined, x='Tanggal', y='Harga', color='Tipe Data', markers=True,
                            title='Harga Live Binance (BNB) vs Hasil Prediksi')
                fig.update_xaxes(title_text='Tanggal')
                fig.update_yaxes(title_text='Harga (USD)')

                # Show the combined data chart in Streamlit
                st.plotly_chart(fig)

                    

    # Button to refresh the page
    if st.button('Reset'):
        st.experimental_rerun()
elif selected_menu == "Tentang":
    # konten prediksi
    st.title("About Me")


st.markdown(hide_streamlit_style, unsafe_allow_html=True)