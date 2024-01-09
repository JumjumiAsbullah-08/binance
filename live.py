from sklearn.metrics import mean_squared_error
import numpy as np

# Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung selisih antara prediksi dan nilai sebenarnya
errors = y_pred - y_test

# Kuadratkan selisih
squared_errors = np.square(errors)

# Hitung rata-rata dari kuadrat selisih
mse = np.mean(squared_errors)

# Ambil akar kuadrat untuk mendapatkan RMSE
rmse = np.sqrt(mse)

print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
