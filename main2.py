import streamlit as st
st.set_page_config(page_title="Prediksi Gizi Balita", layout="centered")

import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
from sklearn.linear_model import LinearRegression

# Fungsi untuk encode gambar lokal jadi base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("posyandu.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
    background-color: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Model LightGBM
with open('lgbm_terlatih_80.pkl', 'rb') as f:
    model = pickle.load(f)

# Tabel Ideal Gizi berdasarkan jenis kelamin dan usia 
data_laki = pd.DataFrame({
    'Kelompok Usia (bulan)': ["0-12", "13-24", "25-36", "37-48", "49-60"],
    'Berat Ideal (kg)': [7.9, 10.9, 12.9, 14.8, 16.7],
    'Tinggi Ideal (cm)': [72.5, 81.7, 89.5, 96.3, 102.7],
    'Kategori': ["Normal"] * 5
})
data_perempuan = pd.DataFrame({
    'Kelompok Usia (bulan)': ["0-12", "13-24", "25-36", "37-48", "49-60"],
    'Berat Ideal (kg)': [7.3, 10.2, 12.4, 14.3, 16.1],
    'Tinggi Ideal (cm)': [70.5, 80.0, 87.8, 94.5, 100.7],
    'Kategori': ["Normal"] * 5
})

# Judul aplikasi
st.title("Prediksi Status Gizi Anak Berdasarkan Riwayat Penimbangan")

# Input jenis kelamin
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
jenis_kelamin_encoded = 1 if jenis_kelamin == "Laki-laki" else 0

# Input usia dan pengukuran (minimal 2 rentang)
st.subheader("Input Riwayat Penimbangan (Usia, Berat, Tinggi)")
usia_input = st.multiselect("Usia (bulan)", options=list(range(0, 61)), default=[4, 5, 6, 7, 8])

berat_input = []
tinggi_input = []
for u in usia_input:
    berat = st.number_input(f"Berat badan pada usia {u} bulan (kg)", min_value=0.0, max_value=30.0, step=0.1, key=f"berat_{u}")
    tinggi = st.number_input(f"Tinggi badan pada usia {u} bulan (cm)", min_value=30.0, max_value=150.0, step=0.1, key=f"tinggi_{u}")
    berat_input.append(berat)
    tinggi_input.append(tinggi)

# Simpan ke dalam DataFrame
data = pd.DataFrame({
    "Usia": usia_input,
    "Berat": berat_input,
    "Tinggi": tinggi_input,
    "JK": [jenis_kelamin_encoded]*len(usia_input)
})

# Tampilkan tabel
st.write("### Data Riwayat Penimbangan")
st.dataframe(data)

# Visualisasi data pertumbuhan
if not data.empty:
    st.write("### Grafik Pertumbuhan Berat dan Tinggi Badan")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Usia (bulan)')
    ax1.set_ylabel('Berat (kg)', color='tab:red')
    ax1.plot(data["Usia"], data["Berat"], color='tab:red', marker='o', label='Berat')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Tinggi (cm)', color='tab:blue')
    ax2.plot(data["Usia"], data["Tinggi"], color='tab:blue', marker='s', label='Tinggi')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    st.pyplot(fig)

# Prediksi berat & tinggi bulan depan
st.write("### Prediksi Berat dan Tinggi untuk Bulan Berikutnya")
if len(data) >= 2:
    reg_berat = LinearRegression().fit(data[["Usia"]], data["Berat"])
    reg_tinggi = LinearRegression().fit(data[["Usia"]], data["Tinggi"])

    usia_pred = max(data["Usia"]) + 1
    berat_pred = reg_berat.predict([[usia_pred]])[0]
    tinggi_pred = reg_tinggi.predict([[usia_pred]])[0]

    st.write(f"**Prediksi usia {usia_pred} bulan:**")
    st.write(f"Berat badan: {berat_pred:.2f} kg")
    st.write(f"Tinggi badan: {tinggi_pred:.2f} cm")

    berat_now = data["Berat"].values[-1]
    if berat_pred < berat_now:
        status = "Penurunan berat badan"
        rekomendasi = "Disarankan untuk konsultasi ke tenaga gizi dan memastikan asupan energi anak terpenuhi."
    elif berat_pred > berat_now:
        status = "Kenaikan berat badan"
        rekomendasi = "Pertumbuhan anak menunjukkan tren positif. Tetap jaga pola makan dan pantau secara rutin."
    else:
        status = "Berat badan stagnan"
        rekomendasi = "Pertimbangkan evaluasi lebih lanjut jika stagnasi terus terjadi dalam beberapa bulan."

    st.write(f"**Status Prediksi:** {status}")
    st.info(rekomendasi)

    # Klasifikasi status gizi BB/U
    try:
        input_model = pd.DataFrame({
            "JK": [jenis_kelamin_encoded],
            "Usia Saat Ukur": [usia_pred],
            "Berat": [berat_pred],
            "Tinggi": [tinggi_pred]
        })

        klasifikasi = model.predict(input_model)[0]
        label_map = {0: "Kurang", 1: "Normal", 2: "Lebih"}
        klasifikasi_label = label_map[klasifikasi]
        st.success(f"Prediksi Status Gizi BB/U Bulan Depan: **{klasifikasi_label}**")

        df_ideal = pd.DataFrame(data_laki if jenis_kelamin == "Laki-laki" else data_perempuan)
        st.markdown("### ℹ️ Informasi Tambahan")
        st.write(f"📌 **Kelompok usia anak Anda: `{usia_pred}` bulan**")
        st.markdown("### 📋 Tabel Ideal Berdasarkan Jenis Kelamin")
        st.dataframe(df_ideal, use_container_width=True)

        # Rekomendasi berdasarkan klasifikasi
        st.markdown("### 🧾 Rekomendasi Gizi")
        if klasifikasi_label == "Kurang":
            st.warning("""
            Anak Anda termasuk kategori **kurang**. Pastikan makan 3–5 kali sehari dengan porsi kecil, beri camilan sehat, dan nutrisi cukup.
            Jika porsi makan sedikit, ubah jadwal makannya menjadi 4-5x sehari dengan porsi yang kecil.
            Kenalkan anak dengan konsep kenyang dan lapar, berikan camilan sehat, dan hindari minuman tinggi gula.
            Jika diperlukan, segera konsultasikan dengan tenaga kesehatan atau dokter gizi.
            """)
        elif klasifikasi_label == "Lebih":
            st.warning("""
            Anak Anda termasuk kategori **lebih**. Terapkan pola makan sehat, kurangi makanan manis dan tinggi lemak.
            Dorong aktivitas fisik rutin dan batasi waktu pasif anak. Diskusikan dengan ahli gizi jika diperlukan.
            """)
        else:
            st.success("Status gizi anak Anda tergolong **normal**. Lanjutkan pemantauan berkala dan pastikan kebutuhan gizinya tetap terpenuhi.")

    except FileNotFoundError:
        st.warning("Model LightGBM tidak ditemukan. Pastikan file 'lgbm_terlatih_80.pkl' tersedia.")

else:
    st.warning("Minimal dua titik data diperlukan untuk melakukan prediksi.")
