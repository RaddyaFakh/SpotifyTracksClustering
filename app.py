import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Spotify Tracks Clustering", layout="centered")

# Load model dan scaler
# Menggunakan st.cache_resource agar model hanya dimuat sekali dan aplikasi lebih cepat
@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_spotify.joblib')
    scaler = joblib.load('scaler_spotify.joblib')
    return kmeans, scaler

kmeans, scaler = load_models()

# Judul Utama
st.title('🎵 Segmentasi Lagu Spotify')
st.write('Aplikasi ini menggunakan model **K-Means Clustering** untuk mengelompokkan lagu berdasarkan karakteristik audio (audio features) ke dalam 6 profil segmen yang berbeda.')
st.markdown("---")

# Sidebar untuk Input User
st.sidebar.header('Pilih Karakteristik Audio')

def user_input_features():
    # Menyesuaikan range slider dengan standar nilai metrik Spotify
    popularity = st.sidebar.slider('Popularity', 0, 100, 50)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('Loudness (dB)', -60.0, 0.0, -10.0)
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.1)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
    tempo = st.sidebar.slider('Tempo (BPM)', 0.0, 250.0, 120.0)

    # Dictionary input
    data = {
        'popularity': popularity,
        'danceability': danceability,
        'energy': energy,
        'loudness': loudness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'tempo': tempo
    }
    # Menjadikan dictionary sebagai Pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Memanggil fungsi input
input_df = user_input_features()

# Menampilkan input user di halaman utama
st.subheader('Nilai Audio Features yang Anda Masukkan:')
st.dataframe(input_df)

# Tombol untuk memicu prediksi
if st.button('Tentukan Segmen Lagu'):
    # 1. Transform/Scale data input user menggunakan scaler yang sudah dilatih
    scaled_input = scaler.transform(input_df)
    
    # 2. Lakukan Prediksi
    prediction = kmeans.predict(scaled_input)[0]
    
    # 3. Pemetaan label Cluster ke Nama Profil (Sesuai hasil interpretasi Business Understanding)
    cluster_names = {
        0: "Mellow & Acoustic Vocal Tracks",
        1: "Mainstream / Popular Hits",
        2: "High-Energy & Fast-Paced (Rock/Metal)",
        3: "Upbeat Electronic & EDM",
        4: "Ambient, Classical & Sleeping Tracks",
        5: "Undiscovered / Underground Upbeat"
    }
    
    # 4. Tampilkan Hasil
    st.markdown("---")
    st.subheader('Hasil Segmentasi:')
    st.success(f'Lagu dengan karakteristik di atas termasuk dalam **Cluster {prediction}**')
    st.info(f'**Profil Cluster:** {cluster_names[prediction]}')