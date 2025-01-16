# app.py
import streamlit as st
import pickle
from utils.preprocess import preprocess_input
import os

model_path = 'model/model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
else:
    raise FileNotFoundError(f"File {model_path} tidak ditemukan.")

# Load model and vectorizer
with open('model/model.pkl', 'rb') as f:
    data = pickle.load(f)

    model = data['model']
    vectorizer = data['vectorizer']

st.title("Sistem Prediksi Bidang Penelitian")

# Input form
st.sidebar.header("Masukkan Data")
title = st.sidebar.text_input("Judul Artikel", "Contoh Judul")
abstract = st.sidebar.text_area("Abstrak Artikel", "Contoh Abstrak")

if st.sidebar.button("Prediksi"):
    # Preprocess input
    input_vector = preprocess_input(title, abstract, vectorizer)
    # Prediction
    prediction = model.predict(input_vector)
    st.write("Prediksi Bidang: ", "Computer Science" if prediction[0] == 1 else "Bukan Computer Science")

# Optional: Display sample data
if st.checkbox("Tampilkan Data Contoh"):
    import pandas as pd
    data = pd.read_csv('data/tugas1.csv')
    st.write(data.head())
