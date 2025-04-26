import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing, encoder_Mode_pendaftaran, encoder_Urutan_pendaftaran, encoder_Program_studi, encoder_Waktu_kuliah, encoder_Kualifikasi_sebelumnya, encoder_Kebutuhan_pendidikan_khusus, encoder_Jenis_kelamin, encoder_Penerima_beasiswa
from prediction import prediction

import streamlit as st

col1, col2 = st.columns([1, 4])  # Lebar kolom 1 lebih kecil dari kolom 2

with col1:
    st.image(
        "https://raw.githubusercontent.com/M-Mahfudl-Awaludin/Laskar-AI/4aa066fe97e55d722f237f11525e055f1ac22431/Belajar%20Penerapan%20Data%20Science/Submission2/assets/logo-mahasiswa-png-8.png",
        width=110
    )

with col2:
    st.markdown("""
        <h1 style="font-size:32px; color:#1f77b4; margin-bottom:0;">Status Mahasiswa App</h1>
        <p style="font-size:16px; color:gray; margin-top:0;">Prototype untuk prediksi dan visualisasi status mahasiswa</p>
    """, unsafe_allow_html=True)


data = pd.DataFrame()

st.subheader("📝 Data Pendaftaran Mahasiswa")
with st.expander("📂 Informasi Pendaftaran"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Mode_pendaftaran = st.selectbox(label='Mode Pendaftaran', options=encoder_Mode_pendaftaran.classes_, index=0)
    data["Mode_pendaftaran"] = [Mode_pendaftaran]
    with col2:
        Urutan_pendaftaran = st.selectbox(label='Urutan pendaftaran', options=encoder_Urutan_pendaftaran.classes_, index=0)
    data["Urutan_pendaftaran"] = [Urutan_pendaftaran]
    with col3:
        Program_studi = st.selectbox(label='Program studi', options=encoder_Program_studi.classes_, index=0)
    data["Program_studi"] = [Program_studi]

st.subheader("📅 Jadwal & Kualifikasi")
with st.expander("📚 Waktu Kuliah & Latar Belakang"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Waktu_kuliah = st.selectbox(label='Waktu kuliah', options=encoder_Waktu_kuliah.classes_, index=0)
    data["Waktu_kuliah"] = [Waktu_kuliah]
    with col2:
        Kualifikasi_sebelumnya = st.selectbox(label='Kualifikasi sebelumnya', options=encoder_Kualifikasi_sebelumnya.classes_, index=0)
    data["Kualifikasi_sebelumnya"] = [Kualifikasi_sebelumnya]
    with col3:
        Nilai_kualifikasi_sebelumnya = float(st.number_input(label='Nilai kualifikasi sebelumnya', value=0.0))
    data["Nilai_kualifikasi_sebelumnya"] = Nilai_kualifikasi_sebelumnya

st.subheader("📊 Data Personal & Beasiswa")
with st.expander("🧍 Informasi Pribadi"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Nilai_penerimaan = float(st.number_input(label='Nilai penerimaan', value=0.0))
    data["Nilai_penerimaan"] = Nilai_penerimaan
    with col2:
        Kebutuhan_pendidikan_khusus = st.selectbox(label='Kebutuhan P.khusus', options=encoder_Kebutuhan_pendidikan_khusus.classes_, index=0)
    data["Kebutuhan_pendidikan_khusus"] = [Kebutuhan_pendidikan_khusus]
    with col3:
        Jenis_kelamin = st.selectbox(label='Jenis kelamin', options=encoder_Jenis_kelamin.classes_, index=0)
    data["Jenis_kelamin"] = [Jenis_kelamin]
    with col4:
        Penerima_beasiswa = st.selectbox(label='Penerima beasiswa', options=encoder_Penerima_beasiswa.classes_, index=0)
    data["Penerima_beasiswa"] = [Penerima_beasiswa]

st.subheader("📈 Data Akademik Semester 1")
with st.expander("Semester 1"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Usia_saat_daftar = float(st.number_input(label='Usia Daftar', value=18))
    data["Usia_saat_daftar"] = Usia_saat_daftar
    with col2:
        Unit_kurikuler_semester_1_dikreditkan = float(st.number_input(label='Unit K.Smt1 dikreditkan', value=0))
    data["Unit_kurikuler_semester_1_dikreditkan"] = Unit_kurikuler_semester_1_dikreditkan
    with col3:
        Unit_kurikuler_semester_1_daftar = float(st.number_input(label='Unit K.Smt1 daftar', value=0))
    data["Unit_kurikuler_semester_1_daftar"] = Unit_kurikuler_semester_1_daftar

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Evaluasi_unit_kurikuler_semester_1 = float(st.number_input(label='Evaluasi K.Smt1', value=0))
    data["Evaluasi_unit_kurikuler_semester_1"] = Evaluasi_unit_kurikuler_semester_1
    with col2:
        Unit_kurikuler_semester_1_disetujui = float(st.number_input(label='Unit K.Smt1 disetujui', value=0))
    data["Unit_kurikuler_semester_1_disetujui"] = Unit_kurikuler_semester_1_disetujui
    with col3:
        Nilai_unit_kurikuler_semester_1 = float(st.number_input(label='Nilai unit K.Smt1', value=0))
    data["Nilai_unit_kurikuler_semester_1"] = Nilai_unit_kurikuler_semester_1
    with col4:
        Unit_kurikuler_semester_1_tanpa_evaluasi = float(st.number_input(label='Unit K.Smt1 tanpa eval', value=0))
    data["Unit_kurikuler_semester_1_tanpa_evaluasi"] = Unit_kurikuler_semester_1_tanpa_evaluasi

st.subheader("📈 Data Akademik Semester 2")
with st.expander("Semester 2"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Unit_kurikuler_semester_2_dikreditkan = float(st.number_input(label='Unit K.Smt2 dikreditkan', value=0))
    data["Unit_kurikuler_semester_2_dikreditkan"] = Unit_kurikuler_semester_2_dikreditkan
    with col2:
        Unit_kurikuler_semester_2_daftar = float(st.number_input(label='Unit K.Smt2 daftar', value=0))
    data["Unit_kurikuler_semester_2_daftar"] = Unit_kurikuler_semester_2_daftar
    with col3:
        Evaluasi_unit_kurikuler_semester_2 = float(st.number_input(label='Evaluasi Unit K.Smt2', value=0))
    data["Evaluasi_unit_kurikuler_semester_2"] = Evaluasi_unit_kurikuler_semester_2

    col1, col2 = st.columns(2)
    with col1:
        Nilai_unit_kurikuler_semester_2 = float(st.number_input(label='Nilai Unit K.Smt2', value=0))
    data["Nilai_unit_kurikuler_semester_2"] = Nilai_unit_kurikuler_semester_2
    with col2:
        Unit_kurikuler_semester_2_tanpa_evaluasi = float(st.number_input(label='Unit K.Smt1 tanpa Eval', value=0))
    data["Unit_kurikuler_semester_2_tanpa_evaluasi"] = Unit_kurikuler_semester_2_tanpa_evaluasi


with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Prediction: {}".format(prediction(new_data)))
