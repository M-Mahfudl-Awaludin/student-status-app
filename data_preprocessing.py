import joblib
import numpy as np
import pandas as pd

encoder_Mode_pendaftaran = joblib.load("model/encoder_Mode_pendaftaran.joblib")
encoder_Urutan_pendaftaran = joblib.load("model/encoder_Urutan_pendaftaran.joblib")
encoder_Program_studi = joblib.load("model/encoder_Program_studi.joblib")
encoder_Waktu_kuliah = joblib.load("model/encoder_Waktu_kuliah.joblib")
encoder_Kualifikasi_sebelumnya = joblib.load("model/encoder_Kualifikasi_sebelumnya.joblib")
encoder_Kebutuhan_pendidikan_khusus = joblib.load("model/encoder_Kebutuhan_pendidikan_khusus.joblib")
encoder_Jenis_kelamin = joblib.load("model/encoder_Jenis_kelamin.joblib")
encoder_Penerima_beasiswa = joblib.load("model/encoder_Penerima_beasiswa.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_Nilai_kualifikasi_sebelumnya = joblib.load("model/scaler_Nilai_kualifikasi_sebelumnya.joblib")
scaler_Nilai_penerimaan = joblib.load("model/scaler_Nilai_penerimaan.joblib")
scaler_Usia_saat_daftar = joblib.load("model/scaler_Usia_saat_daftar.joblib")
scaler_Unit_kurikuler_semester_1_dikreditkan = joblib.load("model/scaler_Unit_kurikuler_semester_1_dikreditkan.joblib")
scaler_Unit_kurikuler_semester_1_daftar  = joblib.load("model/scaler_Unit_kurikuler_semester_1_daftar.joblib")
scaler_Evaluasi_unit_kurikuler_semester_1 = joblib.load("model/scaler_Evaluasi_unit_kurikuler_semester_1.joblib")
scaler_Unit_kurikuler_semester_1_disetujui = joblib.load("model/scaler_Unit_kurikuler_semester_1_disetujui.joblib")
scaler_Nilai_unit_kurikuler_semester_1 = joblib.load("model/scaler_Nilai_unit_kurikuler_semester_1.joblib")
scaler_Unit_kurikuler_semester_1_tanpa_evaluasi = joblib.load("model/scaler_Unit_kurikuler_semester_1_tanpa_evaluasi.joblib")
scaler_Unit_kurikuler_semester_2_dikreditkan = joblib.load("model/scaler_Unit_kurikuler_semester_2_dikreditkan.joblib")
scaler_Unit_kurikuler_semester_2_daftar = joblib.load("model/scaler_Unit_kurikuler_semester_2_daftar.joblib")
scaler_Evaluasi_unit_kurikuler_semester_2 = joblib.load("model/scaler_Evaluasi_unit_kurikuler_semester_2.joblib")
scaler_Unit_kurikuler_semester_2_disetujui = joblib.load("model/scaler_Unit_kurikuler_semester_2_disetujui.joblib")
scaler_Nilai_unit_kurikuler_semester_2  = joblib.load("model/scaler_Nilai_unit_kurikuler_semester_2.joblib")
scaler_Unit_kurikuler_semester_2_tanpa_evaluasi  = joblib.load("model/scaler_Unit_kurikuler_semester_2_tanpa_evaluasi.joblib")

pca_numerical_columns_1 = [
    'Nilai_kualifikasi_sebelumnya',
    'Nilai_penerimaan',
    'Usia_saat_daftar',
     
]
 
pca_numerical_columns_2 = [
    'Unit_kurikuler_semester_1_dikreditkan',
    'Unit_kurikuler_semester_1_daftar',
    'Evaluasi_unit_kurikuler_semester_1',
    'Unit_kurikuler_semester_1_disetujui',
    'Nilai_unit_kurikuler_semester_1',
    'Unit_kurikuler_semester_1_tanpa_evaluasi',
    'Unit_kurikuler_semester_2_dikreditkan',
    'Unit_kurikuler_semester_2_daftar',
    'Evaluasi_unit_kurikuler_semester_2',
    'Unit_kurikuler_semester_2_disetujui',
    'Nilai_unit_kurikuler_semester_2',
    'Unit_kurikuler_semester_2_tanpa_evaluasi',
]
 
def data_preprocessing(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contains all the data to make prediction 

    Returns:
        Pandas DataFrame: Dataframe that contains all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    # Encoding categorical columns using pre-trained encoders
    df["Mode_pendaftaran"] = encoder_Mode_pendaftaran.transform(data["Mode_pendaftaran"])
    df["Urutan_pendaftaran"] = encoder_Urutan_pendaftaran.transform(data["Urutan_pendaftaran"])
    df["Program_studi"] = encoder_Program_studi.transform(data["Program_studi"])
    df["Waktu_kuliah"] = encoder_Waktu_kuliah.transform(data["Waktu_kuliah"])
    df["Kualifikasi_sebelumnya"] = encoder_Kualifikasi_sebelumnya.transform(data["Kualifikasi_sebelumnya"])
    df["Kebutuhan_pendidikan_khusus"] = encoder_Kebutuhan_pendidikan_khusus.transform(data["Kebutuhan_pendidikan_khusus"])
    df["Jenis_kelamin"] = encoder_Jenis_kelamin.transform(data["Jenis_kelamin"])
    df["Penerima_beasiswa"] = encoder_Penerima_beasiswa.transform(data["Penerima_beasiswa"])

    # Scaling numerical columns for PCA 1
    data["Nilai_kualifikasi_sebelumnya"] = scaler_Nilai_kualifikasi_sebelumnya.transform(np.asarray(data["Nilai_kualifikasi_sebelumnya"]).reshape(-1,1))[0]
    data["Nilai_penerimaan"] = scaler_Nilai_penerimaan.transform(np.asarray(data["Nilai_penerimaan"]).reshape(-1,1))[0]
    data["Usia_saat_daftar"] = scaler_Usia_saat_daftar.transform(np.asarray(data["Usia_saat_daftar"]).reshape(-1,1))[0]
    
    # Apply PCA 1
    df[["pc1_1", "pc1_2", "pc1_3"]] = pca_1.transform(data[pca_numerical_columns_1])
    
    # Scaling numerical columns for PCA 1
    data["Nilai_kualifikasi_sebelumnya"] = scaler_Nilai_kualifikasi_sebelumnya.transform(data[["Nilai_kualifikasi_sebelumnya"]])
    data["Nilai_penerimaan"] = scaler_Nilai_penerimaan.transform(data[["Nilai_penerimaan"]])
    data["Usia_saat_daftar"] = scaler_Usia_saat_daftar.transform(data[["Usia_saat_daftar"]])
    
    # Scaling numerical columns for PCA 2
    data["Unit_kurikuler_semester_1_dikreditkan"] = scaler_Unit_kurikuler_semester_1_dikreditkan.transform(data[["Unit_kurikuler_semester_1_dikreditkan"]])
    data["Unit_kurikuler_semester_1_daftar"] = scaler_Unit_kurikuler_semester_1_daftar.transform(data[["Unit_kurikuler_semester_1_daftar"]])
    data["Evaluasi_unit_kurikuler_semester_1"] = scaler_Evaluasi_unit_kurikuler_semester_1.transform(data[["Evaluasi_unit_kurikuler_semester_1"]])
    data["Unit_kurikuler_semester_1_disetujui"] = scaler_Unit_kurikuler_semester_1_disetujui.transform(data[["Unit_kurikuler_semester_1_disetujui"]])
    data["Nilai_unit_kurikuler_semester_1"] = scaler_Nilai_unit_kurikuler_semester_1.transform(data[["Nilai_unit_kurikuler_semester_1"]])
    data["Unit_kurikuler_semester_1_tanpa_evaluasi"] = scaler_Unit_kurikuler_semester_1_tanpa_evaluasi.transform(data[["Unit_kurikuler_semester_1_tanpa_evaluasi"]])
    
    data["Unit_kurikuler_semester_2_dikreditkan"] = scaler_Unit_kurikuler_semester_2_dikreditkan.transform(data[["Unit_kurikuler_semester_2_dikreditkan"]])
    data["Unit_kurikuler_semester_2_daftar"] = scaler_Unit_kurikuler_semester_2_daftar.transform(data[["Unit_kurikuler_semester_2_daftar"]])
    data["Evaluasi_unit_kurikuler_semester_2"] = scaler_Evaluasi_unit_kurikuler_semester_2.transform(data[["Evaluasi_unit_kurikuler_semester_2"]])
    data["Unit_kurikuler_semester_2_disetujui"] = scaler_Unit_kurikuler_semester_2_disetujui.transform(data[["Unit_kurikuler_semester_2_disetujui"]])
    data["Nilai_unit_kurikuler_semester_2"] = scaler_Nilai_unit_kurikuler_semester_2.transform(data[["Nilai_unit_kurikuler_semester_2"]])
    data["Unit_kurikuler_semester_2_tanpa_evaluasi"] = scaler_Unit_kurikuler_semester_2_tanpa_evaluasi.transform(data[["Unit_kurikuler_semester_2_tanpa_evaluasi"]])

    
    # Apply PCA 2
    df[["pc2_1", "pc2_2"]] = pca_2.transform(data[pca_numerical_columns_2])
    
    return df
