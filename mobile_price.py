import pickle
import streamlit as st

model = pickle.load(open('estimasi_Mobile_Price_rediction.sav', 'rb'))

st.title('Mobile Price Prediksi')

Sale = st.number_input('Input Penjualan')
weight = st.number_input('Input Bobot')
resoloution = st.number_input('Input Resolusi')
ppi = st.number_input('Input ppi')
cpu_core = st.number_input('Input Inti CPU')
cpu_freq = st.number_input('Input Frequensi CPU')
internal_mem = st.number_input('Input Mem Internal')
ram = st.number_input('Input RAM')
RearCam = st.number_input('Input Kamera Belakang')
Front_Cam = st.number_input('Input Kamera Depan')
Battery = st.number_input('Input Baterai')
thickness = st.number_input('Input Ketebalan')


predict = ''

if st.button('Estimasi Mobile'):
    predict = model.predict(
        [[Sale, weight, resoloution, ppi, cpu_core, cpu_freq, internal_mem,
            ram, RearCam, Front_Cam, Battery, thickness]]
    )
    st.write('Estimasi : ', predict)
