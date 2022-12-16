import tensorflow as tf
import json
import pickle
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import pandas as pd

# Импорт предварительно натренированных моделей
model = tf.keras.models.load_model(r'C:\Users\mufti\Desktop\LabRabML_2 (1)\LabRabML\lab_1\Neural_Network.h5')

with open(r'C:\Users\mufti\Desktop\LabRabML_2 (1)\LabRabML\lab_1\Linear_Regression_model.bin', mode='rb') as f:
    model_lin = pickle.load(f)

with open(r'C:\Users\mufti\Desktop\LabRabML_2 (1)\LabRabML\lab_1\Linear_Regression_model.json', mode='r') as f:
    m1dict = json.load(f)

with open(r'C:\Users\mufti\Desktop\LabRabML_2 (1)\LabRabML\lab_1\Neural_Network.json', mode='r') as f:
    m2dict = json.load(f)

with open(r'C:\Users\mufti\Desktop\LabRabML_2 (1)\LabRabML\lab_1\Norm_scale.bin', mode = 'rb') as f:
    ScalerNormX, ScalerNormY = pickle.load(f)

st.sidebar.header("Характеристика алмаза")
carat = st.sidebar.slider('Караты', min_value=0.2, max_value=5.01, value=2.0, step=0.001)

x = st.sidebar.slider('Размер по x', min_value=0.0, max_value=10.74, value=5.0, step=0.001)

y = st.sidebar.slider('Размер по y', min_value=0.0, max_value=58.9, value=14.0, step=0.001)

z = st.sidebar.slider('Размер по z', min_value=0.0, max_value=31.8, value=20.0, step=0.001)

clarity = st.sidebar.slider('Чистота', min_value=1, max_value=7, value=3, step=1)

#Помещаем данные в таблицу
dfX = pd.DataFrame(data=[[carat, x, y, z, clarity]], columns=m1dict["features"])

st.write('<style>div.block-container{padding:0rem; margin: 0rem; justify-content: space-between; }</style>', unsafe_allow_html=True)

st.title("исходные X")
st.write(dfX)

# Вывод нормализованой таблицы
st.title("нормализованные X")
Norm_dfX = pd.DataFrame(data=ScalerNormX.transform(dfX), columns=m2dict["features"])
st.write(Norm_dfX)


# Вычисление целевого значения по входным данным и вывод
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.title("Линейная регрессия")
        y_pred = model_lin.predict(dfX)
        st.write(f"R2={m1dict['R2']:0.6f}")
        st.write(f"RMSE={m1dict['RMSE']:0.6f}")
        st.write("исходный Y")
        y_pred = pd.DataFrame(data=y_pred, columns=m1dict["target"])
        st.write(y_pred)

    with col2:
        st.title("Нейронная сеть")
        with tf.device('/CPU:0'):
            yNorm_pred = model.predict(Norm_dfX)
        st.write(f"R2={m2dict['R2']:0.6f}")
        st.write(f"RMSE={m2dict['RMSE']:0.6f}")
        st.write("нормализованый Y")
        yNorm_pred = pd.DataFrame(data=yNorm_pred, columns=m2dict["target"])
        st.write(yNorm_pred)
        st.write("исходный Y")
        yNorm_pred = pd.DataFrame(data=ScalerNormY.inverse_transform(yNorm_pred), columns=m2dict["target"])
        st.write(yNorm_pred)


