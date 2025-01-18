'''
import streamlit as st 
import pickle 
import os
from streamlit_option_menu import option_menu
import numpy as np
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")

working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'D:/BTL PYTHON/best_diabetes_model_2.sav','rb'))

with st.sidebar: 
    selected = option_menu('Multipel disease prediction system',
                           ['Diabetes Prediction'], default_index=0)
    


if (selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
    with col2:
        Age = st.text_input('Age of the Person')

    
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        #user_input = [2, 88, 74, 19, 53, 29, 0.229, 22]
        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])
        

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'
    
    st.success(diab_diagnosis)

'''

import streamlit as st
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import pandas as pd
import warnings
from PIL import Image, ImageEnhance
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import pyttsx3  # Thay thế Dispatch để tạo giọng nói trên Streamlit

print(tf.__version__)
# Cấu hình trang Streamlit
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="")

# Tải model đã huấn luyện
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'best_diabetes_model_2.sav'), 'rb'))
#diabetes_model = pickle.load(open(f'D:\BTL PYTHON\diabetes.pkl','rb'))
# Tạo scaler từ dữ liệu huấn luyện ban đầu
scaler = StandardScaler()
diabetes_dataset = pd.read_csv("diabetes.csv")
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler.fit(X)
# model pneumonia

# Tải mô hình dự đoán
model = load_model('D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/chest_xray.h5')

# Hàm phát âm
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', ['Diabetes Prediction', 'PNEUMONIA'], default_index=0)

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        
        # Chuẩn hóa dữ liệu đầu vào
        input_data_reshaped = np.asarray(user_input).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_reshaped)

        diab_prediction = diabetes_model.predict(input_data_scaled)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'

        st.success(diab_diagnosis)

if selected == 'PNEUMONIA':
    st.title("PNEUMONIA Detection App")
    st.markdown("<h2 style='text-align: center; color: #035874;'>Chest X-ray PNEUMONIA Detection</h2>", unsafe_allow_html=True)

    # Thêm ảnh GIF (nếu muốn, Streamlit không hỗ trợ GIF động như PyQt, chỉ hiển thị ảnh tĩnh)
    st.image("D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/picture.gif", caption="Chest X-ray GIF", use_column_width=True)

    # Tải lên ảnh
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh được tải lên
        image_to_predict = Image.open(uploaded_file).convert('RGB')
        st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)

        # Chuẩn bị ảnh cho mô hình
        img_file = image_to_predict.resize((224, 224))
        x = image.img_to_array(img_file)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Dự đoán
        classes = model.predict(img_data)

        # Hiển thị kết quả
        if classes[0][0] > 0.5:
            result_text = "Result is Normal"
        else:
            result_text = "Affected by PNEUMONIA"

        st.write("Prediction:", result_text)
        speak(result_text)



