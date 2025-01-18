import streamlit as st 
import pickle 
import os
from streamlit_option_menu import option_menu
import sklearn 
import warnings

from PIL import Image, ImageEnhance
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import streamlit as st
import pyttsx3  # Thay thế Dispatch để tạo giọng nói trên Streamlit

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")

working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'D:\BTL PYTHON\diabetes.pkl','rb'))

model = load_model('D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/chest_xray.h5')

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


#heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl','rb'))
#kidney_disease_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl','rb'))
print(type(diabetes_model))
NewBMI_Overweight=0
NewBMI_Underweight=0
NewBMI_Obesity_1=0
NewBMI_Obesity_2=0 
NewBMI_Obesity_3=0
NewInsulinScore_Normal=0 
NewGlucose_Low=0
NewGlucose_Normal=0 
NewGlucose_Overweight=0
NewGlucose_Secret=0

with st.sidebar:
    selected = option_menu("Mulitple Disease Prediction", 
                ['Diabetes Prediction',
                 'PNEUMONIA'
                ],
                 menu_icon='hospital-fill',
                 icons=['activity','heart', 'person'],
                 default_index=0)

if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("BloodPressure Value")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value")
    with col2:
        Insulin = st.text_input("Insulin Value")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    with col2:
        Age = st.text_input("Age")
    diabetes_result = ""
    ok = -1
    if st.button("Diabetes Test Result"):
        if float(BMI)<=18.5:
            NewBMI_Underweight = 1
        elif 18.5 < float(BMI) <=24.9:
            pass
        elif 24.9<float(BMI)<=29.9:
            NewBMI_Overweight =1
        elif 29.9<float(BMI)<=34.9:
            NewBMI_Obesity_1 =1
        elif 34.9<float(BMI)<=39.9:
            NewBMI_Obesity_2=1
        elif float(BMI)>39.9:
            NewBMI_Obesity_3 = 1
        
        if 16<=float(Insulin)<=166:
            NewInsulinScore_Normal = 1

        if float(Glucose)<=70:
            NewGlucose_Low = 1
        elif 70<float(Glucose)<=99:
            NewGlucose_Normal = 1
        elif 99<float(Glucose)<=126:
            NewGlucose_Overweight = 1
        elif float(Glucose)>126:
            NewGlucose_Secret = 1

        user_input=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                    BMI,DiabetesPedigreeFunction,Age, NewBMI_Underweight,
                    NewBMI_Overweight,NewBMI_Obesity_1,
                    NewBMI_Obesity_2,NewBMI_Obesity_3,NewInsulinScore_Normal, 
                    NewGlucose_Low,NewGlucose_Normal, NewGlucose_Overweight,
                    NewGlucose_Secret]
        
        user_input = [float(x) for x in user_input]
        prediction = diabetes_model.predict([user_input])
        if prediction[0]==1:
            ok = 1
            diabetes_result = "The person has diabetic - co benh"
        else:
            ok = 0
            diabetes_result = "The person has no diabetic"
    st.success(diabetes_result)

    #them
    #with st.sidebar:
    st.markdown("### Lifestyle and Preventive Recommendations:")
    if ok==1:
        st.write("- **Diet**: Reduce sugar and carbohydrate intake. Eat more vegetables and fiber.")
        st.write("- **Exercise**: Engage in daily physical activities like walking or aerobic exercises.")
        st.write("- **Health Monitoring**: Regularly check blood sugar levels and maintain a healthy weight.")
    elif ok == 0:
        st.write("- Maintain a balanced diet and regular physical activity.")
        st.write("- Avoid excessive sugar intake and monitor your health regularly.")

if selected == 'PNEUMONIA':
    ok = -1
    st.title("PNEUMONIA Detection App")
    st.markdown("<h2 style='text-align: center; color: #035874;'>Chest X-ray PNEUMONIA Detection</h2>", unsafe_allow_html=True)

    # Thêm ảnh GIF (nếu muốn, Streamlit không hỗ trợ GIF động như PyQt, chỉ hiển thị ảnh tĩnh)
    #st.image("D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/picture.gif", caption="Chest X-ray GIF", use_column_width=True)
    st.image("D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/picture.gif", caption="Chest X-ray GIF", use_container_width=True)
    # Tải lên ảnh
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh được tải lên
        image_to_predict = Image.open(uploaded_file).convert('RGB')
        #st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)
        st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)
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
            ok = 0
        else:
            result_text = "Affected by PNEUMONIA"
            ok = 1

        st.write("Prediction:", result_text)
        speak(result_text)

    st.markdown("### Preventive Measures for Pneumonia:")
    if ok == 1:
        st.write("- **Keep Warm**: Avoid cold environments, especially in winter.")
        st.write("- **Boost Immunity**: Eat foods rich in vitamin C, and ensure adequate rest.")
        st.write("- **Avoid Lung Irritants**: Stay away from smoke and air pollutants.")
    elif ok == 0:
        st.write("- **Stay Active**: Engage in regular exercise to strengthen your respiratory system.")
        st.write("- **Avoid Smoking**: Keep your lungs healthy by avoiding smoking and secondhand smoke.")
