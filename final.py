import streamlit as st
from streamlit_option_menu import option_menu
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

diabetes_model = pickle.load(open(f'D:\BTL PYTHON\diabetes_final_2.pkl','rb'))
heart_model = pickle.load(open(f'D:\BTL PYTHON\heart.pkl', 'rb'))

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

# Tạo menu trong sidebar với `option_menu`
with st.sidebar:
    selected_page = option_menu(
        menu_title="Main Menu",  # Tên menu
        options=["Home", "Prediction", "Advice"],  # Các lựa chọn trong menu
        icons=["house", "activity", "lightbulb"],  # Icon cho từng mục
        menu_icon="cast",  # Icon của menu chính
        default_index=0,  # Mục mặc định
    )

# Trang Home
if selected_page == "Home":
    st.title("Welcome to the Health Prediction App")
    #st.subheader("About This Application")
    st.write("""
        # About This Application
        This application is designed to assist users in predicting the risk of certain diseases based on their health data or medical images. It leverages advanced **Machine Learning** and **Deep Learning** techniques to provide accurate predictions and personalized health advice.

        ---

        ## Diseases Covered
        1. **Diabetes**  
        Prediction based on health indicators such as glucose level, blood pressure, BMI, and other factors.  

        2. **Pneumonia**  
        Detection using chest X-ray images analyzed by a deep learning model.  

        3. **Heart Disease**  
        Prediction based on health metrics like cholesterol level, resting heart rate, and blood pressure.  

        This application aims to provide users with actionable insights into their health status and assist in early detection of potential health risks.

        ---

        ## Features and Functions

        ### 1. Disease Prediction
        - **Diabetes Prediction**:  
        Enter your health metrics, such as glucose level, blood pressure, and BMI, to predict the likelihood of diabetes.  

        - **Pneumonia Detection**:  
        Upload a chest X-ray image, and the deep learning model will analyze it to determine if there are signs of pneumonia.  

        - **Heart Disease Prediction**:  
        Provide health indicators, such as cholesterol levels and resting ECG data, to predict the risk of heart disease.  

        ### 2. Personalized Health Advice
        After the prediction, the application offers tailored health advice based on your input data to help you manage your health better.

        ### 3. Interactive Visualization
        View charts and graphs of your health metrics and predictions to better understand your health trends.

        ---

        ## How to Use the Application

        ### 1. Select a Page  
        Use the menu on the left sidebar to navigate between the pages: **Home**, **Prediction**, and **Advice**.

        ### 2. Prediction  
        Choose a prediction model from the "Prediction" page:
        - **Diabetes Prediction**: Input your health data and click "Predict" to get the result.
        - **Pneumonia Detection**: Upload a chest X-ray image and click "Predict" to see the diagnosis.
        - **Heart Disease Prediction**: Input your health metrics and click "Predict" for the result.  

        View the prediction result and interpret the health risk.

        ### 3. Advice  
        Visit the "Advice" page for personalized health recommendations based on your input data.

        ---
    """)


# Trang Prediction
elif selected_page == "Prediction":
    st.title("Health Prediction")
    st.write("Select a prediction model and enter your data to receive a health prediction.")

    # Tạo menu con để chọn mô hình dự đoán trong trang Prediction
    model_choice = st.selectbox("Choose a Prediction Model", ["Diabetes Prediction", "Pneumonia Detection", "Heart Disease Prediction"])

    # Phần dự đoán bệnh tiểu đường
    if model_choice == "Diabetes Prediction":
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
                diabetes_result = "The person has diabetic"
            else:
                ok = 0
                diabetes_result = "The person has no diabetic"
        st.success(diabetes_result)

    elif model_choice == "Heart Disease Prediction":
        st.title("Heart disease Prediction Using Machine Learning")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input("Age")
        with col2:
            sex = st.text_input("Sex")
        with col3:
            cp = st.text_input("Chest Pain types")
        with col1:
            trestbps = st.text_input("Resting blood Pressure")
        with col2:
            chol = st.text_input("Serum Cholestroal in mg/dl")
        with col3:
            fbs = st.text_input("Fasting blood sugar > 120 mg/dl")
        with col1:
            restecg = st.text_input("Resting Electrocardiographic results")
        with col2:
            thalach = st.text_input("Maximum heart rate achieved")
        with col3:
            exang = st.text_input("Exercise Induced Angina")
        with col1:
            oldpeak = st.text_input("St depression induced by excercise")
        with col2:
            slope = st.text_input("Slope of the peak exercise ST segment")
        with col3:
            ca = st.text_input("Major vessels colored by flourosopy")
        with col1:
            thal = st.text_input("thal: 0 = normal; 1 fixed defect; 2 = reversable defect")
        heart_disease_result = ""
        
        if st.button("Heart Disease Test Result"):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            prediction = heart_model.predict([user_input])
            if prediction[0] == 1:
                heart_disease_result = "This person is having heart disease"
            else:
                heart_disease_result = "This person does not have heart disease"
        st.success(heart_disease_result)
        

    # Phần dự đoán viêm phổi
    elif model_choice == "Pneumonia Detection":
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
        

# Trang Advice
elif selected_page == "Advice":
    st.title("Personalized Health Advice")
    st.write("Get personalized advice based on your health indicators.")
    # Mã gợi ý lời khuyên sức khỏe theo chỉ số đầu vào
