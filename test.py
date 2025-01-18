import streamlit as st 
import pickle 
import os
from streamlit_option_menu import option_menu

# Xác định thư mục làm việc hiện tại
working_dir = os.path.dirname(os.path.abspath(__file__))

# Khởi tạo các biến
NewBMI_Overweight = 0
NewBMI_Underweight = 0
NewBMI_Obesity_1 = 0
NewBMI_Obesity_2 = 0 
NewBMI_Obesity_3 = 0
NewInsulinScore_Normal = 0 
NewGlucose_Low = 0
NewGlucose_Normal = 0 
NewGlucose_Overweight = 0
NewGlucose_Secret = 0

# Đường dẫn tới file chứa mô hình
file_path = os.path.join(working_dir, 'diabetes_DT.pkl')

# Tải mô hình từ file
with open(file_path, 'rb') as file:
    diabetes_model = pickle.load(file)

# Kiểm tra loại của mô hình đã tải
print(type(diabetes_model))

# Dữ liệu đầu vào
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age = [1, 85, 66, 29, 0, 26.6, 0.351, 31]

# Xử lý các biến BMI
if float(BMI) <= 18.5:
    NewBMI_Underweight = 1
elif 18.5 < float(BMI) <= 24.9:
    pass
elif 24.9 < float(BMI) <= 29.9:
    NewBMI_Overweight = 1
elif 29.9 < float(BMI) <= 34.9:
    NewBMI_Obesity_1 = 1
elif 34.9 < float(BMI) <= 39.9:
    NewBMI_Obesity_2 = 1
elif float(BMI) > 39.9:
    NewBMI_Obesity_3 = 1

# Xử lý các biến Insulin
if 16 <= float(Insulin) <= 166:
    NewInsulinScore_Normal = 1

# Xử lý các biến Glucose
if float(Glucose) <= 70:
    NewGlucose_Low = 1
elif 70 < float(Glucose) <= 99:
    NewGlucose_Normal = 1
elif 99 < float(Glucose) <= 126:
    NewGlucose_Overweight = 1
elif float(Glucose) > 126:
    NewGlucose_Secret = 1

# Dữ liệu đầu vào cho mô hình
user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
              BMI, DiabetesPedigreeFunction, Age, NewBMI_Underweight,
              NewBMI_Overweight, NewBMI_Obesity_1,
              NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal, 
              NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
              NewGlucose_Secret]

# Chuyển đổi dữ liệu đầu vào sang kiểu float
user_input = [float(x) for x in user_input]

# Dự đoán với mô hình đã tải
prediction = diabetes_model.predict([user_input])
print(prediction[0])
