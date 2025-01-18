import warnings
from PIL import Image, ImageEnhance
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import streamlit as st
import pyttsx3  # Thay thế Dispatch để tạo giọng nói trên Streamlit

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Tải mô hình dự đoán
model = load_model('D:/BTL PYTHON/pneumonia/Chest_x_ray_Detection/chest_xray.h5')

# Hàm phát âm
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Cấu hình giao diện Streamlit
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
