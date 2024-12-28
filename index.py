import streamlit as st
import pickle
import numpy as np
import hashlib
from pydantic import BaseModel, ValidationError, Field
import logging
from sklearn.preprocessing import StandardScaler

# Logging sozlash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelni yuklash
MODEL_PATH_HEART = 'RandomForest_heart.pkl'  # Yurak kasalligi modelining yo'li
MODEL_PATH_NETWORK = 'RandomForest_network.pkl'  # Tarmoq hujumlari modeli

# Model faylining xeshini yaratish
def get_model_hash(file_path):
    return hashlib.sha256(open(file_path, 'rb').read()).hexdigest()

MODEL_HASH_HEART = get_model_hash(MODEL_PATH_HEART)
MODEL_HASH_NETWORK = get_model_hash(MODEL_PATH_NETWORK)

# Modelni tekshirish
def verify_model(file_path, expected_hash):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash == expected_hash

if not verify_model(MODEL_PATH_HEART, MODEL_HASH_HEART):
    logger.error("Yurak kasalligi modeli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.error("Yurak kasalligi modeli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.stop()

if not verify_model(MODEL_PATH_NETWORK, MODEL_HASH_NETWORK):
    logger.error("Tarmoq hujumlari modeli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.error("Tarmoq hujumlari modeli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.stop()

with open(MODEL_PATH_HEART, 'rb') as f:
    model_heart = pickle.load(f)

with open(MODEL_PATH_NETWORK, 'rb') as f:
    model_network = pickle.load(f)

# Streamlit interfeysi
st.set_page_config(page_title="Yurak Kasalligi va Tarmoq Hujumlari Aniqlash", layout="centered")

# Sarlavha va kirish matni
st.markdown(
"""
<div style="background-color: #f63366; padding: 10px; border-radius: 10px;">
<h1 style="color: white; text-align: center;">Yurak Kasalligi va Tarmoq Hujumlarini Aniqlash Tizimi</h1>
</div>
""",
unsafe_allow_html=True,
)
st.write("Iltimos, quyidagi ma'lumotlarni kiriting va tizim hujum yoki kasallik bor-yo'qligini aniqlaydi.")

# Tizimdan foydalanish uchun turli xil opsiyalar
model_option = st.selectbox("Modelni tanlang", ["Yurak Kasalligi", "Tarmoq Hujumlari"])

# Yurak kasalligi uchun kirish ma'lumotlari
if model_option == "Yurak Kasalligi":
    # Kirish parametrlarini olish
    class InputHeart(BaseModel):
        age: int = Field(..., ge=1, le=120)  
        sex: int = Field(..., ge=0, le=1)
        cp: int = Field(..., ge=0, le=3)
        trestbps: int = Field(..., ge=80, le=200)
        chol: int = Field(..., ge=100, le=400)
        fbs: int = Field(..., ge=0, le=1)
        restecg: int = Field(..., ge=0, le=2)
        thalach: int = Field(..., ge=50, le=200)
        exang: int = Field(..., ge=0, le=1)
        oldpeak: float = Field(..., ge=0.0, le=6.0)
        slope: int = Field(..., ge=0, le=2)
        ca: int = Field(..., ge=0, le=3)
        thal: int = Field(..., ge=3, le=7)

    # Foydalanuvchi ma'lumotlarini olish
    try:
        age = st.number_input("Yosh", min_value=1, max_value=120, value=30)  
        sex = st.selectbox("Jins", options=["Erkak", "Ayol"])
        cp = st.selectbox("Ko'krak og'rig'i turi", options=[0, 1, 2, 3])
        trestbps = st.number_input("Dam olishda qon bosimi", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum xolesterin miqdori", min_value=100, max_value=400, value=200)
        fbs = st.selectbox("Qon shakar darajasi 120 dan yuqori?", options=[0, 1])
        restecg = st.selectbox("Dam olishdagi elektrokardiogram", options=[0, 1, 2])
        thalach = st.number_input("Maksimal yurak tezligi", min_value=50, max_value=200, value=150)
        exang = st.selectbox("Yurak og'rig'i bo'ldimi?", options=[0, 1])
        oldpeak = st.number_input("Oldingi qiyinchilik", min_value=0.0, max_value=6.0, value=1.0)
        slope = st.selectbox("Sloy turi", options=[0, 1, 2])
        ca = st.selectbox("Qon tomirlarini soni", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia turi", options=[3, 6, 7])

        sex_encoded = 0 if sex == "Erkak" else 1

        user_input_heart = InputHeart(
            age=age,
            sex=sex_encoded,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal,
        )
    except ValidationError as e:
        logger.error(f"Ma'lumotlar noto'g'ri: {e}")
        st.error(f"Ma'lumotlar noto'g'ri: {e}")
        st.stop()

    # Modelni bashorat qilish
    if st.button("Bashorat qilish"):
        features_heart = np.array([[
            user_input_heart.age, user_input_heart.sex, user_input_heart.cp, user_input_heart.trestbps, user_input_heart.chol,
            user_input_heart.fbs, user_input_heart.restecg, user_input_heart.thalach, user_input_heart.exang,
            user_input_heart.oldpeak, user_input_heart.slope, user_input_heart.ca, user_input_heart.thal
        ]])

        # Standartlashtirish
        scaler = StandardScaler()
        features_heart = scaler.fit_transform(features_heart)

        # Bashorat qilish
        prediction = model_heart.predict(features_heart)
        if prediction[0] == 1:
            st.success("Bashorat: Sizda yurak kasalligi mavjud.")
        else:
            st.success("Bashorat: Yurak kasalligi aniqlanmadi.")

# Tarmoq hujumlarini aniqlash
elif model_option == "Tarmoq Hujumlari":
    # Kirish parametrlarini olish
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration", min_value=0)
        src_bytes = st.number_input("Source Bytes", min_value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0)
    with col2:
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0)
        hot = st.number_input("Hot", min_value=0)

    # Qo'shimcha ustunlarni standart qiymatlar bilan to'ldirish
    default_values = [0] * (118 - 5)
    features_network = [duration, src_bytes, dst_bytes, wrong_fragment, hot] + default_values

    # Bashorat tugmasi
    if st.button("Bashorat qilish"):
        features_network = np.array(features_network).reshape(1, -1)
        prediction = model_network.predict(features_network)[0]
        if prediction == 0:
            st.success("✅ Natija: Normal faoliyat")
        else:
            st.error("🚨 Natija: Hujum aniqlangan!")
