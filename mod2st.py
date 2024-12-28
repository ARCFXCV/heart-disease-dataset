import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import streamlit as st
from pydantic import BaseModel, ValidationError, Field
import hashlib
import logging

# Logging sozlash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Ma'lumotlarni yuklash
url = "https://raw.githubusercontent.com/ARCFXCV/heart-disease-dataset/refs/heads/main/heart.csv"
data = pd.read_csv(url)

# 2. X (kirish o'zgaruvchilari) va y (natija) ni aniqlash
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Maqsadli o'zgaruvchi (kasallik holati)

# 3. Ma'lumotlarni train va testga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Ma'lumotlarni standartlashtirish
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Random Forest modelini yaratish va o‘qitish
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 6. Modelni saqlash
MODEL_PATH = 'RandomForest.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(rf, f)

# Model faylining xeshini yaratish
MODEL_HASH = hashlib.sha256(open(MODEL_PATH, 'rb').read()).hexdigest()

# 7. Streamlit interfeysi yaratish
st.title("Yurak Kasalligi Bashorati")

# Xavfsizlik eslatmasi
st.warning("Iltimos, kiritilgan ma'lumotlarni ehtiyotkorlik bilan tekshirib, xatoliklar mavjudligini aniqlang. Dasturdagi xatoliklar xavfsizlikka ta'sir qilishi mumkin!")

# 8. Kiruvchi ma'lumotlar uchun validatsiya modeli
class InputData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Yoshni 1 dan boshlash")  # Yoshni 1 dan boshlash
    sex: int = Field(..., ge=0, le=1, description="Jinsni 0 yoki 1 sifatida kiriting: 0 - Erkak, 1 - Ayol")
    cp: int = Field(..., ge=0, le=3, description="Ko'krak og'rig'i turini 0-3 orasida kiriting")
    trestbps: int = Field(..., ge=80, le=200, description="Dam olishdagi qon bosimi (80-200 oralig'ida)")
    chol: int = Field(..., ge=100, le=400, description="Serum xolesterin miqdori (100-400 oralig'ida)")
    fbs: int = Field(..., ge=0, le=1, description="Qon shakar darajasini 0 yoki 1 sifatida kiriting: 0 - 120 dan past, 1 - 120 dan yuqori")
    restecg: int = Field(..., ge=0, le=2, description="Dam olishdagi elektrokardiogram: 0, 1, yoki 2")
    thalach: int = Field(..., ge=50, le=200, description="Maksimal yurak tezligini (50-200 oralig'ida) kiriting")
    exang: int = Field(..., ge=0, le=1, description="Yurak og'rig'i bo'ldimi? 0 - yo'q, 1 - bor")
    oldpeak: float = Field(..., ge=0.0, le=6.0, description="Oldingi qiyinchilikni 0.0 dan 6.0 gacha kiriting")
    slope: int = Field(..., ge=0, le=2, description="Sloy turini 0, 1 yoki 2 sifatida kiriting")
    ca: int = Field(..., ge=0, le=3, description="Qon tomirlarini sonini 0 dan 3 gacha kiriting")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia turini 3, 6 yoki 7 sifatida kiriting")

# 9. Foydalanuvchi kiritadigan qiymatlarni olish
try:
    age = st.number_input("Yosh", min_value=1, max_value=120, value=30)  # Yoshni manfiy kiritishni oldini olish
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

    user_input = InputData(
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
    logger
