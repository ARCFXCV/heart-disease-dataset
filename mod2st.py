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
    age: int = Field(..., ge=1, le=120)  # Yoshni 1 dan boshlash
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
    logger.error(f"Ma'lumotlar noto'g'ri: {e}")
    st.error(f"Ma'lumotlar noto'g'ri: {e}")
    st.stop()

# 10. Modelni yuklash va xeshni tekshirish
def verify_model(file_path, expected_hash):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash == expected_hash

if not verify_model(MODEL_PATH, MODEL_HASH):
    logger.error("Model fayli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.error("Model fayli buzilgan yoki ruxsatsiz o'zgartirilgan.")
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# 11. Bashorat qilish
if st.button("Bashorat qilish"):
    features = np.array([[
        user_input.age, user_input.sex, user_input.cp, user_input.trestbps, user_input.chol,
        user_input.fbs, user_input.restecg, user_input.thalach, user_input.exang,
        user_input.oldpeak, user_input.slope, user_input.ca, user_input.thal
    ]])

    # Standartlashtirish
    features = scaler.transform(features)

    # Bashorat
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("Bashorat: Sizda yurak kasalligi mavjud.")
    else:
        st.success("Bashorat: Yurak kasalligi aniqlanmadi.")

# 12. Modelni baholash
y_pred = rf.predict(X_test)
def evaluation(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    st.write(f"Model Accuracy: {accuracy:.2f}%")
    st.write(f"Classification Report:\n {metrics.classification_report(y_test, y_pred)}")
    cm = metrics.confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

evaluation(y_test, y_pred)
