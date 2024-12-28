import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import streamlit as st

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
with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(rf, f)

# 7. Streamlit interfeysi yaratish
st.title("Yurak Kasalligi Bashorati")

# 8. Foydalanuvchi kiritadigan qiymatlarni olish
age = st.number_input("Yosh", min_value=0, max_value=120, value=30)
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

# 9. Jinsni raqamli ko‘rinishga o‘tkazish
sex_encoded = 0 if sex == "Erkak" else 1

# 10. Foydalanuvchi kiritgan qiymatlarni tekshirish
error_message = ""

# Yoshni tekshirish
if age < 1 or age > 120:
    error_message += "Yosh qiymati noto'g'ri. Iltimos, 1 va 120 orasida qiymat kiriting.\n"

# Qon bosimini tekshirish
if trestbps < 80 or trestbps > 200:
    error_message += "Dam olishdagi qon bosimi noto'g'ri. Iltimos, 80 va 200 orasida qiymat kiriting.\n"

# Xolesterin miqdorini tekshirish
if chol < 100 or chol > 400:
    error_message += "Serum xolesterin miqdori noto'g'ri. Iltimos, 100 va 400 orasida qiymat kiriting.\n"

# Oldpeak (qiyinchilik darajasi)ni tekshirish
if oldpeak < 0.0 or oldpeak > 6.0:
    error_message += "Oldpeak qiymati noto'g'ri. Iltimos, 0.0 va 6.0 orasida qiymat kiriting.\n"

# Agar xatoliklar mavjud bo'lsa, xatolik xabarini chiqarish va bashoratni to'xtatish
if error_message:
    st.error(error_message)
else:
    # 11. Agar barcha qiymatlar to'g'ri bo'lsa, bashorat qilish
    if st.button("Bashorat qilish"):
        features = np.array([[age, sex_encoded, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Standartlashtirish
        features = scaler.transform(features)

        # Modelni yuklash va bashorat qilish
        try:
            with open('RandomForest.pkl', 'rb') as file:
                model = pickle.load(file)

            prediction = model.predict(features)
            if prediction[0] == 1:
                st.success("Bashorat: Sizda yurak kasalligi mavjud.")
            else:
                st.success("Bashorat: Yurak kasalligi aniqlanmadi.")
        except Exception as e:
            st.error(f"Xato yuz berdi: {e}")

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
