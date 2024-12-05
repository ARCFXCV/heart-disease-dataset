import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import streamlit as st

# Ma'lumotlarni yuklash
url = "https://raw.githubusercontent.com/ARCFXCV/heart-disease-dataset/refs/heads/main/heart.csv"
data = pd.read_csv(url)

# X (kirish o'zgaruvchilari) va y (natija) ni aniqlash
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']  # Maqsadli o'zgaruvchi (kasallik holati)

# Datasetni 80% train va 20% test uchun ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ma'lumotlarni skalalash (standartlashtirish)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest modelini yaratish
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Modelni sinab ko'rish
y_pred = rf.predict(X_test)

# Modelning aniqligini baholash
def evaluation(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    st.write(f"Model Accuracy: {accuracy:.2f}%")
    st.write(f"Classification Report:\n {metrics.classification_report(y_test, y_pred)}")
    cm = metrics.confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

# Modelni baholash
evaluation(y_test, y_pred)

# Modelni faylga saqlash
with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Streamlit interfeysi
st.title("Yurak Kasalligi Bashorati")

# Kirish qiymatlarini olish
age = st.number_input("Yosh", min_value=0, max_value=120)
sex = st.selectbox("Jins", options=["Erkak", "Ayol"])  # 0: Erkak, 1: Ayol
cp = st.selectbox("Ko'krak og'rig'i turi", options=[0, 1, 2, 3])
trestbps = st.number_input("Dam olishda qon bosimi", min_value=80, max_value=200)
chol = st.number_input("Serum xolesterin miqdori", min_value=100, max_value=400)
fbs = st.selectbox("Qon shakar darajasi 120 dan yuqori?", options=[0, 1])
restecg = st.selectbox("Dam olishdagi elektrokardiogram", options=[0, 1, 2])
thalach = st.number_input("Maksimal yurak tezligi", min_value=50, max_value=200)
exang = st.selectbox("Yurak og'rig'i bo'ldimi?", options=[0, 1])
oldpeak = st.number_input("Oldingi qiyinchilik", min_value=0.0, max_value=6.0)
slope = st.selectbox("Sloy turi", options=[0, 1, 2])
ca = st.selectbox("Qon tomirlarini soni", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia turi", options=[3, 6, 7])

# Bashorat qilish uchun tugma
if st.button("Bashorat qilish"):
   features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
   features = np.array(features)  # to'liq ikki o'lchovli formatga kiritish
   features = scaler.transform(features)  # Transform qilish

    # Modelni yuklash
    try:
        with open('RandomForest.pkl', 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Modelni yuklashda xato: {e}")

    # Bashorat qilish
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("Bashorat: Sizda yurak kasalligi mavjud.")
    else:
        st.success("Bashorat: Yurak kasalligi aniqlanmadi.")
