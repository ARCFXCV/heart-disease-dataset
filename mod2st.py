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
Age = st.number_input("Yosh: ", min_value=0, max_value=120)
Glukoza = st.number_input("Glukoza miqdori: ", format="%.1f", min_value=0.0)
BMI = st.number_input("BMI: ", format="%.1f", min_value=0.0)
Hypertension = st.selectbox("Gipertenziya mavjudmi?", options=[0, 1])
HeartDisease = st.selectbox("Yurak kasalligi mavjudmi?", options=[0, 1])
AfricanAmerican = st.selectbox("Irq: Afro-amerikalik", options=[0, 1])
Asian = st.selectbox("Irq: Osiyolik", options=[0, 1])
Caucasian = st.selectbox("Irq: Kavkazlik", options=[0, 1])
Hispanic = st.selectbox("Irq: Ispan tilida so'zlashuvchi", options=[0, 1])
Other = st.selectbox("Irq: Boshqa", options=[0, 1])

features = np.array([[Age, Glukoza, BMI, Hypertension, HeartDisease, 
                      AfricanAmerican, Asian, Caucasian, Hispanic, Other]])

# Bashorat qilish uchun tugma
# To'g'ri indentatsiya
if st.button("Bashorat qilish"):
    features = np.array([[Age, Glukoza, BMI, Hypertension, HeartDisease, 
                          AfricanAmerican, Asian, Caucasian, Hispanic, Other]])
    features = scaler.transform(features) 

    try:
        with open('RandomForest.pkl', 'rb') as file:
            decision_tree_model = pickle.load(file)
    except Exception as e:
        st.error(f"Modelni yuklashda xato: {e}")



    # Bashorat qilish
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("Bashorat: Sizda yurak kasalligi mavjud.")
    else:
        st.success("Bashorat: Yurak kasalligi aniqlanmadi.")
