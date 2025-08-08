import streamlit as st
import pandas as pd
import lightgbm as lgb

# ✅ 모델 불러오기 및 학습
@st.cache_resource
def load_model():
    data = pd.read_csv("cardio_train.csv", sep=';')
    data["age_years"] = (data["age"] / 365).astype(int)
    data["bmi"] = data["weight"] / (data["height"] / 100) ** 2
    data["hypertension"] = ((data["ap_hi"] >= 140) | (data["ap_lo"] >= 90)).astype(int)

    features = ["age_years", "gender", "ap_hi", "ap_lo", "bmi",
                "cholesterol", "gluc", "smoke", "alco", "active", "hypertension"]
    X = data[features]
    y = data["cardio"]

    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
    model = lgb.train(params, train_data, num_boost_round=30)
    return model, features

model, features = load_model()

# ✅ 사용자 입력 UI
st.set_page_config(page_title="심혈관질환 위험도 예측기")
st.title("🫀 심혈관질환 10년 위험도 예측기")
st.markdown("건강 정보를 입력하면 10년 내 심혈관질환 위험을 예측합니다.")

age = st.slider("나이", 20, 90, 50)
gender = st.radio("성별", ["남성", "여성"])
ap_hi = st.number_input("수축기 혈압 (ap_hi)", value=120)
ap_lo = st.number_input("이완기 혈압 (ap_lo)", value=80)
height_cm = st.number_input("키 (cm)", value=170)
weight_kg = st.number_input("몸무게 (kg)", value=65)
chol = st.radio("콜레스테롤 수치", ["정상", "경계", "높음"])
gluc = st.radio("혈당 수치", ["정상", "경계", "높음"])
smoke = st.checkbox("흡연")
alco = st.checkbox("음주")
active = st.checkbox("활동적 생활")

# ✅ 입력값 전처리
bmi = weight_kg / ((height_cm / 100) ** 2)
gender_num = 1 if gender == "남성" else 0
chol_map = {"정상": 1, "경계": 2, "높음": 3}
gluc_map = {"정상": 1, "경계": 2, "높음": 3}
hypertension = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0

input_data = {
    "age_years": age,
    "gender": gender_num,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "bmi": bmi,
    "cholesterol": chol_map[chol],
    "gluc": gluc_map[gluc],
    "smoke": int(smoke),
    "alco": int(alco),
    "active": int(active),
    "hypertension": hypertension
}

# ✅ 예측 버튼
if st.button("🔍 위험도 예측"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.subheader(f"📈 예측된 10년 내 심혈관질환 위험도: {round(prediction * 100, 2)}%")
    if prediction >= 0.15:
        st.warning("⚠️ 위험도가 높습니다. 정기적인 검사와 건강 관리가 필요합니다.")
    else:
        st.success("🎉 전반적으로 위험도가 낮습니다. 지금처럼 건강을 유지하세요!")

st.caption("🧠 Powered by Streamlit + LightGBM")
