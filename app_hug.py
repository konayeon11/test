# -*- coding: utf-8 -*-
# app_hug.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
import requests
import os
import json

# ✅ Hugging Face API 설정
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
API_KEY = st.secrets["hf"]["api_key"]  # Streamlit Secrets에 저장된 키 사용

if not API_KEY:
    st.error("❌ Hugging Face API 키가 설정되어 있지 않습니다. Streamlit Secrets에 'hf.api_key'를 등록해주세요.")
    st.stop()

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# ✅ GPT 기반 조언 함수 (Hugging Face 사용)
def generate_lifestyle_advice(risk_factors: dict):
    risk_factors_str = json.dumps(risk_factors, ensure_ascii=False)
    prompt = f"""
안녕하세요
"""

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list):
            return result[0]["generated_text"].strip()
        elif "generated_text" in result:
            return result["generated_text"].strip()
        else:
            return "⚠️ 조언을 생성하는 데 문제가 발생했습니다."
    else:
        return f"❌ API 호출 실패: {response.status_code} - {response.text}"

# ✅ 모델 로드 (데모용 LightGBM 모델 생성)
@st.cache_resource
def load_model():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=11)
    lgb_train = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=10)
    return model

model = load_model()

# 콜레스테롤 수치에 따른 범주형 분류 함수
def classify_cholesterol(chol_value):
    # 예시 기준 (mg/dL)
    if chol_value < 200:
        return 1  # 정상
    elif 200 <= chol_value < 240:
        return 2  # 경계
    else:
        return 3  # 높음

# 혈당 수치에 따른 범주형 분류 함수
def classify_gluc(gluc_value):
    # 예시 기준 (mg/dL, 공복 혈당 기준)
    if gluc_value < 100:
        return 1  # 정상
    elif 100 <= gluc_value < 126:
        return 2  # 경계
    else:
        return 3  # 높음

# ✅ 사용자 입력 UI
st.set_page_config(page_title="심혈관질환 예측기", layout="centered")
st.title("🫀 심혈관질환 10년 위험도 예측기")
st.markdown("건강 정보를 입력하면 10년 내 심혈관질환 위험을 예측하고, 필요 시 GPT 기반 **맞춤형 건강 조언**을 제공합니다.")

st.header("📋 건강 정보 입력")
age = st.slider("나이", 20, 90, 50)
gender = st.radio("성별", ["남성", "여성"])
ap_hi = st.number_input("수축기 혈압", value=120)
ap_lo = st.number_input("이완기 혈압", value=80)
height_cm = st.number_input("키(cm)", value=170)
weight_kg = st.number_input("몸무게(kg)", value=65)

# 변경: 콜레스테롤 수치를 수치형으로 받음
chol = st.number_input("콜레스테롤 수치 (mg/dL)", min_value=100, max_value=400, value=180)

# 변경: 혈당 수치를 수치형으로 받음
gluc = st.number_input("혈당 수치 (공복 mg/dL)", min_value=50, max_value=300, value=90)

smoke = st.checkbox("흡연")
alco = st.checkbox("음주")
active = st.checkbox("활동적 생활")

# ✅ 입력 처리
bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else 0
gender_num = 1 if gender == "남성" else 0
chol_cat = classify_cholesterol(chol)
gluc_cat = classify_gluc(gluc)
hypertension = int(ap_hi >= 140 or ap_lo >= 90)

user_input = {
    "age_years": age,
    "gender": gender_num,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "bmi": bmi,
    "cholesterol": chol_cat,  # 분류 결과 할당
    "gluc": gluc_cat,         # 분류 결과 할당
    "smoke": int(smoke),
    "alco": int(alco),
    "active": int(active),
    "hypertension": hypertension
}

# ✅ 위험 예측
def predict_risk(model, user_input: dict):
    features = ["age_years", "gender", "ap_hi", "ap_lo", "bmi",
                "cholesterol", "gluc", "smoke", "alco", "active", "hypertension"]
    df = pd.DataFrame([user_input])
    return model.predict(df)[0]

if st.button("🔍 위험도 예측"):
    risk = predict_risk(model, user_input)
    risk_percent = round(risk * 100, 2)
    st.subheader(f"📈 예측 결과: {risk_percent}%")

    risk_factors = {
        "고혈압": hypertension == 1,
        "흡연": smoke,
        "음주": alco,
        "비만": bmi >= 25,
        "고콜레스테롤": chol_cat >= 2,
        "고혈당": gluc_cat >= 2,
        "운동 부족": not active,
    }

    if risk_percent >= 15:
        st.warning("⚠️ 심혈관계 위험이 높은 편입니다. 생활습관 개선이 필요합니다.")
        st.markdown("💡 **GPT 기반 맞춤형 건강 처방**")
        with st.spinner("GPT가 건강 조언을 생성 중입니다..."):
            advice = generate_lifestyle_advice(risk_factors)
            st.success("생활 처방 도착 ✅")
            st.markdown(advice)
    else:
        st.success("🎉 전반적으로 위험도가 낮습니다! 지금처럼 건강을 잘 유지하세요.")

st.markdown("---")
st.caption("🧠 Powered by LightGBM + KoGPT2 | Made with ❤️ using Streamlit")
