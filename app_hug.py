# -*- coding: utf-8 -*-
# app_hug.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
import requests
import os
import json

# ✅ Hugging Face API 설정
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM3-3B"
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
# 역할 (Role)
당신은 심혈관질환 예방과 관리를 전문으로 하는 의료 전문가입니다.
최신 연구와 임상 지침에 기반하여 일반인의 건강 행동을 효과적으로 변화시킬 수 있도록 돕는 역할을 맡고 있습니다.

# 대상 (Audience)
당신의 조언을 받을 대상은 건강에 관심이 높지만 의료 지식은 많지 않은 40~60대 일반 환자입니다.
이 환자는 최근 심혈관 위험 평가를 받았으며, 자신의 상태에 맞는 생활 습관 개선 방안을 찾고자 합니다.

# 입력 정보 (Input)
다음은 이 환자의 심혈관질환 관련 주요 위험 요인입니다:
{risk_factors}

# 작업 목표 (Task)
다음 항목을 포함하여, 환자 맞춤형 건강 처방을 제공하세요:

1. **각 위험 요인이 심혈관질환 발병에 어떤 영향을 주는지 설명**
   - 단순히 "위험하다"가 아니라, 왜 그런지 병태생리적으로 1~2문장 내외로 설명
   - 가능한 경우 숫자 (예: 혈압 140 이상이면 위험이 2배 증가) 활용

2. **각 위험 요인을 줄이기 위한 구체적인 실천 방안 제시**
   - 매일 걷기, 염분 줄이기 등 일상에서 쉽게 실천 가능한 행동으로
   - 각 행동이 해당 위험 요인을 어떻게 개선하는지도 간단히 설명

3. **개인화된 실천 팁 또는 격려 메시지 삽입**
   - 예: "50대 이후에는 혈압 관리가 특히 중요합니다."
   - 대상자의 나이, 성별, 위험 조합 등을 고려한 맞춤 코멘트

4. **필요 시 의료적 조치나 전문의 상담 권고 포함**
   - 특정 수치(혈압, 혈당 등)가 기준치를 넘는 경우 병원 진료를 권고
   - 약물 치료, 혈액 검사 등 실제적 조치도 언급 가능

5. **신뢰할 수 있는 참고 자료 2~3개 추천**
   - 유튜브 영상: 제목, 설명, 링크
   - 건강 저널: 논문 제목, 핵심 내용, 링크
   - 공공기관 건강 정보 사이트 등

6. **생활습관 변화 유지를 위한 '작은 성공 경험' 제안**
   - 행동을 습관화하기 위한 구체적이고 측정 가능한 예시 포함
   - 예: "매일 아침 혈압 측정하고 기록하기", "주 3회 친구와 걷기 챌린지 참여"

# 형식 및 톤 (Format & Tone)
- 각 위험 요인은 번호나 소제목으로 구분하여 명확하게 구성
- 중요한 실천 포인트는 굵은 글씨로 강조
- 설명은 전문성을 담되, 반드시 쉽게 이해할 수 있도록 쓰세요
- 필요 시 간단한 의료 용어는 ()나 예시로 풀어 설명
- 전체 어조는 친절하면서도 신뢰감 있는 의료 전문가 톤 유지
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
