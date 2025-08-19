# -*- coding: utf-8 -*-
# app_gpt.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
import json
import requests
from openai import OpenAI

# ==============================
# 🔑 API 키 불러오기
# ==============================
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
HF_API_KEY = st.secrets.get("huggingface", {}).get("api_key")
HF_MODEL = st.secrets.get("huggingface", {}).get("model", "nvidia/Llama-3.3-Nemotron-Super-49B-v1.5")
HF_API_URL = st.secrets.get("huggingface", {}).get(
    "api_url", f"https://api-inference.huggingface.co/models/{HF_MODEL}"
)

# ✅ 백엔드 자동 선택
BACKEND = "hf" if HF_API_KEY else ("openai" if OPENAI_API_KEY else None)
if BACKEND is None:
    st.error("❌ 사용할 API 키가 없습니다. secrets.toml에 [huggingface.api_key] 또는 [openai.api_key]를 등록해주세요.")
    st.stop()

st.caption(f"⚙️ 현재 백엔드: {BACKEND.upper()}")

# ==============================
# 🧠 맞춤형 조언 생성 함수
# ==============================
def generate_lifestyle_advice(risk_factors: dict) -> str:
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

    if BACKEND == "hf":
        # Hugging Face 호출
        try:
            resp = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "repetition_penalty": 1.05,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "error" in data:
                return f"⚠️ 모델 응답 대기 또는 오류: {data.get('error')}"
            return f"⚠️ 예상치 못한 응답: {data}"
        except Exception as e:
            return f"❌ Hugging Face 호출 실패: {str(e)}"

    else:
        # OpenAI 호출
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 심혈관질환 예방 및 관리에 전문성을 가진 의사입니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI 호출 실패: {str(e)}"

# ==============================
# 📊 데모용 LightGBM 모델
# ==============================
@st.cache_resource
def load_model():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=11, random_state=42)
    lgb_train = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=10)
    return model

model = load_model()

# ==============================
# ⚖️ 입력값 전처리 함수
# ==============================
def classify_cholesterol(chol_value):
    if chol_value < 200: return 1
    elif 200 <= chol_value < 240: return 2
    else: return 3

def classify_gluc(gluc_value):
    if gluc_value < 100: return 1
    elif 100 <= gluc_value < 126: return 2
    else: return 3

# ==============================
# 🖥️ Streamlit UI
# ==============================
st.set_page_config(page_title="심혈관질환 예측기", layout="centered")
st.title("🩺 심혈관질환 10년 위험도 예측기")

st.header("📋 건강 정보 입력")
age = st.slider("나이", 20, 90, 50)
gender = st.radio("성별", ["남성", "여성"])
ap_hi = st.number_input("수축기 혈압", value=120)
ap_lo = st.number_input("이완기 혈압", value=80)
height_cm = st.number_input("키(cm)", value=170)
weight_kg = st.number_input("몸무게(kg)", value=65)
chol = st.number_input("콜레스테롤 수치 (mg/dL)", min_value=100, max_value=400, value=180)
gluc = st.number_input("혈당 수치 (공복 mg/dL)", min_value=50, max_value=300, value=90)
smoke = st.checkbox("흡연")
alco = st.checkbox("음주")
active = st.checkbox("활동적 생활")

# 입력 처리
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
    "cholesterol": chol_cat,
    "gluc": gluc_cat,
    "smoke": int(smoke),
    "alco": int(alco),
    "active": int(active),
    "hypertension": hypertension,
}

# 예측 함수
def predict_risk(model, user_input: dict):
    features = ["age_years", "gender", "ap_hi", "ap_lo", "bmi",
                "cholesterol", "gluc", "smoke", "alco", "active", "hypertension"]
    df = pd.DataFrame([user_input])[features]
    return float(model.predict(df)[0])

# 버튼 동작
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
        st.markdown("💡 **맞춤형 건강 처방 (백엔드: {} 사용)**".format(BACKEND.upper()))
        with st.spinner("모델이 건강 조언을 생성 중입니다..."):
            advice = generate_lifestyle_advice(risk_factors)
            st.success("생활 처방 도착 ✅")
            st.markdown(advice)
    else:
        st.success("🎉 전반적으로 위험도가 낮습니다! 지금처럼 건강을 잘 유지하세요.")

st.markdown("---")
st.caption("🧠 Powered by LightGBM + OpenAI/HuggingFace (Nemotron 49B) | Made with ❤️ using Streamlit")
