import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json

from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# إعدادات الصفحة
st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني وتحليل المشاعر")

# تحميل نموذج Whisper
@st.cache_resource
def load_whisper_model():     return WhisperModel("tiny", device="cpu")
", device="cpu")

whisper_model = load_whisper_model()

# تحميل نموذج تحليل المشاعر
@st.cache_resource
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# التصحيحات اليدوية
corrections = {
    "الفتور": "الفاتورة",
    "زياد": "زيادة",
    "الليزوم": "اللزوم",
    "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل",
    "مع بول": "مع بوليصة",
    "تازي": "تازة",
    "ادام الفني": "أداء الفني"
}

def clean_text(text):
    import re
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    return " ".join([seg.text for seg in segments])

# واجهة رفع الملفات
uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية (WAV/MP3)", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files:
    st.info("جاري المعالجة...")
    results = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        call_id = os.path.splitext(uploaded_file.name)[0]
        raw = transcribe_audio(tmp_path)
        clean = clean_text(raw)
        corrected = manual_correction(clean)
        sentiment = sentiment_pipeline(corrected)[0]
        label = sentiment["label"]
        score = round(sentiment["score"], 2)
        rank = "High" if label == "negative" and score > 0.8 else "Medium" if label == "negative" else "Low"

        results.append({
            "call_id": call_id,
            "text_raw": raw,
            "text_clean": clean,
            "text_corrected": corrected,
            "sentiment_label": label,
            "sentiment_score": score,
            "rank": rank
        })

    df = pd.DataFrame(results)

    # عرض النتائج
    st.subheader("📋 نتائج التحليل")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], use_container_width=True)

    # رسوم بيانية
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names="sentiment_label", title="توزيع المشاعر")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df, x="rank", color="rank", title="تصنيف المكالمات")
        st.plotly_chart(fig2, use_container_width=True)

    # تحميل النتائج
    st.subheader("⬇️ تحميل النتائج")
    json_str = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button("📥 تحميل كـ JSON", data=json_str, file_name="call_results.json", mime="application/json")

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 تحميل كـ CSV", data=csv_data, file_name="call_results.csv", mime="text/csv")
