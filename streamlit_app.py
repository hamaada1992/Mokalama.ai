import streamlit as st
import os
import tempfile
import pandas as pd
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc
import re

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني")

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="float32")

@st.cache_resource
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def clean_text(text):
    return re.sub(r"[^\u0600-\u06FF\s]", "", text).strip()

corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة", "تازي": "تازة", "ادام الفني": "أداء الفني",
    "أبريبا": "أبغي", "ديل": "بديل", "العلان": "الإعلان", "شنل": "شنو", "فين": "أين",
    "الضفع": "الدفع", "بيقال": "باي بال", "المواضف": "الموظف", "مهدب": "مهذب"
}

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path, model):
    segments, _ = model.transcribe(path)
    return " ".join([seg.text for seg in segments])

whisper_model = load_whisper_model()
sentiment_pipeline = load_sentiment_model()

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files:
    results = []
    with st.spinner("⏳ جارٍ معالجة الملفات..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            call_id = os.path.splitext(uploaded_file.name)[0]
            raw = transcribe_audio(tmp_path, whisper_model)
            clean = clean_text(raw)
            corrected = manual_correction(clean)

            try:
                sentiment = sentiment_pipeline(corrected)[0] if len(corrected.split()) > 2 else {"label": "neutral", "score": 0.5}
            except:
                sentiment = {"label": "neutral", "score": 0.5}

            results.append({
                "call_id": call_id,
                "text_raw": raw,
                "text_clean": clean,
                "text_corrected": corrected,
                "sentiment_label": sentiment["label"],
                "sentiment_score": round(sentiment["score"], 2)
            })

            os.unlink(tmp_path)

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("📥 تحميل JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json")
        st.download_button("📥 تحميل CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv")
        clear_memory()
    else:
        st.warning("⚠️ لم يتم إنتاج نتائج")
