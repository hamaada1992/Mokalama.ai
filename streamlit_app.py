import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc
import re

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني - نسخة محسّنة")

@st.cache_resource
def load_whisper_model():
    try:
        model = WhisperModel("medium", device="cpu", compute_type="float32")  # أكثر دقة
        st.success("✅ تم تحميل نموذج Whisper (medium)")
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل نموذج Whisper: {str(e)}")
        return None

@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        st.success("✅ تم تحميل نموذج تحليل المشاعر")
        return pipe
    except Exception as e:
        st.error(f"❌ فشل تحميل نموذج تحليل المشاعر: {str(e)}")
        return None

whisper_model = load_whisper_model()
sentiment_pipeline = load_sentiment_model()

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# تحسين تصحيح اللهجات
common_errors = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بيقال": "باي بال", "المون": "المنتج", "سكرًا": "شكراً", "تقزيي": "زي", "ديل": "بديل",
    "أبريبا": "أبغي", "طوب": "معطوب", "العلان": "الإعلان", "مهدب": "مهذب", "المواضف": "الموظف"
}

def normalize_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def correct_common_errors(text):
    for wrong, right in common_errors.items():
        text = text.replace(wrong, right)
    return text

# كشف الموضوع
topics_keywords = {
    "الدفع": ["دفع", "فاتورة", "بيبال", "بطاقة"],
    "الشحن": ["شحن", "توصيل", "موعد", "تأخر"],
    "الجودة": ["جودة", "مكسور", "تالف", "معطوب"],
    "الاسترجاع": ["بديل", "استرجاع", "إرجاع"],
    "العروض": ["عرض", "عروض", "خصم"],
    "الدعم الفني": ["موظف", "خدمة", "مهذب", "دعم"],
    "العنوان": ["عنوان", "موقع", "منطقتي", "الرياض"]
}

def detect_topic(text):
    scores = {k: sum(text.count(w) for w in ws) for k, ws in topics_keywords.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "غير معروف"

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path, beam_size=5)
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files and whisper_model and sentiment_pipeline:
    st.info("🔄 جاري التحويل والتحليل... انتظر قليلًا")
    results = []
    with st.spinner("🔁 المعالجة جارية..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                call_id = os.path.splitext(uploaded_file.name)[0]
                raw = transcribe_audio(tmp_path)
                clean = normalize_text(raw)
                corrected = correct_common_errors(clean)
                topic = detect_topic(corrected)

                if corrected.strip() == "" or len(corrected.split()) < 3:
                    sentiment = {"label": "neutral", "score": 0.5}
                else:
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
                    "rank": rank,
                    "topic": topic
                })
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"❌ خطأ في الملف {uploaded_file.name}: {str(e)}")

    df = pd.DataFrame(results)

    st.subheader("📊 جدول التحليل")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank", "topic"]], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="topic", title="توزيع المواضيع"), use_container_width=True)

    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")

    clear_memory()

else:
    st.warning("📂 يرجى رفع ملفات صوتية ووجود اتصال جيد لتحميل النماذج.")
