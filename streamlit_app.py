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

# إعداد الصفحة
st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني بدقة عالية")

# إدارة الذاكرة
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# تحميل نموذج Whisper
@st.cache_resource
def load_whisper_model():
    try:
        model = WhisperModel("base", device="cpu", compute_type="float32")
        return model
    except Exception as e:
        st.error(f"❌ Whisper Model Load Failed: {str(e)}")
        return None

whisper_model = load_whisper_model()

# نموذج تحليل المشاعر
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.warning(f"⚠️ فشل في تحميل نموذج تحليل المشاعر: {e}")
        return None

# نموذج استخراج الموضوع
@st.cache_resource
def load_topic_model():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.warning(f"⚠️ تعذر تحميل نموذج استخراج الموضوع: {e}")
        return None

sentiment_pipeline = load_sentiment_model()
topic_pipeline = load_topic_model()

# تصحيحات اللهجات
corrections = {
    "شنل": "شنو", "أبريبا": "أبغي", "مع بول": "مع بوليصة", "تازي": "تازة",
    "ادام الفني": "أداء الفني", "الفتور": "الفاتورة", "زياد": "زيادة",
    "الليزوم": "اللزوم", "المصادة": "المساعدة", "بدي بطل": "بدي أبدل",
    "فين": "أين", "العلان": "الإعلان", "تأبل": "تقبل", "الضفع": "الدفع",
    "بيقال": "باي بال", "المواضف": "الموظف", "مهدب": "مهذب",
    "طوب": "معطوب", "ديل": "بديل", "ميت": "ما يت", "اد": "أعدّل"
}

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def clean_text(text):
    return re.sub(r"[^\u0600-\u06FF\s]", "", text).strip()

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path)
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"❌ خطأ تحويل الصوت: {e}")
        return ""

# مواضيع مقترحة للاستخدام في تصنيف الموضوع
TOPIC_CANDIDATES = [
    "الدفع", "الشحن", "الإرجاع", "المنتج", "السعر", "التوصيل",
    "مشكلة فنية", "طلب مساعدة", "الضمان", "استفسار عام", "خدمة العملاء"
]

# تحميل الملفات
uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files and whisper_model and sentiment_pipeline:
    st.info("🔄 جاري المعالجة...")
    results = []

    with st.spinner("⏳ معالجة الملفات..."):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                call_id = os.path.splitext(uploaded_file.name)[0]
                raw = transcribe_audio(tmp_path)
                clean = clean_text(raw)
                corrected = manual_correction(clean)

                # تحليل المشاعر
                try:
                    sentiment = sentiment_pipeline(corrected)[0] if len(corrected.split()) >= 3 else {"label": "neutral", "score": 0.5}
                except:
                    sentiment = {"label": "neutral", "score": 0.5}

                # استخراج الموضوع
                try:
                    topic = topic_pipeline(corrected, TOPIC_CANDIDATES)
                    best_topic = topic["labels"][0]
                except:
                    best_topic = "غير محدد"

                rank = "High" if sentiment["label"] == "negative" and sentiment["score"] > 0.8 else \
                       "Medium" if sentiment["label"] == "negative" else "Low"

                results.append({
                    "call_id": call_id,
                    "text_raw": raw,
                    "text_clean": clean,
                    "text_corrected": corrected,
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": round(sentiment["score"], 2),
                    "rank": rank,
                    "topic": best_topic
                })

                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"❌ خطأ في معالجة الملف {uploaded_file.name}: {e}")

    if not results:
        st.warning("⚠️ لم يتم إنتاج نتائج")
        st.stop()

    df = pd.DataFrame(results)

    # جدول النتائج
    st.subheader("📋 النتائج")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank", "topic"]], use_container_width=True)

    # الرسوم البيانية
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df, x="topic", color="rank", title="تصنيف حسب المواضيع"), use_container_width=True)

    # تحميل البيانات
    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 تحميل JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 تحميل CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")

    clear_memory()

elif not whisper_model:
    st.error("❌ لم يتم تحميل نموذج تحويل الصوت")
elif not sentiment_pipeline:
    st.error("❌ لم يتم تحميل نموذج تحليل المشاعر")
