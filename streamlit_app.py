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
st.title("🎧 تحليل مكالمات الدعم الفني بدقة وسرعة")

@st.cache_resource
def load_whisper_model():
    try:
        # استخدام نموذج أصغر لتسريع المعالجة
        model = WhisperModel("tiny", device="cpu", compute_type="float32")
        st.success("✅ تم تحميل نموذج Whisper (tiny) بنجاح")
        return model
    except Exception as e:
        st.error(f"❌ فشل في تحميل نموذج Whisper: {str(e)}")
        return None

whisper_model = load_whisper_model()

@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        st.success("✅ تم تحميل نموذج تحليل المشاعر بنجاح")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"❌ فشل في تحميل نموذج تحليل المشاعر: {str(e)}")
        return None

sentiment_pipeline = load_sentiment_model()

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة", "تازي": "تازة", "ادام الفني": "أداء الفني",
    "عايز": "أريد", "هينفع": "ينفع", "كده": "هكذا", "زي": "مثل", "اتأخرت": "تأخرت",
    "وش": "ما", "أبغى": "أريد", "معطوب": "تالف", "طلبية": "طلب", "مافي": "لا يوجد",
    "بدي": "أريد", "كتير": "كثير", "ما بصير": "لا يجوز", "ردلي": "رد علي", "هيك": "هكذا",
    "عنجد": "حقًا", "لساتني": "ما زلت"
}

topics_keywords = {
    "الدفع": ["دفع", "فاتورة", "بيبال", "بطاقة", "تحويل"],
    "الشحن": ["شحن", "توصيل", "موعد", "وصل", "تأخر", "استلام"],
    "الاسترجاع": ["استرجاع", "إرجاع", "بديل", "مكسور", "تبديل"],
    "الجودة": ["جودة", "تالف", "معطوب", "سيئ", "ممتاز", "كسور"],
    "العروض": ["عرض", "عروض", "خصم", "تخفيض", "سعر خاص"],
    "الدعم الفني": ["دعم", "فني", "مساعدة", "مشاكل", "الموظف", "خدمة"],
    "العنوان": ["عنوان", "موقع", "الرياض", "تعديل", "منطقتي"]
}

def detect_topic(text):
    scores = {topic: sum(text.count(k) for k in keys) for topic, keys in topics_keywords.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "غير معروف"

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\u0600-\u06FF\s]", "", text)).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path, beam_size=1)
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files and whisper_model and sentiment_pipeline:
    st.info("🔄 جاري المعالجة... الرجاء الانتظار")
    results = []

    with st.spinner("⏳ يتم تحويل وتحليل المكالمات..."):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                call_id = os.path.splitext(uploaded_file.name)[0]
                raw = transcribe_audio(tmp_path)
                clean = clean_text(raw)
                corrected = manual_correction(clean)
                topic = detect_topic(corrected)

                sentiment = {"label": "neutral", "score": 0.5}
                if len(corrected.split()) >= 3:
                    try:
                        sentiment = sentiment_pipeline(corrected)[0]
                    except Exception as e:
                        st.warning(f"⚠️ تعذر تحليل الجملة: {corrected}")

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
                st.error(f"❌ خطأ في ملف {uploaded_file.name}: {str(e)}")

    if not results:
        st.error("❌ لم يتم إنتاج أي نتائج")
        st.stop()

    df = pd.DataFrame(results)
    st.subheader("📋 النتائج")

    sentiment_filter = st.multiselect("تصفية حسب المشاعر", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
    filtered_df = df[df["sentiment_label"].isin(sentiment_filter)] if sentiment_filter else df

    def color_sentiment(row):
        base = "color: black; text-align: right"
        if row["sentiment_label"] == "negative":
            return [f"{base}; background-color: #ffcccc"] * len(row)
        elif row["sentiment_label"] == "positive":
            return [f"{base}; background-color: #ccffcc"] * len(row)
        else:
            return [f"{base}; background-color: #ffffcc"] * len(row)

    st.dataframe(
        filtered_df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank", "topic"]].style.apply(color_sentiment, axis=1),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="topic", title="توزيع المواضيع"), use_container_width=True)

    st.download_button("📥 تحميل JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 تحميل CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")

    clear_memory()
else:
    st.warning("🚫 يرجى رفع ملفات صوتية وتحميل النماذج لإظهار النتائج.")
