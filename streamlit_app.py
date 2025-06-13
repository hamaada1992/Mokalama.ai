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
st.title("🎧 تحليل مكالمات الدعم الفني بدقة عالية")

@st.cache_resource
def load_whisper_model():
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        st.success("✅ تم تحميل نموذج Whisper بنجاح")
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

@st.cache_resource
def load_topic_model():
    try:
        from transformers import AutoModelForSequenceClassification
        topic_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-topic"
        tokenizer = AutoTokenizer.from_pretrained(topic_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(topic_model_name)
        topic_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        st.success("✅ تم تحميل نموذج استخراج الموضوع بنجاح")
        return topic_pipeline
    except Exception as e:
        st.warning("⚠️ تعذر تحميل نموذج استخراج الموضوع")
        return None

topic_pipeline = load_topic_model()

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم",
    "المصادة": "المساعدة", "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة",
    "تازي": "تازة", "ادام الفني": "أداء الفني", "اللعظم": "اللازم",
    "مش زي ما مكتوب": "مش زي ما هو مكتوب", "بأفين": "بقى فين", "واللسه": "ولسه",
    "تجربتي معاكم كانت متس": "تجربتي معاكم كانت ممتازة", "هكررها": "سأكررها",
    "ينفعاد": "ينفع أعدّل", "أبل": "قبل", "ما حده": "ما حدا", "بسير": "بصير",
    "يعتيكم": "يعطيكم", "عافي": "العافية", "واجد": "كثير", "يوماين": "يومين",
    "ما تبكون": "ما تكونون", "عضروري": "ضروري", "ميت شهر": "ما يتشحن",
    "تأبل": "تقبل", "الضفع": "الدفع", "بيقال": "باي بال", "المواضف": "الموظف",
    "مهدب": "مهذب", "العلان": "الإعلان", "شنل": "شنو", "طوب": "معطوب",
    "أبريبا": "أبغي", "ديل": "بديل", "اد": "أعدّل", "العنوان": "عنوان",
    "ميت": "ما يت"
}

def clean_text(text):
    text = re.sub(r"[^؀-ۿ\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path)
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files and whisper_model and sentiment_pipeline:
    st.info("🔄 جاري المعالجة...")
    results = []

    with st.spinner("⏳ جاري معالجة الملفات..."):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                call_id = os.path.splitext(uploaded_file.name)[0]
                raw = transcribe_audio(tmp_path)
                clean = clean_text(raw)
                corrected = manual_correction(clean)

                sentiment = {"label": "neutral", "score": 0.5}
                topic = {"label": "unknown", "score": 0.0}
                if len(corrected.split()) > 2:
                    try:
                        sentiment = sentiment_pipeline(corrected)[0]
                    except: pass
                    if topic_pipeline:
                        try:
                            topic = topic_pipeline(corrected)[0]
                        except: pass

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
                    "topic": topic["label"]
                })

                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"❌ خطأ في الملف {uploaded_file.name}: {str(e)}")

    df = pd.DataFrame(results)
    st.subheader("📋 النتائج")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank", "topic"]], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df, x="rank", color="rank", title="تصنيف المكالمات"), use_container_width=True)

    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")

    clear_memory()
elif not whisper_model or not sentiment_pipeline:
    st.error("❌ لم يتم تحميل النماذج بنجاح، يرجى تحديث الصفحة")
