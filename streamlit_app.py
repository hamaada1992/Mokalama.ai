
import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
import re
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني بدقة عالية")

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="float32")

@st.cache_resource
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

whisper_model = load_whisper_model()
sentiment_pipeline = load_sentiment_model()

corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة", "تازي": "تازة", "ادام الفني": "أداء الفني",
    "فين": "أين", "بأفين": "بقى فين", "شنل": "شنو", "المواضف": "الموظف", "مهدب": "مهذب",
    "العلان": "الإعلان", "طوب": "معطوب", "أبريبا": "أبغي", "ديل": "بديل", "اد": "أعدّل",
    "ميت": "ما يت", "ما تبكون": "ما تكونون", "عضروري": "ضروري", "يتشحل": "يتشحن",
    "تغليف": "التغليف", "الجودي": "الجودة", "ازبوعة": "أسبوع", "تجربتي معاكم كانت متس": "تجربتي معاكم كانت ممتازة",
    "هكررها": "سأكررها", "تانيا": "ثانية", "أبل": "قبل"
}

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def clean_text(text):
    text = re.sub(r"[^؀-ۿ\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    return " ".join([seg.text for seg in segments])

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files:
    st.info("🔄 جاري التحليل...")
    results = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        call_id = os.path.splitext(uploaded_file.name)[0]
        raw = transcribe_audio(tmp_path)
        clean = clean_text(raw)
        corrected = manual_correction(clean)

        if len(corrected.split()) < 3 or re.search(r"[a-zA-Z]", corrected):
            continue

        try:
            sentiment = sentiment_pipeline(corrected)[0]
        except Exception:
            sentiment = {"label": "neutral", "score": 0.5}

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

    if results:
        df = pd.DataFrame(results)
        st.subheader("📋 النتائج")
        st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names="sentiment_label", title="توزيع المشاعر")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(df, x="rank", color="rank", title="تصنيف المكالمات")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("⬇️ تحميل النتائج")
        st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
        st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")
    else:
        st.warning("⚠️ لم يتم العثور على نتائج قابلة للتحليل")
# إعداد الصفحة
st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني وتحليل المشاعر والموضوع")

# تحميل نموذج Whisper
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu")

whisper_model = load_whisper_model()

# تحميل نموذج المشاعر
@st.cache_resource
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# تحميل نموذج استخراج الموضوع
@st.cache_resource
def load_topic_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        hypothesis_template="المكالمة تتعلق بـ {}.",
        device=-1
    )

topic_pipeline = load_topic_model()

# قائمة المواضيع المحتملة
TOPIC_CANDIDATES = [
    "الدفع", "الشحن", "الإرجاع", "خدمة العملاء",
    "مشكلة فنية", "طلب مساعدة", "استفسار عام",
    "الضمان", "التوصيل", "المنتج", "السعر"
]

# التصحيحات اليدوية للكلمات الشائعة في اللهجات
corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم",
    "المصادة": "المساعدة", "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة",
    "تازي": "تازة", "ادام الفني": "أداء الفني", "أبري": "أبغي", "بدبت": "بدي بديل",
    "ادام الفنيكا": "أداء الفني كان", "اتاخير": "تأخير", "المسادة": "المساعدة",
    "معبول": "معقول", "ازبوعة": "أسبوع", "الجودي": "الجودة", "اتغليف": "التغليف",
    "اما كتير": "أعجبني كثيرًا", "العلال": "الإعلان", "الموضف": "الموظف",
    "أفين": "فين", "أبل": "قبل", "يتشحل": "يتشحن", "بالدي": "بدي",
    "شنل": "شنو", "مهدب": "مهذب", "طوب": "معطوب", "ديل": "بديل",
    "اد": "أعدّل", "ميت": "ما يت", "ما تبكون": "ما تكونون",
    "عضروري": "ضروري", "تجربتي معاكم كانت متس": "تجربتي معاكم كانت ممتازة",
    "هكررها": "سأكررها", "تانيا": "ثانية"
}

def clean_text(text):
    text = re.sub(r"[^؀-ۿ\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    return " ".join([seg.text for seg in segments])

uploaded_files = st.file_uploader("📂 ارفع ملفات صوتية", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files:
    st.info("🔄 جاري التحليل...")
    results = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        call_id = os.path.splitext(uploaded_file.name)[0]
        raw = transcribe_audio(tmp_path)
        clean = clean_text(raw)
        corrected = manual_correction(clean)

        try:
            sentiment = sentiment_pipeline(corrected)[0] if len(corrected.split()) >= 3 else {"label": "neutral", "score": 0.5}
        except:
            sentiment = {"label": "neutral", "score": 0.5}

        try:
            topic_result = topic_pipeline(corrected, TOPIC_CANDIDATES)
            best_topic = topic_result["labels"][0]
            best_topic_score = round(topic_result["scores"][0], 2)
        except:
            best_topic = "غير محدد"
            best_topic_score = 0.0

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
            "topic": best_topic,
            "topic_score": best_topic_score
        })

    df = pd.DataFrame(results)
    st.subheader("📋 النتائج")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank", "topic", "topic_score"]], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="topic", y="topic_score", title="مواضيع المكالمات"), use_container_width=True)

    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")
