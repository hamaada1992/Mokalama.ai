import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني بدقة عالية")
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu")

whisper_model = load_whisper_model()

@st.cache_resource
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة", "تازي": "تازة", "ادام الفني": "أداء الفني",
    "اخذ وقت اكثر من اللّعظم": "أخذ وقت أكثر من اللازم", "اللعظم": "اللازم", "مش زي ما مكتوب": "مش زي ما هو مكتوب",
    "بأفين": "بقى فين", "فين": "أين", "واللسه": "ولسه", "تجربتي معاكم كانت متس": "تجربتي معاكم كانت ممتازة",
    "هكررها": "سأكررها", "تانيا": "ثانية", "ينفعاد": "ينفع أعدّل", "ينفع اد": "ينفع أعدّل", "أبل": "قبل",
    "ما حده": "ما حدا", "لخبر هك": "الخبر هيك", "بسير": "بصير", "يعتيكم": "يعطيكم", "عافي": "العافية",
    "تأخر واجد": "تأخر كثير", "واجد": "كثير", "ضروري": "بشكل عاجل",
    "لو سمحت متى بيكون التوصيل للرياض بالعادة": "متى يوصل الطلب للرياض عادة؟",
    "يوماين": "يومين", "ما تبكون": "ما تكونون", "عضروري": "ضروري"
}

def clean_text(text):
    import re
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
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
    st.info("🔄 جاري المعالجة...")
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
            if corrected.strip() == "" or len(corrected.split()) < 3:
                sentiment = {"label": "neutral", "score": 0.5}
            else:
                sentiment = sentiment_pipeline(corrected)[0]
        except Exception:
            sentiment = {"label": "neutral", "score": 0.5}
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
            "rank": rank
        })

    df = pd.DataFrame(results)
    st.subheader("📋 النتائج")
    st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df, x="rank", color="rank", title="تصنيف المكالمات"), use_container_width=True)

    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")
