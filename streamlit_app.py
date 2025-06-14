import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني وتحليل المشاعر")

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
    # 🧩 تصحيحات لغوية عامة
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم",
    "المصادة": "المساعدة", "الجودي": "الجودة", "اتغليف": "التغليف",
    "العلال": "الإعلان", "اد": "أعدّل", "ميت": "ما يت", "فين": "أين",

    # 🇪🇬 مصري
    "بأفين": "بقى فين", "مش زي ما مكتوب": "مش زي ما هو مكتوب", "هكررها": "سأكررها",
    "تاني": "ثانية", "عايز": "أريد", "مش فاهم": "لا أفهم", "متاخر": "متأخر", "تمام": "جيد",
    "تمام كده": "جيد جدًا", "حاسس": "أشعر", "عامل ايه": "كيف حالك", "عندي مشكلة": "أواجه مشكلة",

    # 🇸🇦 سعودي / خليجي
    "أبري": "أبغي", "أبريبا": "أبغي", "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة",
    "شنل": "شنو", "شنه": "شنو", "تازي": "تازة", "تأخر واجد": "تأخر كثير", "واجد": "كثير",
    "بالدي": "بدي", "عندي استفسار": "أريد أن أسأل", "خلنا": "دعنا", "وش": "ما",
    "ما عليه": "لا بأس", "يشتغل؟": "هل يعمل؟",

    # 🇵🇸🇱🇧🇯🇴🇸🇾 شامي
    "ما حده": "ما حدا", "بسير": "بصير", "لخبر هك": "الخبر هيك", "يعتيكم": "يعطيكم",
    "عافي": "العافية", "منيح": "جيد", "كتير": "كثير", "ماشي الحال": "تمام",
    "شو": "ما", "قديش": "كم", "قديش السعر": "كم السعر",

    # 🇲🇦🇩🇿🇹🇳 مغاربي
    "برشا": "كثير", "ما فهمتش": "لم أفهم", "خاطر": "لأن", "واش": "هل", "شنو": "ما",
    "زعما": "هل يمكن", "حاجة": "شيء", "علاش": "لماذا", "تقدا": "تصلح",
    "هادي": "هذه", "هديك": "تلك", "ماعجبنيش": "لم يعجبني",

    # 🇸🇩 سوداني
    "شنو": "ما", "كتير": "كثير", "تمام": "جيد", "كويس": "جيد", "ساكت": "فقط",
    "خلاص": "انتهى", "لسه": "ما زال", "غايتو": "على كل حال",

    # 🇾🇪 يمني
    "معي": "عندي", "ليش": "لماذا", "ذي": "الذي", "هانا": "أنا هنا",
    "قع": "كن", "انقد": "أدفع", "حيد": "اذهب", "شاص": "سيارة",

    # ✨ تصحيحات فنية وعملية
    "يتشحل": "يتشحن", "مشبوك": "متصل", "فاصل": "غير متصل", "شبكة تعبانة": "الاتصال ضعيف",
    "ما يرد": "لا يجيب", "اتصال سيء": "الاتصال ضعيف", "الصوت يقطع": "الصوت غير واضح",

    # تصحيحات تركيبية ومرادفات
    "ادام الفني": "أداء الفني", "ادام الفنيكا": "أداء الفني كان", "بدبت": "بدي بديل",
    "ازبوعة": "أسبوع", "أما كتير": "أعجبني كثيرًا", "الموضف": "الموظف"
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
    segments, _ = whisper_model.transcribe(path, language="ar")
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
