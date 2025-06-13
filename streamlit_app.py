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

# إصلاح مشكلة torch.classes
try:
    if hasattr(torch.classes, '__path__'):
        pass
except RuntimeError as e:
    st.warning(f"⚠️ تم تجاوز مشكلة torch.classes: {str(e)}")

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني بدقة عالية")

@st.cache_resource
def load_whisper_model():
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="float32")
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

# إدارة الذاكرة المعدلة
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # تمت إزالة استدعاء legacy_caching.clear_cache() لأنه لم يعد مدعوماً

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

                try:
                    if corrected.strip() == "" or len(corrected.split()) < 3:
                        sentiment = {"label": "neutral", "score": 0.5}
                    else:
                        sentiment = sentiment_pipeline(corrected)[0]
                except Exception as e:
                    sentiment = {"label": "neutral", "score": 0.5}
                    st.warning(f"⚠️ تعذر تحليل الجملة: {corrected} - {str(e)}")

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
                
                # حذف الملف المؤقت
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
            except Exception as e:
                st.error(f"❌ خطأ في معالجة الملف {uploaded_file.name}: {str(e)}")

    if not results:
        st.error("❌ لم يتم إنتاج أي نتائج، يرجى التحقق من الملفات وحاول مرة أخرى")
        st.stop()

    df = pd.DataFrame(results)
    
    # التطويرات الجديدة: فلتر وتلوين الجدول مع تحسين قابلية القراءة
    st.subheader("📋 النتائج")
    
    # فلتر حسب المشاعر
    sentiment_filter = st.multiselect(
        "تصفية حسب المشاعر",
        options=["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"]
    )
    
    if sentiment_filter:
        filtered_df = df[df["sentiment_label"].isin(sentiment_filter)]
    else:
        filtered_df = df.copy()
    
    # تلوين الصفوف حسب المشاعر مع ضمان قابلية القراءة
    def color_sentiment(row):
        styles = ["color: black; text-align: right"] * len(row)  # نص أسود ومحاذاة لليمين
        
        if row["sentiment_label"] == "negative":
            styles = [f"{s}; background-color: #ffcccc" for s in styles]
        elif row["sentiment_label"] == "positive":
            styles = [f"{s}; background-color: #ccffcc" for s in styles]
        else:
            styles = [f"{s}; background-color: #ffffcc" for s in styles]  # اللون الأصفر للمشاعر المحايدة
        
        return styles
    
    # عرض الجدول مع التلوين
    if not filtered_df.empty:
        styled_df = filtered_df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]] \
            .style.apply(color_sentiment, axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("⚠️ لا توجد نتائج تطابق عوامل التصفية")

    col1, col2 = st.columns(2)
    with col1:
        if not df.empty:
            st.plotly_chart(px.pie(df, names="sentiment_label", title="توزيع المشاعر"), use_container_width=True)
        else:
            st.warning("⚠️ لا توجد بيانات للرسم البياني")
    with col2:
        if not df.empty:
            st.plotly_chart(px.bar(df, x="rank", color="rank", title="تصنيف المكالمات"), use_container_width=True)
        else:
            st.warning("⚠️ لا توجد بيانات للرسم البياني")

    st.subheader("⬇️ تحميل النتائج")
    st.download_button("📥 JSON", json.dumps(results, ensure_ascii=False, indent=2), file_name="call_results.json", mime="application/json")
    st.download_button("📥 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="call_results.csv", mime="text/csv")

    # تنظيف الذاكرة
    clear_memory()
elif not whisper_model or not sentiment_pipeline:
    st.error("❌ لم يتم تحميل النماذج بنجاح، يرجى تحديث الصفحة وحاول مرة أخرى")
