import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll")

import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import json
import torch
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
import arabic_reshaper
from bidi.algorithm import get_display

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني وتحليل المشاعر")

# Initialize Arabic text processor
def format_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Model selection
st.sidebar.header(format_arabic("⚙️ إعدادات النموذج"))
model_size = st.sidebar.radio(
    format_arabic("حجم نموذج الصوت"), 
    ["tiny", "base", "small"], 
    index=1,
    help=format_arabic("اختر tiny لتحليل أسرع أو small/base لدقة أعلى")
)

@st.cache_resource(show_spinner=False)
def load_whisper_model(size):
    st.info(format_arabic(f"⏳ جاري تحميل نموذج الصوت ({size})..."))
    return WhisperModel(
        size, 
        device="cpu", 
        compute_type="int8",
        download_root="./whisper_models"
    )

whisper_model = load_whisper_model(model_size)

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            truncation=True,
            max_length=256,
            top_k=1
        )
    except Exception as e:
        st.error(format_arabic(f"❌ فشل تحميل نموذج تحليل المشاعر: {str(e)}"))
        st.stop()

sentiment_pipeline = load_sentiment_model()

# Enhanced corrections dictionary with regex patterns
corrections = [
    (r"\bالفتور\b", "الفاتورة"),
    (r"\bزياد\b", "زيادة"),
    (r"\bالليزوم\b", "اللزوم"),
    (r"\bالمصادة\b", "المساعدة"),
    (r"\bبدي بطل\b", "بدي أبدل"),
    (r"\bمع بول\b", "مع بوليصة"),
    (r"\bتازي\b", "تازة"),
    (r"\bإ?دام الفني\b", "أداء الفني"),
    (r"\bبالدي\b", "بدي"),
    (r"\bالجودي\b", "الجودة"),
    (r"\bالتوقعاتي\b", "توقعاتي"),
    (r"\bمع طوب\b", "معطوب"),
    (r"\bأبريبا ديل\b", "أبغي بديل"),
    (r"\bشنل\b", "شنو"),
    (r"\bالعلان\b", "الإعلان"),
    (r"\bالمواضف\b", "الموظف"),
    (r"\bمهدب\b", "مهذب"),
    (r"\bالضفع\b", "الدفع"),
    (r"\bبيقال\b", "بايبال"),
    (r"\bاللعظم\b", "اللازم"),
    (r"\bينفعاد\b", "ينفع اعدل"),
    (r"\bأبل ميت شهر\b", "قبل ما يتشحن"),
    (r"\bلخبر هك ما بسير\b", "الخبر هيك ما بصير"),
    (r"\bيعتيكم\b", "يعطيكم"),
    (r"\bعن جد التعامل\b", "عنجد التعامل"),
    (r"\bوشحن وصلت\b", "والشحنة وصلت"),
    (r"\bما تابكون التوصيل\b", "متى بيكون التوصيل"),
    (r"\bتأخر واجد\b", "تأخر كثير"),
    (r"\bعضروري\b", "ضروري"),
    (r"\bتبكون\b", "بيكون"),
    (r"\bبدي ألغي طلب رأم واحد لاتي خمس سبعة تسعة\b", "بدي ألغي طلب رقم 13579"),
    (r"\bالمنتقج\b", "المنتج"),
    (r"\bأفين\b", "أين"),
    (r"\bاللسة\b", "لحد الآن"),
    (r"\bمحد\b", "ما حد"),
    (r"\bها كررها\b", "رح أكررها"),
    (r"\bأبدل\b", "أبدل"),
    (r"\bرأم\b", "رقم"),
    (r"\bغيط\b", "الغيت"),
    (r"\bسعى\b", "تسعة"),
    (r"\bلاتي\b", "واحد"),
    (r"\bخمس سبعات\b", "خمسة سبعة"),
    (r"\bبدأل\b", "بدي ألغي"),
    (r"\bعزر الجعو\b", "على الجودة"),
    (r"\bتأبل\b", "طريقة"),
    (r"\bاضافع\b", "الدفع"),
    (r"\bمنتك\b", "المنتج"),
    (r"\bمأبول\b", "مقبول"),
]

# Common Arabic technical terms
tech_terms = {
    "فاتورة": "فاتورة",
    "دفع": "دفع",
    "منتج": "منتج",
    "توصيل": "توصيل",
    "شحن": "شحن",
    "خدمة": "خدمة",
    "دعم": "دعم",
    "فني": "فني",
    "جودة": "جودة",
    "استبدال": "استبدال",
    "إرجاع": "إرجاع",
    "ضمان": "ضمان",
    "بطاقة": "بطاقة",
    "دفع": "دفع",
    "تأخير": "تأخير",
    "طلب": "طلب",
    "رقم": "رقم",
    "عميل": "عميل",
    "مشكلة": "مشكلة",
    "حل": "حل",
    "تسريع": "تسريع",
    "استفسار": "استفسار"
}

def clean_text(text):
    # Remove non-Arabic characters except spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    # Remove extra spaces
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    # First pass: fix common mistakes with regex
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text)
    
    # Second pass: ensure technical terms are correct
    for term, correct in tech_terms.items():
        if term in text:
            text = text.replace(term, correct)
    
    return text

def transcribe_audio(path):
    try:
        segments, info = whisper_model.transcribe(
            path, 
            beam_size=5,
            vad_filter=True,
            language="ar",
            initial_prompt="تحدث عن الفواتير والدفع والمنتجات والتوصيل والدعم الفني والخدمات التقنية",
            word_timestamps=False
        )
        
        # Collect segments with confidence scores
        transcriptions = []
        for segment in segments:
            if segment.text.strip():
                transcriptions.append({
                    "text": segment.text,
                    "confidence": segment.avg_logprob
                })
        
        # Combine high-confidence segments first
        high_conf = [t["text"] for t in transcriptions if t["confidence"] > -0.5]
        low_conf = [t["text"] for t in transcriptions if t["confidence"] <= -0.5]
        
        return " ".join(high_conf + low_conf)[:4000]  # Limit to 4000 characters
    except Exception as e:
        st.error(format_arabic(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}"))
        return ""

# File upload section
uploaded_files = st.file_uploader(
    format_arabic("📂 ارفع ملفات صوتية (WAV, MP3, FLAC)"), 
    type=["wav", "mp3", "flac"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(format_arabic(f"🔄 جاري تحليل {len(uploaded_files)} مكالمة..."))
    start_time = time.time()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress_percent = int((i + 1) / len(uploaded_files) * 100)
        progress_bar.progress(progress_percent)
        status_text.text(format_arabic(f"📞 جاري معالجة المكالمة {i+1}/{len(uploaded_files)}: {uploaded_file.name}"))
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            call_id = os.path.splitext(uploaded_file.name)[0]
            
            # Transcription
            raw_text = transcribe_audio(tmp_path)
            
            # Skip processing if transcription failed
            if not raw_text.strip():
                results.append({
                    "call_id": call_id,
                    "error": "فشل التحويل الصوتي",
                    "text_raw": "",
                    "text_clean": "",
                    "text_corrected": "",
                    "sentiment_label": "error",
                    "sentiment_score": 0.0,
                    "rank": "Error"
                })
                continue
            
            # Text processing
            clean_text = clean_text(raw_text)
            corrected_text = manual_correction(clean_text)
            
            # Sentiment analysis (skip if empty)
            if not corrected_text.strip():
                sentiment = {"label": "neutral", "score": 0.0}
                st.warning(format_arabic(f"المكالمة {call_id}: النص فارغ بعد التنظيف"))
            else:
                # Truncate to 256 tokens for model input
                sentiment_result = sentiment_pipeline(corrected_text[:1000])
                sentiment = sentiment_result[0] if sentiment_result else {"label": "neutral", "score": 0.0}
            
            # Determine rank
            label = sentiment["label"]
            score = round(sentiment["score"], 2)
            
            # Enhanced ranking system
            if label == "negative":
                if score > 0.85:
                    rank = format_arabic("عالية جداً")
                elif score > 0.7:
                    rank = format_arabic("عالية")
                else:
                    rank = format_arabic("متوسطة")
            elif label == "positive":
                rank = format_arabic("إيجابية")
            else:
                rank = format_arabic("محايدة")

            results.append({
                "call_id": call_id,
                "text_raw": raw_text,
                "text_clean": clean_text,
                "text_corrected": corrected_text,
                "sentiment_label": label,
                "sentiment_score": score,
                "rank": rank
            })
            
        except Exception as e:
            st.error(format_arabic(f"❌ خطأ في معالجة المكالمة {uploaded_file.name}: {str(e)}"))
            results.append({
                "call_id": uploaded_file.name,
                "error": str(e),
                "text_raw": "",
                "text_clean": "",
                "text_corrected": "",
                "sentiment_label": "error",
                "sentiment_score": 0.0,
                "rank": "Error"
            })
        finally:
            # Clean up temporary file
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(uploaded_files) if uploaded_files else 0
    st.success(format_arabic(f"✅ اكتمل التحليل! الوقت الإجمالي: {total_time:.1f} ثانية ({avg_time:.1f} ثانية/مكالمة)"))

    if results:
        df = pd.DataFrame(results)
        
        # Display results with tabs
        tab1, tab2, tab3 = st.tabs(["النتائج", "الرسوم البيانية", "تحميل البيانات"])
        
        with tab1:
            st.subheader(format_arabic("📋 ملخص النتائج"))
            st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], 
                         use_container_width=True, height=400)
            
            # Detailed view
            st.subheader(format_arabic("🔍 التفاصيل الكاملة"))
            for idx, row in df.iterrows():
                with st.expander(format_arabic(f"المكالمة: {row['call_id']} (المشاعر: {row['sentiment_label']})")):
                    st.caption(format_arabic("النص الأصلي:"))
                    st.write(format_arabic(row['text_raw']))
                    st.caption(format_arabic("النص المصحح:"))
                    st.write(format_arabic(row['text_corrected']))
                    st.caption(format_arabic(f"تحليل المشاعر: {row['sentiment_label']} (ثقة: {row['sentiment_score']:.2f})"))
                    st.caption(format_arabic(f"الأهمية: {row['rank']}"))
        
        with tab2:
            st.subheader(format_arabic("📊 التحليل البصري"))
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(df, names="sentiment_label", title=format_arabic("توزيع المشاعر"),
                              color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig3 = px.histogram(df, x="sentiment_score", nbins=20, 
                                   title=format_arabic("توزيع درجات المشاعر"),
                                   color="sentiment_label")
                st.plotly_chart(fig3, use_container_width=True)
                
            with col2:
                fig2 = px.bar(df, x="rank", color="rank", 
                             title=format_arabic("تصنيف أهمية المكالمات"),
                             category_orders={"rank": [format_arabic("عالية جداً"), 
                                                     format_arabic("عالية"), 
                                                     format_arabic("متوسطة"), 
                                                     format_arabic("إيجابية"), 
                                                     format_arabic("محايدة"), 
                                                     "Error"]})
                st.plotly_chart(fig2, use_container_width=True)
                
                fig4 = px.scatter(df, x="sentiment_score", y="rank", color="sentiment_label",
                                title=format_arabic("العلاقة بين درجة المشاعر والأهمية"))
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            st.subheader(format_arabic("⬇️ تحميل البيانات"))
            st.caption(format_arabic("يمكنك تنزيل النتائج كاملة بتنسيق JSON أو CSV"))
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    format_arabic("📥 تحميل JSON"), 
                    json.dumps(results, ensure_ascii=False, indent=2), 
                    file_name="call_results.json", 
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    format_arabic("📥 تحميل CSV"), 
                    df.to_csv(index=False).encode("utf-8-sig"), 
                    file_name="call_results.csv", 
                    mime="text/csv"
                )
            
            # Show sample JSON
            st.caption(format_arabic("معاينة JSON:"))
            st.json(results[0] if len(results) > 0 else {})
