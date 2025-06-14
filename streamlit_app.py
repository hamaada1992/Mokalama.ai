import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix for PyTorch threading issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # Fix for file watcher error

import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import re

st.set_page_config(page_title="تحليل مكالمات الدعم", layout="wide")
st.title("🎧 تحليل مكالمات الدعم الفني وتحليل المشاعر")

# Model selection
st.sidebar.header("⚙️ إعدادات النموذج")
model_size = st.sidebar.radio("حجم نموذج الصوت", ["tiny", "base"], index=1, help="اختر tiny لتحليل أسرع أو base لدقة أعلى")

@st.cache_resource(show_spinner=False)
def load_whisper_model(size):
    st.info(f"⏳ جاري تحميل نموذج الصوت ({size})...")
    return WhisperModel(size, device="cpu", compute_type="int8")

whisper_model = load_whisper_model(model_size)

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer,
            truncation=True,
            max_length=512
        )
    except Exception as e:
        st.error(f"❌ فشل تحميل نموذج تحليل المشاعر: {str(e)}")
        st.stop()

sentiment_pipeline = load_sentiment_model()

# Enhanced corrections dictionary
corrections = {
    "الفتور": "الفاتورة",
    "زياد": "زيادة",
    "الليزوم": "اللّزوم",
    "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدّل",
    "مع بول": "مع بوليصة",
    "تازي": "تازة",
    "ادام الفني": "أداء الفني",
    "بالدي": "بدي",
    "إدام": "أداء",
    "الجودي": "الجودة",
    "التوقعاتي": "توقّعاتي",
    "مع طوب": "معطوب",
    "أبريبا ديل": "أبغى بديل",
    "شنل": "شنو",
    "العلان": "الإعلان",
    "المواضف": "الموظّف",
    "مهدب": "مهذّب",
    "الضفع": "الدفع",
    "بيقال": "بايبال",
    "اللعظم": "اللازم",
    "ينفعد": "ينفع أعدّل",
    "أبل ميت شهل": "قبل ما يتشحن",
    "لخبر هك ما بسير": "الخبر هيك ما بصير",
    "يعتيكم": "يعطيكم",
    "عن جد التعامل": "عنجد التعامل",
    "وشحن وصلت": "والشحنة وصلت",
    "ما تابكون": "متى بيكون",
    "عضروري": "ضروري",
    "تبكون": "بيكون",
    "ما تبكون التوصيل": "متى بيكون التوصيل",
    "بدي ألغي طلب رأم واحد لاتي خمس سبعة تسعة": "بدي ألغي طلب رقم ١٣٥٧٩",
    "المنتك": "المنتج",
    "التحفة": "رائع",
    "مأبول": "مقبول",
    "مواعد": "موعد",
    "تأخر واجد": "تأخّر كثير",
    "شن": "شنو",
    "ها كررها": "هأكرّرها",
    "بدير جعل من تج": "بدي أرجّع المنتج",
    "غيط لب رأم": "ألغي طلب رقم"
}

def clean_text(text):
    # Remove non-Arabic characters except spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    # Remove extra spaces
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(
            path, 
            beam_size=3,  # Faster decoding
            vad_filter=True,  # Remove silence
            language="ar"
        )
        return " ".join([seg.text for seg in segments])[:5000]  # Limit to 5000 characters
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

# File upload section
uploaded_files = st.file_uploader(
    "📂 ارفع ملفات صوتية (WAV, MP3, FLAC)", 
    type=["wav", "mp3", "flac"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"🔄 جاري تحليل {len(uploaded_files)} مكالمة...")
    start_time = time.time()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress_percent = int((i + 1) / len(uploaded_files) * 100)
        progress_bar.progress(progress_percent)
        status_text.text(f"📞 جاري معالجة المكالمة {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            call_id = os.path.splitext(uploaded_file.name)[0]
            
            # Transcription
            raw = transcribe_audio(tmp_path)
            
            # Skip processing if transcription failed
            if not raw.strip():
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
            clean = clean_text(raw)
            corrected = manual_correction(clean)
            
            # Sentiment analysis (skip if empty)
            if not corrected.strip():
                sentiment = {"label": "neutral", "score": 0.0}
                st.warning(f"المكالمة {call_id}: النص فارغ بعد التنظيف")
            else:
                # Truncate to 512 tokens for model input
                sentiment = sentiment_pipeline(corrected[:512])[0]
            
            # Determine rank
            label = sentiment["label"]
            score = round(sentiment["score"], 2)
            
            # Enhanced ranking system
            if label == "negative":
                if score > 0.85:
                    rank = "عالية جداً"
                elif score > 0.7:
                    rank = "عالية"
                else:
                    rank = "متوسطة"
            elif label == "positive":
                rank = "إيجابية"
            else:
                rank = "محايدة"

            results.append({
                "call_id": call_id,
                "text_raw": raw,
                "text_clean": clean,
                "text_corrected": corrected,
                "sentiment_label": label,
                "sentiment_score": score,
                "rank": rank
            })
            
        except Exception as e:
            st.error(f"❌ خطأ في معالجة المكالمة {uploaded_file.name}: {str(e)}")
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
    st.success(f"✅ اكتمل التحليل! الوقت الإجمالي: {total_time:.1f} ثانية ({avg_time:.1f} ثانية/مكالمة)")

    if results:
        df = pd.DataFrame(results)
        
        # Display results with tabs
        tab1, tab2, tab3 = st.tabs(["النتائج", "الرسوم البيانية", "تحميل البيانات"])
        
        with tab1:
            st.subheader("📋 ملخص النتائج")
            st.dataframe(df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], 
                         use_container_width=True, height=400)
            
            # Detailed view
            st.subheader("🔍 التفاصيل الكاملة")
            for idx, row in df.iterrows():
                with st.expander(f"المكالمة: {row['call_id']} (المشاعر: {row['sentiment_label']})"):
                    st.caption("النص الأصلي:")
                    st.write(row['text_raw'])
                    st.caption("النص المصحح:")
                    st.write(row['text_corrected'])
                    st.caption(f"تحليل المشاعر: {row['sentiment_label']} (ثقة: {row['sentiment_score']:.2f})")
                    st.caption(f"الأهمية: {row['rank']}")
        
        with tab2:
            st.subheader("📊 التحليل البصري")
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(df, names="sentiment_label", title="توزيع المشاعر",
                              color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig3 = px.histogram(df, x="sentiment_score", nbins=20, 
                                   title="توزيع درجات المشاعر",
                                   color="sentiment_label")
                st.plotly_chart(fig3, use_container_width=True)
                
            with col2:
                fig2 = px.bar(df, x="rank", color="rank", 
                             title="تصنيف أهمية المكالمات",
                             category_orders={"rank": ["عالية جداً", "عالية", "متوسطة", "إيجابية", "محايدة", "Error"]})
                st.plotly_chart(fig2, use_container_width=True)
                
                fig4 = px.scatter(df, x="sentiment_score", y="rank", color="sentiment_label",
                                title="العلاقة بين درجة المشاعر والأهمية")
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            st.subheader("⬇️ تحميل النتائج")
            st.caption("يمكنك تنزيل النتائج كاملة بتنسيق JSON أو CSV")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 تحميل JSON", 
                    json.dumps(results, ensure_ascii=False, indent=2), 
                    file_name="call_results.json", 
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    "📥 تحميل CSV", 
                    df.to_csv(index=False).encode("utf-8-sig"), 
                    file_name="call_results.csv", 
                    mime="text/csv"
                )
            
            # Show sample JSON
            st.caption("معاينة JSON:")
            st.json(results[0] if len(results) > 0 else {})
