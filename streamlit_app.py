import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import json
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import re
import numpy as np

# تحسينات المظهر العام
st.set_page_config(
    page_title="تحليل مكالمات الدعم الفني",
    layout="wide",
    page_icon="📞",
    initial_sidebar_state="expanded"
)

# تخصيص الألوان
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --danger: #e74c3c;
        --warning: #f39c12;
        --dark: #2c3e50;
    }
    
    .st-emotion-cache-1y4p8pa {
        background-color: #f8f9fa;
    }
    
    .stAlert {
        border-left: 4px solid var(--danger);
        border-radius: 4px;
    }
    
    .critical-call {
        border: 2px solid var(--danger);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fff8f8;
    }
    
    .header {
        color: var(--dark);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📞 تحليل مكالمات الدعم الفني وتحليل المشاعر")

# ========== إعدادات النموذج ==========
st.sidebar.header("⚙️ إعدادات النموذج")
model_size = st.sidebar.radio(
    "حجم نموذج الصوت",
    ["tiny", "base"], 
    index=1,
    help="اختر tiny لتحليل أسرع أو base لدقة أعلى"
)

# ========== تحميل النماذج ==========
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

# ========== تحسينات القاموس ==========
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
    "شن": "شنو",
    "العلان": "الإعلان",
    "المواضف": "الموظّف",
    "مهدب": "مهذّب",
    "الضفع": "الدفع",
    "بيقال": "بايبال",
    "اللعظم": "اللازم",
    "ينفعد": "ينفع أعدّل",
    "أبل ميت شهل": "قبل ما يتشحن",
    "أبل ميت شهر": "قبل ما يتشحن",
    "لخبر هك ما بسير": "الخبر هيك ما بصير",
    "يعتيكم": "يعطيكم",
    "عن جد التعامل": "عنجد التعامل",
    "وشحن وصلت": "والشحنة وصلت",
    "ما تابكون": "متى بيكون",
    "ما تبكون": "متى بيكون",
    "تبكون": "بيكون",
    "عضروري": "ضروري",
    "بدي ألغي طلب رأم واحد لاتي خمس سبعة تسعة": "بدي ألغي طلب رقم ١٣٥٧٩",
    "المنتك": "المنتج",
    "التحفة": "رائع",
    "مأبول": "مقبول",
    "مواعد": "موعد",
    "تأخر واجد": "تأخّر كثير",
    "ها كررها": "هأكرّرها",
    "بدير جعل من تج": "بدي أرجّع المنتج",
    "غيط لب رأم": "ألغي طلب رقم"
}

# ========== استخراج المواضيع ==========
TOPIC_KEYWORDS = {
    "فواتير": ["فاتورة", "دفع", "بايبال", "الدفع", "الفاتورة"],
    "توصيل": ["توصيل", "شحنة", "وصلت", "توصيل", "التوصيل", "موعد توصيل", "تاريخ توصيل"],
    "استفسار": ["استفسار", "سؤال", "استعلام"],
    "شكوى": ["شكوى", "مشكلة", "اعتراض", "خطأ", "غلط"],
    "خدمة فنية": ["فني", "تقني", "صيانة", "إصلاح"],
    "بطاقة": ["بطاقة", "كارت", "ائتمان", "مدى"],
    "تأخير": ["تأخير", "تأخرت", "متأخرة", "تأخر", "أبطأ"],
    "إلغاء طلب": ["إلغاء", "الغي", "ألغي", "تراجع", "الغاء", "إلغاء طلب"]
}

def detect_topic(text):
    if not text:
        return "غير محدد"
    
    text = text.lower()
    topic_counts = {topic: 0 for topic in TOPIC_KEYWORDS}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                topic_counts[topic] += 1
                
    if max(topic_counts.values()) == 0:
        return "أخرى"
    
    return max(topic_counts, key=topic_counts.get)

# ========== وظائف مساعدة ==========
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(
            path, 
            beam_size=3,
            vad_filter=True,
            language="ar"
        )
        return " ".join([seg.text for seg in segments])[:5000]
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

# ========== واجهة تحميل الملفات ==========
uploaded_files = st.file_uploader(
    "📂 ارفع ملفات صوتية (WAV, MP3, FLAC)", 
    type=["wav", "mp3", "flac"], 
    accept_multiple_files=True
)

# ========== الفلاتر ==========
if uploaded_files:
    st.sidebar.header("🔍 تصفية النتائج")
    
    # فلترة المشاعر
    sentiment_options = ["سلبي", "إيجابي", "محايد"]
    selected_sentiments = st.sidebar.multiselect(
        "المشاعر",
        options=sentiment_options,
        default=sentiment_options
    )
    
    # فلترة الأهمية
    rank_options = ["عالية جداً", "عالية", "متوسطة", "إيجابية", "محايدة"]
    selected_ranks = st.sidebar.multiselect(
        "مستوى الأهمية",
        options=rank_options,
        default=rank_options
    )

# ========== معالجة المكالمات ==========
if uploaded_files:
    st.info(f"🔄 جاري تحليل {len(uploaded_files)} مكالمة...")
    start_time = time.time()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        progress_percent = int((i + 1) / len(uploaded_files) * 100)
        progress_bar.progress(progress_percent)
        status_text.text(f"📞 جاري معالجة المكالمة {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            call_id = os.path.splitext(uploaded_file.name)[0]
            
            raw = transcribe_audio(tmp_path)
            
            if not raw.strip():
                results.append({
                    "call_id": call_id,
                    "error": "فشل التحويل الصوتي",
                    "text_raw": "",
                    "text_clean": "",
                    "text_corrected": "",
                    "sentiment_label": "error",
                    "sentiment_score": 0.0,
                    "rank": "Error",
                    "topic": "غير محدد"
                })
                continue
            
            clean = clean_text(raw)
            corrected = manual_correction(clean)
            topic = detect_topic(corrected)
            
            if not corrected.strip():
                sentiment = {"label": "neutral", "score": 0.0}
                st.warning(f"المكالمة {call_id}: النص فارغ بعد التنظيف")
            else:
                sentiment = sentiment_pipeline(corrected[:512])[0]
            
            label = sentiment["label"]
            score = round(sentiment["score"], 2)
            
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
                "rank": rank,
                "topic": topic
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
                "rank": "Error",
                "topic": "غير محدد"
            })
        finally:
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    progress_bar.empty()
    status_text.empty()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(uploaded_files) if uploaded_files else 0
    st.success(f"✅ اكتمل التحليل! الوقت الإجمالي: {total_time:.1f} ثانية ({avg_time:.1f} ثانية/مكالمة)")

    if results:
        df = pd.DataFrame(results)
        
        # تطبيق الفلاتر
        sentiment_map = {"سلبي": "negative", "إيجابي": "positive", "محايد": "neutral"}
        selected_labels = [sentiment_map[s] for s in selected_sentiments]
        
        filtered_df = df[
            df['sentiment_label'].isin(selected_labels) &
            df['rank'].isin(selected_ranks)
        ]
        
        # ========== المكالمات الحرجة ==========
        critical_calls = df[(df['sentiment_label'] == 'negative') & (df['rank'].isin(['عالية', 'عالية جداً']))]
        
        if not critical_calls.empty:
            st.warning(f"🚨 تنبيه: يوجد {len(critical_calls)} مكالمة سلبية عالية الأهمية تحتاج إلى متابعة عاجلة!")
            
            for _, row in critical_calls.iterrows():
                with st.expander(f"المكالمة الحرجة: {row['call_id']}", expanded=False):
                    st.markdown(f"**الموضوع:** {row['topic']}")
                    st.markdown(f"**المستوى:** {row['rank']}")
                    st.markdown(f"**نص المكالمة:**")
                    st.write(row['text_corrected'])
        
        # ========== عرض النتائج ==========
        tab1, tab2, tab3 = st.tabs(["النتائج", "الرسوم البيانية", "تحميل البيانات"])
        
        with tab1:
            st.subheader("📋 ملخص النتائج")
            st.dataframe(
                filtered_df[["call_id", "topic", "text_corrected", "sentiment_label", "sentiment_score", "rank"]], 
                use_container_width=True,
                height=400
            )
            
            st.subheader("🔍 التفاصيل الكاملة")
            for idx, row in filtered_df.iterrows():
                with st.expander(f"المكالمة: {row['call_id']} (الموضوع: {row['topic']})", expanded=False):
                    if row['sentiment_label'] == 'negative' and row['rank'] in ['عالية', 'عالية جداً']:
                        st.warning("⚠️ مكالمة سلبية عالية الأهمية - تحتاج متابعة عاجلة!")
                    
                    st.caption("النص الأصلي:")
                    st.write(row['text_raw'])
                    st.caption("النص المصحح:")
                    st.write(row['text_corrected'])
                    st.caption(f"تحليل المشاعر: {row['sentiment_label']} (ثقة: {row['sentiment_score']:.2f})")
                    st.caption(f"الأهمية: {row['rank']}")
                    st.caption(f"الموضوع: {row['topic']}")
        
        with tab2:
            st.subheader("📊 التحليل البصري")
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(
                    filtered_df, 
                    names="sentiment_label", 
                    title="توزيع المشاعر",
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                fig3 = px.bar(
                    filtered_df, 
                    x="topic", 
                    color="sentiment_label",
                    title="المواضيع حسب المشاعر",
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                )
                st.plotly_chart(fig3, use_container_width=True)
                
            with col2:
                fig2 = px.bar(
                    filtered_df, 
                    x="rank", 
                    color="sentiment_label",
                    title="مستويات الأهمية",
                    category_orders={"rank": ["عالية جداً", "عالية", "متوسطة", "إيجابية", "محايدة", "Error"]},
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                fig4 = px.treemap(
                    filtered_df, 
                    path=['topic', 'sentiment_label'], 
                    values='sentiment_score',
                    title="توزيع المواضيع والمشاعر",
                    color='sentiment_label',
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            st.subheader("⬇️ تحميل النتائج")
            
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
                    filtered_df.to_csv(index=False).encode("utf-8-sig"), 
                    file_name="call_results.csv", 
                    mime="text/csv"
                )
            
            st.caption("معاينة JSON:")
            st.json(results[0] if len(results) > 0 else {})
