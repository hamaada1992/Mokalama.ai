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
import concurrent.futures

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
    
    /* إصلاح شامل للون الخط في الجداول */
    .stDataFrame * {
        color: black !important;
    }
    
    .stDataFrame th, .stDataFrame td {
        color: black !important;
        background-color: transparent !important;
    }
    
    .stDataFrame table {
        color: black !important;
        font-family: Arial, sans-serif !important;
        font-size: 14px !important;
    }
    
    .stDataFrame th {
        font-weight: bold !important;
        background-color: #f0f0f0 !important;
    }
    
    .positive-row {
        background-color: #d4f8e8 !important;
        color: black !important;
    }
    
    .neutral-row {
        background-color: #fff9db !important;
        color: black !important;
    }
    
    .negative-row {
        background-color: #ffdbdb !important;
        color: black !important;
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

# جميع أحجام النماذج الأساسية فقط
MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]

# وصف النماذج
MODEL_DESCRIPTIONS = {
    "tiny": "الأسرع - دقة منخفضة (40-50%)",
    "base": "متوسط السرعة - دقة معقولة (60-70%)",
    "small": "جيد - توازن بين السرعة والدقة (70-80%)",
    "medium": "متقدم - دقة عالية (80-90%)",
    "large": "الأفضل - أعلى دقة (90%+)"
}

# اختيار حجم النموذج
model_size = st.sidebar.selectbox(
    "حجم نموذج الصوت",
    options=MODEL_SIZES,
    index=1,  # تحديد "base" كإفتراضي
    format_func=lambda x: f"{x} - {MODEL_DESCRIPTIONS.get(x, '')}",
    help="اختر نموذجاً مناسباً. النماذج الأكبر حجماً أكثر دقة ولكنها أبطأ"
)

# ========== تحميل النماذج ==========
@st.cache_resource(show_spinner=False)
def load_whisper_model(size):
    st.info(f"⏳ جاري تحميل نموذج الصوت ({size})...")
    try:
        return WhisperModel(size, device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"❌ فشل تحميل نموذج الصوت: {str(e)}")
        st.stop()

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
            max_length=128
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
    "غيط لب رأم": "ألغي طلب رقم",
    "أصوى": "أسوء",
    "أخذ متعملاء": "خدمة عملاء",
    "الزباء": "الزبائن",
    "خدمت":"خدمة",
    "مرحضة":"مرحبا"
}

# ========== استخراج المواضيع (موسع) ==========
TOPIC_KEYWORDS = {
    "فواتير": ["فاتورة", "دفع", "بايبال", "الدفع", "الفاتورة", "مبلغ", "رسوم"],
    "توصيل": ["توصيل", "شحنة", "وصلت", "توصيل", "التوصيل", "موعد توصيل", "تاريخ توصيل", "تسليم", "الشحن"],
    "استفسار": ["استفسار", "سؤال", "استعلام", "استفسار", "استعلام"],
    "شكوى": ["شكوى", "مشكلة", "اعتراض", "خطأ", "غلط", "مستاء", "غير راضي"],
    "خدمة فنية": ["فني", "تقني", "صيانة", "إصلاح", "عطل", "أعطال"],
    "بطاقة": ["بطاقة", "كارت", "ائتمان", "مدى", "بطاقة ائتمان"],
    "تأخير": ["تأخير", "تأخرت", "متأخرة", "تأخر", "أبطأ", "تأجيل"],
    "إلغاء طلب": ["إلغاء", "الغي", "ألغي", "تراجع", "الغاء", "إلغاء طلب", "ألغيت"],
    "استبدال منتج": ["استبدال", "بديل", "أبدل", "تغيير", "استبداله", "بدي أبدل"],
    "استرجاع منتج": ["استرجاع", "أرجع", "إرجاع", "أسترد", "رد المنتج"],
    "خصومات وعروض": ["عرض", "خصم", "تخفيض", "عروض", "تنزيلات", "سعر مخفض"],
    "جودة المنتج": ["جودة", "نوعية", "رديئة", "رديء", "ممتازة", "سيئة"],
    "خدمة العملاء": ["خدمة العملاء", "خدمة الزبائن", "دعم العملاء", "موظف خدمة"],
    "تحديث معلومات": ["تحديث", "معلومات", "عنوان", "رقم الجوال", "بيانات", "تعديل بيانات"],
    "المنتجات": ["منتج", "سلعة", "بضاعة", "صنف", "المشتريات", "الطلب"],
    "تتبع الطلبات": ["تتبع", "أين طلبي", "مكان الشحنة", "موقع الشحنة", "تتبع شحنة"],
    "الضمان": ["ضمان", "كفالة", "صلاحية الضمان", "فترة الضمان"]
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
    if not text:
        return ""
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    if not text:
        return ""
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(
            path, 
            beam_size=1,
            vad_filter=True,
            language="ar",
            without_timestamps=True
        )
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت إلى نص: {str(e)}")
        return ""

# ========== معالجة مكالمة واحدة ==========
def process_call(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        call_id = os.path.splitext(uploaded_file.name)[0]
        
        raw = transcribe_audio(tmp_path)
        
        if not raw.strip():
            return {
                "call_id": call_id,
                "error": "فشل التحويل الصوتي",
                "text_raw": "",
                "text_clean": "",
                "text_corrected": "",
                "sentiment_label": "error",
                "sentiment_score": 0.0,
                "rank": "Error",
                "topic": "غير محدد"
            }
        
        clean = clean_text(raw)
        corrected = manual_correction(clean)
        topic = detect_topic(corrected)
        
        if not corrected.strip():
            final_label = "neutral"
            final_score = 0.0
        else:
            # تقسيم النص إلى أجزاء صغيرة لتجنب أخطاء طول السياق
            max_chunk_size = 128
            chunks = [corrected[i:i+max_chunk_size] for i in range(0, len(corrected), max_chunk_size)]
            sentiments = sentiment_pipeline(chunks)
            
            # حساب متوسط النتائج
            scores = []
            labels = []
            for s in sentiments:
                labels.append(s['label'])
                scores.append(s['score'])
            
            # تحديد التصنيف النهائي بناءً على المتوسط
            label_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for l in labels:
                label_counts[l] += 1
            
            final_label = max(label_counts, key=label_counts.get)
            final_score = round(sum(scores) / len(scores), 2)
        
        if final_label == "negative":
            if final_score > 0.85:
                rank = "عالية جداً"
            elif final_score > 0.7:
                rank = "عالية"
            else:
                rank = "متوسطة"
        elif final_label == "positive":
            rank = "إيجابية"
        else:
            rank = "محايدة"

        return {
            "call_id": call_id,
            "text_raw": raw,
            "text_clean": clean,
            "text_corrected": corrected,
            "sentiment_label": final_label,
            "sentiment_score": final_score,
            "rank": rank,
            "topic": topic
        }
        
    except Exception as e:
        return {
            "call_id": uploaded_file.name,
            "error": str(e),
            "text_raw": "",
            "text_clean": "",
            "text_corrected": "",
            "sentiment_label": "error",
            "sentiment_score": 0.0,
            "rank": "Error",
            "topic": "غير محدد"
        }
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass

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
    
    # فلترة المواضيع
    topic_options = sorted(set(TOPIC_KEYWORDS.keys()) | {"أخرى", "غير محدد"})
    selected_topics = st.sidebar.multiselect(
        "المواضيع",
        options=topic_options,
        default=topic_options
    )

# ========== معالجة المكالمات ==========
if uploaded_files:
    st.info(f"🔄 جاري تحليل {len(uploaded_files)} مكالمة...")
    start_time = time.time()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # معالجة متوازية لتحسين السرعة
    max_workers = min(4, len(uploaded_files))  # لا تزيد عن 4 عمال
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_call, file): file for file in uploaded_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                st.error(f"❌ خطأ في معالجة المكالمة {file.name}: {str(e)}")
            
            # تحديث شريط التقدم
            progress_percent = int((i + 1) / len(uploaded_files) * 100)
            progress_bar.progress(progress_percent)
            status_text.text(f"📞 تم معالجة {i+1}/{len(uploaded_files)} مكالمة")

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
            df['rank'].isin(selected_ranks) &
            df['topic'].isin(selected_topics)
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
        if filtered_df.empty:
            st.warning("⚠️ لا توجد نتائج تطابق الفلاتر المحددة")
        else:
            tab1, tab2, tab3 = st.tabs(["النتائج", "الرسوم البيانية", "تحميل البيانات"])
            
            with tab1:
                st.subheader("📋 ملخص النتائج")
                
                # تلوين الصفوف حسب المشاعر
                def color_row(row):
                    styles = [''] * len(row)
                    if row['sentiment_label'] == 'positive':
                        styles = ['background-color: #d4f8e8;'] * len(row)
                    elif row['sentiment_label'] == 'neutral':
                        styles = ['background-color: #fff9db;'] * len(row)
                    elif row['sentiment_label'] == 'negative':
                        styles = ['background-color: #ffdbdb;'] * len(row)
                    return styles
                
                # تطبيق التلوين على DataFrame
                display_df = filtered_df[["call_id", "topic", "sentiment_label", "sentiment_score", "rank"]].copy()
                styled_df = display_df.style.apply(color_row, axis=1)
                
                # عرض الجدول مع تخصيصات إضافية
                st.dataframe(
                    styled_df,
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
                    # توزيع المشاعر
                    sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
                    sentiment_counts.columns = ['sentiment', 'count']
                    fig1 = px.pie(
                        sentiment_counts, 
                        names='sentiment', 
                        values='count',
                        title="توزيع المشاعر",
                        color='sentiment',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # المواضيع حسب المشاعر
                    if not filtered_df.empty:
                        fig3 = px.bar(
                            filtered_df, 
                            x="topic", 
                            color="sentiment_label",
                            title="المواضيع حسب المشاعر",
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    
                with col2:
                    # مستويات الأهمية
                    if not filtered_df.empty:
                        fig2 = px.bar(
                            filtered_df, 
                            x="rank", 
                            color="sentiment_label",
                            title="مستويات الأهمية",
                            category_orders={"rank": ["عالية جداً", "عالية", "متوسطة", "إيجابية", "محايدة"]},
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # توزيع المواضيع والمشاعر
                    if not filtered_df.empty:
                        fig4 = px.treemap(
                            filtered_df, 
                            path=['topic', 'sentiment_label'], 
                            values='sentiment_score',
                            title="توزيع المواضيع والمشاعر",
                            color='sentiment_label',
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
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
