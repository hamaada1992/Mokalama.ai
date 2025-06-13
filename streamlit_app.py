import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
import re
import base64
import numpy as np
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

# تحميل بيانات nltk المطلوبة
nltk.download('punkt')

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام تحليل مكالمات الدعم الفني",
    layout="wide",
    page_icon="📞",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stDownloadButton>button {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    }
    .stRadio>div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 15px;
    }
    .stFileUploader>div {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 20px;
    }
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    .stAlert {
        border-radius: 12px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.title("📞 نظام تحليل مكالمات الدعم الفني المتقدم")
st.markdown("""
<div style="text-align: right; margin-bottom: 30px;">
    <p style="font-size: 18px; color: #555;">
        نظام متكامل لتحليل مكالمات العملاء باستخدام الذكاء الاصطناعي للتعرف على المشاعر، 
        استخراج الكيانات المهمة، وتحديد أولويات المكالمات.
    </p>
</div>
""", unsafe_allow_html=True)

# شريط جانبي للإعدادات
with st.sidebar:
    st.header("⚙️ إعدادات النظام")
    
    # طريقة تحليل المشاعر
    analysis_method = st.radio(
        "طريقة تحليل المشاعر:",
        ["النص الكامل (أسرع)", "تقسيم الجمل (أدق)"],
        index=0
    )
    
    # تفعيل NER
    enable_ner = st.checkbox("تفعيل تحليل الكيانات المسماة (NER)", value=True)
    
    st.divider()
    
    # تحميل ملف التصحيحات
    st.subheader("🔄 تحديث قاموس التصحيح")
    st.markdown("""
    <div style="text-align: right; font-size: 14px; color: #666; margin-bottom: 10px;">
        يمكنك رفع ملف JSON لتحديث قاموس التصحيح التلقائي للنصوص.
        الملف يجب أن يحتوي على أزواج من الكلمات الخاطئة والتصحيح.
    </div>
    """, unsafe_allow_html=True)
    
    corrections_file = st.file_uploader(
        "رفع ملف تصحيحات (JSON)",
        type=["json"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

# تخزين النماذج
@st.cache_resource(show_spinner="جارٍ تحميل نموذج تحويل الصوت إلى نص...")
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

@st.cache_resource(show_spinner="جارٍ تحميل نموذج تحليل المشاعر...")
def load_sentiment_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner="جارٍ تحميل نموذج التعرف على الكيانات...")
def load_ner_model():
    return pipeline("ner", model="hatmimoha/arabic-ner")

# تحميل النماذج
with st.spinner("⏳ جارٍ تحميل النماذج، الرجاء الانتظار..."):
    whisper_model = load_whisper_model()
    sentiment_pipeline = load_sentiment_model()
    if enable_ner:
        ner_pipeline = load_ner_model()
    else:
        ner_pipeline = None

# قاموس التصحيح الافتراضي
default_corrections = {
    "الفتور": "الفاتورة", "زياد": "زيادة", "الليزوم": "اللزوم", "المصادة": "المساعدة",
    "بدي بطل": "بدي أبدل", "مع بول": "مع بوليصة", "تازي": "تازة", "ادام الفني": "أداء الفني",
    "اخذ وقت اكثر من اللّعظم": "أخذ وقت أكثر من اللازم", "اللعظم": "اللازم", "مش زي ما مكتوب": "مش زي ما هو مكتوب",
    "بأفين": "بقى فين", "فين": "أين", "واللسه": "ولسه", "تجربتي معاكم كانت متس": "تجربتي معاكم كانت ممتازة",
    "هكررها": "سأكررها", "تانيا": "ثانية", "ينفعاد": "ينفع أعدّل", "ينفع اد": "ينفع أعدّل", "أبل": "قبل",
    "ما حده": "ما حدا", "لخبر هك": "الخبر هيك", "بسير": "بصير", "يعتيكم": "يعطيكم", "عافي": "العافية",
    "تأخر واجد": "تأخر كثير", "واجد": "كثير", "ضروري": "بشكل عاجل",
    "لو سمحت متى بيكون التوصيل للرياض بالعادة": "متى يوصل الطلب للرياض عادة؟",
    "يوماين": "يومين", "ما تبكون": "ما تكونون", "عضروري": "ضروري",
    "مافيني": "ما فيني", "شكرن": "شكراً", "مشان": "بسبب", "عمي": "عملي", "عليك": "عليكم",
    "شكرا": "شكراً", "عفوا": "عفواً", "بس": "لكن", "شو": "ما هو", "هيدا": "هذا", 
    "مشكور": "مشكورين", "يعطيك": "يعطيكم", "الله": "الله", "يسعد": "يسعدكم",
    "الخدمة": "خدمة العملاء", "بدي": "أريد", "عندي": "لدي", "مش": "ليس", 
    "عامل": "يعمل", "مشكلة": "مشكلة", "مافهمت": "لم أفهم", "وين": "أين",
    "بدي اتكلم": "أريد التحدث", "مع مدير": "مع المدير", "ما رديت": "لم تردوا"
}

corrections = default_corrections
if corrections_file:
    try:
        custom_corrections = json.load(corrections_file)
        corrections.update(custom_corrections)
        st.sidebar.success("✅ تم تحديث قاموس التصحيح بنجاح!")
    except:
        st.sidebar.error("❌ خطأ في تحميل ملف التصحيحات. استخدام القاموس الافتراضي.")

# وظائف المعالجة
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def manual_correction(text):
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def transcribe_audio(path):
    segments, _ = whisper_model.transcribe(path)
    return " ".join([seg.text for seg in segments])

def analyze_sentiment(text):
    if not text.strip() or len(text.split()) < 3:
        return {"label": "neutral", "score": 0.5}
    
    try:
        if analysis_method == "النص الكامل (أسرع)":
            return sentiment_pipeline(text)[0]
        else:
            sentences = sent_tokenize(text)
            results = sentiment_pipeline(sentences)
            
            # حساب متوسط النتائج
            sentiments = [1 if res['label'] == 'positive' else -1 if res['label'] == 'negative' else 0 for res in results]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # تحديد التصنيف النهائي
            if avg_sentiment > 0.2:
                return {"label": "positive", "score": avg_sentiment}
            elif avg_sentiment < -0.2:
                return {"label": "negative", "score": abs(avg_sentiment)}
            else:
                return {"label": "neutral", "score": 0.5}
    except:
        return {"label": "neutral", "score": 0.5}

def extract_entities(text):
    if not enable_ner or not text.strip():
        return []
    
    try:
        entities = ner_pipeline(text)
        # تجميع الكيانات المتجاورة
        merged_entities = []
        current_entity = ""
        current_label = ""
        
        for entity in entities:
            if entity['word'].startswith("##"):
                current_entity += entity['word'][2:]
            else:
                if current_entity:
                    merged_entities.append((current_entity, current_label))
                current_entity = entity['word']
                current_label = entity['entity']
        
        if current_entity:
            merged_entities.append((current_entity, current_label))
            
        return merged_entities
    except:
        return []

# واجهة تحميل الملفات
st.header("📂 رفع ملفات المكالمات الصوتية")
uploaded_files = st.file_uploader(
    "اختر ملفات صوتية (MP3, WAV, FLAC) أو اسحبها وألقها هنا",
    type=["wav", "mp3", "flac"], 
    accept_multiple_files=True,
    help="يمكنك رفع عدة ملفات صوتية في نفس الوقت"
)

if uploaded_files:
    # التحقق من عدد الملفات
    if len(uploaded_files) > 5:
        st.warning("لتحسين الأداء، سيتم معالجة أول 5 ملفات فقط.")
        uploaded_files = uploaded_files[:5]
    
    st.success(f"تم رفع {len(uploaded_files)} ملف صوتي بنجاح!")
    
    with st.expander("عرض الملفات المرفوعة"):
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size//1024} KB)")
    
    if st.button("بدء التحليل", use_container_width=True):
        st.info(f"🔄 جاري معالجة {len(uploaded_files)} مكالمة...")
        results = []
        audio_files = {}

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"جارٍ معالجة الملف {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            call_id = os.path.splitext(uploaded_file.name)[0]
            raw_text = transcribe_audio(tmp_path)
            clean_text_val = clean_text(raw_text)
            corrected_text = manual_correction(clean_text_val)
            
            # تحليل المشاعر
            sentiment = analyze_sentiment(corrected_text)
            label = sentiment["label"]
            score = round(sentiment["score"], 2)
            rank = "High" if label == "negative" and score > 0.8 else "Medium" if label == "negative" else "Low"
            
            # استخراج الكيانات
            entities = extract_entities(corrected_text)
            
            # تخزين بيانات الصوت
            audio_files[call_id] = base64.b64encode(uploaded_file.read()).decode('utf-8')
            uploaded_file.seek(0)  # إعادة تعيين المؤشر
            
            results.append({
                "call_id": call_id,
                "text_raw": raw_text,
                "text_clean": clean_text_val,
                "text_corrected": corrected_text,
                "sentiment_label": label,
                "sentiment_score": score,
                "rank": rank,
                "entities": entities
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            os.unlink(tmp_path)  # حذف الملف المؤقت

        status_text.text("✅ اكتملت معالجة جميع الملفات!")
        st.balloons()
        
        df = pd.DataFrame(results)
        
        # إنشاء علامات التبويب
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 النتائج التفصيلية", 
            "📊 التحليل الإحصائي", 
            "🏷️ الكيانات المسماة",
            "📝 تقرير متكامل"
        ])
        
        with tab1:
            st.subheader("نتائج تحليل المكالمات")
            st.info("تحتوي هذه النتائج على النص المحول والمصحح مع تحليل المشاعر وتصنيف الأولوية")
            
            display_df = df[["call_id", "text_corrected", "sentiment_label", "sentiment_score", "rank"]].copy()
            display_df["sentiment_score"] = display_df["sentiment_score"].apply(lambda x: f"{x:.2f}")
            
            # إضافة أزرار الاستماع
            display_df["استماع"] = display_df["call_id"].apply(
                lambda x: f'<audio controls src="data:audio/wav;base64,{audio_files[x]}" style="height:30px; width:100%;"></audio>'
            )
            
            # عرض الجدول
            st.markdown(
                display_df.to_html(escape=False, index=False), 
                unsafe_allow_html=True
            )
        
        with tab2:
            st.subheader("التحليل الإحصائي")
            st.info("تصورات بيانية لتوزيع المشاعر وتصنيف الأولوية للمكالمات")
            
            col1, col2 = st.columns(2)
            with col1:
                # تخصيص الألوان في الرسوم البيانية
                color_map = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}
                fig1 = px.pie(
                    df, 
                    names="sentiment_label", 
                    title="توزيع المشاعر",
                    color="sentiment_label",
                    color_discrete_map=color_map,
                    hole=0.4
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                rank_order = {"High": 3, "Medium": 2, "Low": 1}
                df["rank_order"] = df["rank"].map(rank_order)
                df = df.sort_values("rank_order")
                
                fig2 = px.bar(
                    df, 
                    x="call_id", 
                    y="sentiment_score", 
                    color="rank",
                    title="تقييم أولوية المكالمات",
                    color_discrete_map={"High": "#F44336", "Medium": "#FF9800", "Low": "#4CAF50"},
                    category_orders={"rank": ["High", "Medium", "Low"]}
                )
                fig2.update_layout(
                    xaxis_title="رقم المكالمة", 
                    yaxis_title="درجة المشاعر",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # مخطط المشاعر الزمني
            st.subheader("تحليل المشاعر الزمني")
            fig3 = px.line(
                df, 
                x="call_id", 
                y="sentiment_score",
                color="sentiment_label",
                markers=True,
                title="تغير المشاعر بين المكالمات",
                color_discrete_map=color_map
            )
            fig3.update_layout(
                xaxis_title="رقم المكالمة",
                yaxis_title="درجة المشاعر",
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            if not enable_ner:
                st.warning("⚠️ تفعيل تحليل الكيانات المسماة من الإعدادات الجانبية")
            else:
                st.subheader("الكيانات المسماة المستخرجة")
                st.info("الكيانات المهمة التي تم التعرف عليها في المكالمات مثل الأسماء والأماكن والمواضيع")
                
                all_entities = []
                for _, row in df.iterrows():
                    if row["entities"]:
                        for entity, label in row["entities"]:
                            all_entities.append({
                                "المكالمة": row["call_id"],
                                "الكيان": entity,
                                "التصنيف": label
                            })
                
                if all_entities:
                    entities_df = pd.DataFrame(all_entities)
                    
                    # تحليل توزيع الكيانات
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("توزيع الكيانات")
                        fig4 = px.bar(
                            entities_df["التصنيف"].value_counts().reset_index(),
                            x="التصنيف",
                            y="count",
                            labels={"التصنيف": "نوع الكيان", "count": "عدد التكرارات"},
                            color="التصنيف",
                            height=400
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    
                    with col2:
                        st.subheader("الكيانات الأكثر تكراراً")
                        top_entities = entities_df["الكيان"].value_counts().head(10).reset_index()
                        fig5 = px.bar(
                            top_entities,
                            x="الكيان",
                            y="count",
                            labels={"الكيان": "اسم الكيان", "count": "عدد التكرارات"},
                            color="الكيان",
                            height=400
                        )
                        st.plotly_chart(fig5, use_container_width=True)
                    
                    # عرض الجدول
                    st.subheader("جميع الكيانات المستخرجة")
                    st.dataframe(entities_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("لم يتم العثور على كيانات مسماة في المكالمات.")
        
        with tab4:
            st.subheader("تقرير تحليل المكالمات المتكامل")
            st.info("ملخص شامل لنتائج التحليل مع التوصيات")
            
            # حساب الإحصائيات
            total_calls = len(df)
            negative_calls = len(df[df["sentiment_label"] == "negative"])
            positive_calls = len(df[df["sentiment_label"] == "positive"])
            high_priority = len(df[df["rank"] == "High"])
            
            # عرض الإحصائيات
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي المكالمات", total_calls)
            col2.metric("المكالمات السلبية", negative_calls, f"{round(negative_calls/total_calls*100)}%")
            col3.metric("مكالمات عالية الأولوية", high_priority)
            
            # تحليل النتائج
            st.subheader("تحليل النتائج")
            if negative_calls > 0:
                st.warning(f"**ملاحظة مهمة:** يوجد {negative_calls} مكالمة سلبية ({round(negative_calls/total_calls*100)}%) تحتاج إلى متابعة فورية.")
            else:
                st.success("**أخبار جيدة:** لا توجد مكالمات سلبية في مجموعة البيانات.")
            
            # عرض المكالمات عالية الأولوية
            if high_priority > 0:
                st.subheader("المكالمات عالية الأولوية")
                st.warning("هذه المكالمات تحتاج إلى متابعة فورية بسبب مشاعر سلبية قوية")
                
                high_priority_df = df[df["rank"] == "High"]
                for _, row in high_priority_df.iterrows():
                    with st.expander(f"مكالمة عاجلة: {row['call_id']} (درجة: {row['sentiment_score']:.2f})", expanded=False):
                        st.caption("**النص المحول:**")
                        st.write(row["text_corrected"])
                        
                        st.caption("**الكيانات المهمة:**")
                        if row["entities"]:
                            entities_list = [f"{entity} ({label})" for entity, label in row["entities"]]
                            st.write(", ".join(entities_list))
                        else:
                            st.write("لم يتم العثور على كيانات مهمة.")
                        
                        st.audio(base64.b64decode(audio_files[row["call_id"]]), format="audio/wav")
            
            # التوصيات
            st.subheader("التوصيات")
            if negative_calls > 0:
                st.markdown("""
                - **متابعة فورية** للمكالمات عالية الأولوية في غضون 24 ساعة
                - **تحليل أسباب** المكالمات السلبية وتحديد أنماط المشاكل المتكررة
                - **تدريب فريق الدعم** على التعامل مع الحالات السلبية
                - **تقديم تعويضات** للعملاء المتضررين في الحالات الشديدة
                """)
            else:
                st.markdown("""
                - **مواصلة التميز** في خدمة العملاء
                - **تحليل المكالمات الإيجابية** لتحديد أفضل الممارسات
                - **مكافأة فريق الدعم** على الأداء المتميز
                """)
            
            # تنزيل التقارير
            st.divider()
            st.subheader("تصدير النتائج")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 تحميل التقرير الكامل (JSON)",
                    json.dumps(results, ensure_ascii=False, indent=2),
                    file_name="call_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📥 تحميل التقرير (CSV)",
                    df.drop(columns=["entities", "text_raw", "text_clean"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name="call_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# رسالة عند عدم رفع ملفات
else:
    st.info("📁 يرجى رفع ملفات صوتية لبدء التحليل")
    
    # معلومات عن كيفية الاستخدام
    with st.expander("🎯 دليل استخدام التطبيق", expanded=True):
        st.markdown("""
        ### كيفية استخدام نظام تحليل المكالمات:
        1. **رفع الملفات الصوتية**: 
            - استخدم زر الرفع أعلاه لاختيار ملفاتك الصوتية (MP3, WAV, FLAC)
            - يمكنك رفع عدة ملفات مرة واحدة (حد أقصى 5 ملفات)
        
        2. **ضبط الإعدادات**:
            - اختيار طريقة تحليل المشاعر (النص الكامل أسرع، تقسيم الجمل أدق)
            - تفعيل/تعطيل تحليل الكيانات المسماة (NER)
            - تحديث قاموس التصحيح من خلال رفع ملف JSON
        
        3. **بدء التحليل**:
            - انقر على زر "بدء التحليل" لمعالجة الملفات
            - انتظر حتى اكتمال المعالجة (قد تستغرق دقيقة لكل مكالمة)
        
        4. **استعراض النتائج**:
            - النتائج التفصيلية لكل مكالمة مع إمكانية الاستماع
            - تحليل إحصائي وبياني لتوزيع المشاعر والأولوية
            - تقرير متكامل مع التوصيات
        
        5. **تصدير النتائج**:
            - حفظ التقرير بصيغة JSON أو CSV لاستخدامها لاحقاً
        """)
    
    # أمثلة على المخرجات
    st.subheader("📊 معاينة لنتائج التحليل")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.ibb.co/5LhRqG6/pie-chart.png", caption="توزيع مشاعر المكالمات")
    with col2:
        st.image("https://i.ibb.co/4W5yYb7/bar-chart.png", caption="تصنيف أولوية المكالمات")
    
    st.image("https://i.ibb.co/4dL5J0y/line-chart.png", caption="تحليل المشاعر الزمني", width=700)

# تذييل الصفحة
st.divider()
st.markdown("""
<div style="text-align: center; color: #777; font-size: 14px; margin-top: 30px;">
    نظام تحليل مكالمات الدعم الفني المتقدم | الإصدار 2.1 | تم التطوير باستخدام الذكاء الاصطناعي
</div>
""", unsafe_allow_html=True)
