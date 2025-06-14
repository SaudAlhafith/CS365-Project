#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import time
import plotly.express as px
import pandas as pd
from .model_utils import load_summarization_models

def render_summarization_tab():
    st.header("📄 Arabic Text Summarization")
    st.markdown("**Compare Seq2Seq LSTM vs AraBART** for Arabic text summarization")
    
    # Load summarization models
    with st.spinner("Loading summarization models..."):
        seq2seq, arabart = load_summarization_models()
    
    # Check model availability
    seq2seq_available = seq2seq is not None and seq2seq.is_available()
    arabart_available = arabart is not None and arabart.is_available()
    
    if not seq2seq_available and not arabart_available:
        st.error("No summarization models available. Please ensure model files are present.")
        return
    
    # Model selection
    available_models = []
    if seq2seq_available:
        available_models.append("Seq2Seq LSTM")
    if arabart_available:
        available_models.append("AraBART")
    
    selected_models = st.multiselect(
        "Select Summarization Models:",
        available_models,
        default=available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Input methods
    input_method = st.radio(
        "Input Method:",
        ["Type Text", "Upload File", "Sample Texts"]
    )
    
    input_text = ""
    
    if input_method == "Type Text":
        input_text = st.text_area(
            "Enter Arabic text to summarize:",
            height=200,
            placeholder="اكتب النص العربي هنا للحصول على ملخص..."
        )
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload text file (.txt)",
            type=['txt']
        )
        
        if uploaded_file is not None:
            try:
                input_text = str(uploaded_file.read(), "utf-8")
                st.success(f"File uploaded successfully! ({len(input_text)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif input_method == "Sample Texts":
        sample_texts = {
            "Technology News": """
تشهد التكنولوجيا في العالم العربي تطورا سريعا وملحوظا في السنوات الأخيرة، حيث تزايد الاستثمار في مجال الذكاء الاصطناعي والتعلم الآلي. 
وقد أطلقت العديد من الدول العربية مبادرات طموحة لتطوير القطاع التقني، بما في ذلك إنشاء مدن تقنية متخصصة ومراكز للابتكار. 
كما شهدت المنطقة نموا كبيرا في عدد الشركات الناشئة التقنية، والتي تعمل على تطوير حلول مبتكرة في مختلف المجالات مثل التجارة الإلكترونية والصحة الرقمية والتعليم الإلكتروني.
هذا التطور السريع يعكس التزام المنطقة بمواكبة التقدم التكنولوجي العالمي والاستفادة من الفرص الهائلة التي توفرها التقنيات الحديثة لتحسين جودة الحياة وتعزيز النمو الاقتصادي.
            """,
            "Education Article": """
يشهد قطاع التعليم في المنطقة العربية تحولا جذريا نحو التعليم الرقمي والتعلم عن بُعد، خاصة بعد جائحة كوفيد-19 التي أجبرت المؤسسات التعليمية على تبني التقنيات الحديثة.
وقد استثمرت الحكومات العربية مليارات الدولارات في تطوير البنية التحتية الرقمية للتعليم، بما في ذلك توفير الأجهزة اللوحية للطلاب وتدريب المعلمين على استخدام التقنيات الحديثة.
كما برزت منصات التعلم الإلكتروني العربية كبديل فعال للتعليم التقليدي، حيث تقدم محتوى تعليميا عالي الجودة باللغة العربية يغطي مختلف المراحل الدراسية والتخصصات.
هذا التحول الرقمي في التعليم يفتح آفاقا جديدة أمام الطلاب العرب للوصول إلى تعليم عالي الجودة بغض النظر عن موقعهم الجغرافي أو ظروفهم الاقتصادية.
            """,
            "Health News": """
تواجه الأنظمة الصحية في العالم العربي تحديات كبيرة تتطلب حلولا مبتكرة وشاملة لتحسين جودة الخدمات الطبية المقدمة للمواطنين.
وتتضمن هذه التحديات نقص في الكوادر الطبية المتخصصة، وعدم توازن في توزيع الخدمات الصحية بين المناطق الحضرية والريفية، بالإضافة إلى محدودية الموارد المالية المخصصة للقطاع الصحي.
لمواجهة هذه التحديات، تتجه العديد من الدول العربية نحو تبني تقنيات الصحة الرقمية مثل التطبيب عن بُعد والذكاء الاصطناعي في التشخيص الطبي، والتي تساعد على تحسين الوصول إلى الخدمات الصحية وتقليل التكاليف.
كما تعمل هذه الدول على تطوير برامج التأمين الصحي الشامل وتحسين البنية التحتية للمستشفيات والمراكز الصحية لضمان تقديم رعاية صحية عالية الجودة لجميع المواطنين.
            """
        }
        
        selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
        if selected_sample:
            input_text = sample_texts[selected_sample]
            st.text_area("Selected text:", value=input_text, height=150, disabled=True)
    
    # Summarization parameters
    with st.expander("⚙️ Summarization Settings"):
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Maximum summary length:", 50, 200, 128)
            min_length = st.slider("Minimum summary length:", 10, 100, 30)
        with col2:
            num_beams = st.slider("Number of beams (for AraBART):", 1, 8, 4)
            show_confidence = st.checkbox("Show confidence scores", True)
    
    # Generate summaries
    if st.button("🔍 Generate Summaries", type="primary", key="generate_summaries_btn") and input_text.strip():
        if len(input_text.strip()) < 50:
            st.warning("Please enter a longer text (at least 50 characters) for better summarization.")
            return
        
        results = {}
        timing_data = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(selected_models)
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Generating summary with {model_name}...")
            
            start_time = time.time()
            
            try:
                if model_name == "Seq2Seq LSTM" and seq2seq_available:
                    if show_confidence:
                        # For Seq2Seq, we don't have confidence, so just generate summary
                        summary = seq2seq.generate_summary(input_text, max_length=max_length)
                        confidence = 0.0  # Placeholder
                    else:
                        summary = seq2seq.generate_summary(input_text, max_length=max_length)
                        confidence = 0.0
                
                elif model_name == "AraBART" and arabart_available:
                    if show_confidence:
                        summary, confidence = arabart.generate_summary_with_confidence(
                            input_text, max_length=max_length, min_length=min_length, num_beams=num_beams
                        )
                    else:
                        summary = arabart.generate_summary(
                            input_text, max_length=max_length, min_length=min_length, num_beams=num_beams
                        )
                        confidence = 0.0
                
                else:
                    continue
                
                inference_time = time.time() - start_time
                
                results[model_name] = {
                    'summary': summary,
                    'confidence': confidence,
                    'time': inference_time,
                    'length': len(summary.split()) if summary else 0
                }
                
                timing_data.append({
                    'Model': model_name,
                    'Time (s)': inference_time
                })
            
            except Exception as e:
                results[model_name] = {
                    'summary': f"Error: {str(e)}",
                    'confidence': 0.0,
                    'time': 0.0,
                    'length': 0
                }
            
            progress_bar.progress((i + 1) / total_models)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.subheader("📝 Summarization Results:")
        
        for model_name, result in results.items():
            with st.expander(f"📄 {model_name} Summary", expanded=True):
                st.markdown(f"**Summary ({result['length']} words):**")
                st.markdown(f"<div class='arabic-text'>{result['summary']}</div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inference Time", f"{result['time']:.2f}s")
                with col2:
                    st.metric("Word Count", result['length'])
                with col3:
                    if show_confidence and result['confidence'] > 0:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    else:
                        st.metric("Confidence", "N/A")
        
        # Performance comparison
        if len(results) > 1:
            st.subheader("⚡ Performance Comparison:")
            
            # Timing chart
            if timing_data:
                fig_timing = px.bar(
                    timing_data,
                    x='Model',
                    y='Time (s)',
                    title="Inference Time Comparison",
                    color='Time (s)',
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig_timing, use_container_width=True)
            
            # Summary comparison table
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Word Count': result['length'],
                    'Time (s)': f"{result['time']:.2f}",
                    'Confidence': f"{result['confidence']:.2f}" if result['confidence'] > 0 else "N/A"
                })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
    
    elif st.button("🔍 Generate Summaries", type="primary", key="generate_summaries_empty_btn"):
        st.warning("Please enter some text to summarize.")

def main():
    render_summarization_tab()

if __name__ == "__main__":
    main() 