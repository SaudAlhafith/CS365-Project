#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
from .model_utils import predict_with_timing, SAMPLE_TEXTS

def render_classification_tab(traditional, bilstm, arabert, selected_models):
    st.header("Text Classification")
    
    st.subheader("📝 Select a sample or enter your own text:")
    sample_choice = st.selectbox("Choose a sample text:", ["Custom"] + list(SAMPLE_TEXTS.keys()))
    
    if sample_choice == "Custom":
        user_text = st.text_area(
            "Enter Arabic text:",
            placeholder="أدخل النص العربي هنا...",
            height=100,
            help="Enter any Arabic text for classification"
        )
    else:
        user_text = st.text_area(
            "Enter Arabic text:",
            value=SAMPLE_TEXTS[sample_choice],
            height=100
        )
    
    if st.button("🚀 Classify Text", type="primary") and user_text.strip():
        st.markdown("---")
        
        st.subheader("📄 Input Text:")
        st.markdown(f'<div class="arabic-text">{user_text}</div>', unsafe_allow_html=True)
        
        st.subheader("🎯 Predictions:")
        
        cols = st.columns(len(selected_models))
        results = {}
        
        with st.spinner("Processing..."):
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    if model_name == "SVM":
                        pred, conf, time_taken, error = predict_with_timing(traditional, user_text, model_name)
                        if error:
                            st.error(f"SVM Error: {error}")
                        else:
                            results[model_name] = {"prediction": pred, "confidence": conf, "time": time_taken}
                            st.metric("SVM", pred, f"Confidence: {conf:.3f}")
                            st.caption(f"Time: {time_taken:.3f}s")
                            
                    elif model_name == "Naive Bayes":
                        pred, conf, time_taken, error = predict_with_timing(traditional, user_text, model_name)
                        if error:
                            st.error(f"Naive Bayes Error: {error}")
                        else:
                            results[model_name] = {"prediction": pred, "confidence": conf, "time": time_taken}
                            st.metric("Naive Bayes", pred, f"Confidence: {conf:.3f}")
                            st.caption(f"Time: {time_taken:.3f}s")
                            
                    elif model_name == "BiLSTM":
                        pred, conf, time_taken, error = predict_with_timing(bilstm, user_text, model_name)
                        if error:
                            st.error(f"BiLSTM Error: {error}")
                        else:
                            results[model_name] = {"prediction": pred, "confidence": conf, "time": time_taken}
                            st.metric("BiLSTM", pred, f"Confidence: {conf:.3f}")
                            st.caption(f"Time: {time_taken:.3f}s")
                            
                    elif model_name == "AraBERT":
                        pred, conf, time_taken, error = predict_with_timing(arabert, user_text, model_name)
                        if error:
                            st.error(f"AraBERT Error: {error}")
                        else:
                            results[model_name] = {"prediction": pred, "confidence": conf, "time": time_taken}
                            st.metric("AraBERT", pred, f"Confidence: {conf:.3f}")
                            st.caption(f"Time: {time_taken:.3f}s")
        
        if results:
            st.subheader("📊 Results Summary:")
            
            df_results = pd.DataFrame([
                {
                    "Model": model,
                    "Prediction": data["prediction"],
                    "Confidence": f"{data.get('confidence', 0):.3f}",
                    "Time (s)": f"{data['time']:.3f}"
                }
                for model, data in results.items()
            ])
            
            st.dataframe(df_results, use_container_width=True)
            
            if len(results) > 1:
                fig_time = px.bar(
                    x=list(results.keys()),
                    y=[data["time"] for data in results.values()],
                    title="Model Inference Time Comparison",
                    labels={"x": "Model", "y": "Time (seconds)"}
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
                confidences = [data.get("confidence", 0) for data in results.values()]
                if any(conf > 0 for conf in confidences):
                    fig_conf = px.bar(
                        x=list(results.keys()),
                        y=confidences,
                        title="Model Confidence Comparison",
                        labels={"x": "Model", "y": "Confidence Score"}
                    )
                    st.plotly_chart(fig_conf, use_container_width=True) 