#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from components.model_utils import load_models, CATEGORIES
from components.classification_tab import render_classification_tab
from components.testing_tab import render_testing_tab
from components.generation_tab import render_generation_tab
from components.summarization_tab import render_summarization_tab

# Page configuration
st.set_page_config(
    page_title="Arabic Text Classification Demo",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic text
st.markdown("""
<style>
.arabic-text {
    direction: rtl;
    text-align: right;
    font-family: 'Arial Unicode MS', Arial, sans-serif;
    font-size: 16px;
}
.model-card {
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔤 Arabic NLP Demo")
    st.markdown("**CS365 Project - Phase 2**: Arabic Text Classification, Generation & Summarization")
    
    # Handle URL parameters to maintain tab state
    query_params = st.query_params
    
    # Determine active tab based on URL or session state
    tab_mapping = {
        "classification": 0,
        "comparison": 1, 
        "testing": 2,
        "generation": 3,
        "summarization": 4
    }
    
    # Get tab from URL parameter or default to first tab
    current_tab_key = query_params.get("tab", "classification")
    if current_tab_key not in tab_mapping:
        current_tab_key = "classification"
    
    active_tab_index = tab_mapping[current_tab_key]
    
    # Store in session state for consistency
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = active_tab_index
    
    # Update session state if URL changed
    if st.session_state.current_tab != active_tab_index:
        st.session_state.current_tab = active_tab_index
    
    # Load models with loading indicator
    with st.spinner("Loading and training models... (This may take a few seconds)"):
        traditional, bilstm, arabert, generator = load_models()
    
    if not all([traditional, bilstm, arabert, generator]):
        st.error("Failed to load models. Please check your model files.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Navigation sidebar
    st.sidebar.markdown("### 🧭 Navigation")
    tab_labels = ["🔍 Text Classification", "📊 Quick Comparison", "🧪 Comprehensive Testing", "✍️ Text Generation", "📄 Text Summarization"]
    tab_keys = ["classification", "comparison", "testing", "generation", "summarization"]
    
    for i, (label, key) in enumerate(zip(tab_labels, tab_keys)):
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
            st.query_params["tab"] = key
            st.rerun()
    
    # Highlight current tab in sidebar
    current_tab_name = tab_labels[active_tab_index]
    # st.sidebar.success(f"**Current**: {current_tab_name}")
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare:",
        ["SVM", "Naive Bayes", "BiLSTM", "AraBERT"],
        default=["SVM", "Naive Bayes", "BiLSTM", "AraBERT"]
    )
    
    # Categories info
    st.sidebar.markdown("### 📋 Categories")
    for cat in CATEGORIES:
        st.sidebar.markdown(f"• {cat}")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This demo showcases Arabic NLP tasks including:
    - **Classification**: SVM, BiLSTM, AraBERT models
    - **Generation**: N-gram based text generation
    - **Summarization**: Seq2Seq LSTM & AraBART models
    """)
    
    # Render content based on active tab
    if active_tab_index == 0:  # Text Classification
        render_classification_tab(traditional, bilstm, arabert, selected_models)
    
    elif active_tab_index == 1:  # Quick Comparison
        st.header("📊 Quick Model Comparison")
        st.markdown("**Quick comparison on sample texts** (for comprehensive testing, use the Testing tab)")
        
        from components.model_utils import SAMPLE_TEXTS, predict_with_timing
        import pandas as pd
        import plotly.express as px
        
        if st.button("🧪 Run Quick Test", type="primary"):
            progress_bar = st.progress(0)
            results_data = []
            
            sample_cases = list(SAMPLE_TEXTS.items())
            
            for i, (category, text) in enumerate(sample_cases):
                expected_category = category.upper()
                
                # Test selected models
                for model_name in selected_models:
                    if model_name == "SVM":
                        pred, conf, time_taken, error = predict_with_timing(traditional, text, model_name)
                    elif model_name == "Naive Bayes":
                        pred, conf, time_taken, error = predict_with_timing(traditional, text, model_name)
                    elif model_name == "BiLSTM":
                        pred, conf, time_taken, error = predict_with_timing(bilstm, text, model_name)
                    elif model_name == "AraBERT":
                        pred, conf, time_taken, error = predict_with_timing(arabert, text, model_name)
                    
                    if not error:
                        results_data.append({
                            "Test Case": f"Case {i+1} ({category})",
                            "Expected": expected_category,
                            "Model": model_name,
                            "Prediction": pred,
                            "Correct": pred == expected_category,
                            "Confidence": conf if conf else 0,
                            "Time (s)": time_taken
                        })
                
                progress_bar.progress((i + 1) / len(sample_cases))
            
            # Display results
            if results_data:
                df_results = pd.DataFrame(results_data)
                
                # Accuracy by model
                col1, col2 = st.columns(2)
                
                with col1:
                    accuracy_by_model = df_results.groupby('Model')['Correct'].mean() * 100
                    fig_acc = px.bar(
                        x=accuracy_by_model.index,
                        y=accuracy_by_model.values,
                        title="Quick Test Accuracy (%)",
                        color=accuracy_by_model.values,
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    speed_by_model = df_results.groupby('Model')['Time (s)'].mean()
                    fig_speed = px.bar(
                        x=speed_by_model.index,
                        y=speed_by_model.values,
                        title="Average Time (s)",
                        color=speed_by_model.values,
                        color_continuous_scale="RdYlBu_r"
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                display_df = df_results.pivot_table(
                    index='Test Case', 
                    columns='Model', 
                    values='Prediction', 
                    aggfunc='first'
                ).reset_index()
                st.dataframe(display_df, use_container_width=True)
    
    elif active_tab_index == 2:  # Comprehensive Testing
        render_testing_tab()
    
    elif active_tab_index == 3:  # Text Generation
        render_generation_tab(generator, arabert)
    
    elif active_tab_index == 4:  # Text Summarization
        render_summarization_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("**CS365 Arabic NLP Project** | Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main() 