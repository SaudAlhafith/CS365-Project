import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.utils.testing import ModelTester

def render_testing_tab():
    st.header("🧪 Comprehensive Model Testing")
    st.markdown("**Rigorous evaluation using the same random state as training to ensure proper data split**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Testing Configuration")
        st.markdown("""
        - **Random State**: 42 (same as training)
        - **Test Size**: 20% of total dataset  
        - **Stratified Split**: Maintains category distribution
        - **Metrics**: Accuracy, F1-Score, Confidence, Inference Time
        - **Batch Processing**: Fast inference using batch predictions
        """)
    
    with col2:
        st.subheader("⚙️ Controls")
        
        # Check if test is currently running
        is_testing = st.session_state.get('run_test', False)
        has_results = hasattr(st.session_state, 'test_results') and st.session_state.test_results and st.session_state.get('test_complete', False)
        
        # Show appropriate button state
        if is_testing:
            st.button("🔄 Testing in Progress...", disabled=True, type="secondary", key="test_disabled_btn")
            st.info("⏳ Please wait while the comprehensive test is running...")
            
            # Show a cancel option (optional)
            if st.button("⛔ Cancel Test", key="cancel_test_btn"):
                st.session_state.run_test = False
                st.session_state.test_complete = False
                st.warning("Test cancelled by user.")
                st.rerun()
        else:
            test_button = st.button("🚀 Start Comprehensive Test", type="primary", key="comprehensive_test_btn")
            
            if test_button:
                st.session_state.run_test = True
                st.session_state.test_complete = False
                # Ensure URL stays on testing tab
                st.query_params["tab"] = "testing"
                st.rerun()  # Immediately rerun to show loading state
        
        # Clear results button (only show if there are results)
        if has_results and not is_testing:
            st.markdown("---")
            if st.button("🗑️ Clear Results", key="clear_results_btn"):
                # Clear all test-related session state
                if 'test_results' in st.session_state:
                    del st.session_state.test_results
                if 'summary_df' in st.session_state:
                    del st.session_state.summary_df
                if 'test_complete' in st.session_state:
                    del st.session_state.test_complete
                st.success("Results cleared!")
                st.rerun()
    
    # Only run test if button was clicked and test isn't already complete
    if hasattr(st.session_state, 'run_test') and st.session_state.run_test and not st.session_state.get('test_complete', False):
        
        # Create containers for progress tracking
        progress_container = st.container()
        
        with progress_container:
            tester = ModelTester(random_state=42, test_size=0.2)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress tracking function
            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(message)
            
            tester.set_progress_callback(update_progress)
            
            status_text.text("📊 Preparing test data...")
            progress_bar.progress(0.05)
            
            X_test, y_test = tester.prepare_test_data()
            total_samples = len(X_test)
            st.info(f"Testing on {total_samples} samples using batch processing...")
            
            status_text.text("🔧 Testing Traditional Models...")
            tester.test_traditional_models()
            
            status_text.text("🧠 Testing BiLSTM Model (Batch Processing)...")
            tester.test_bilstm_model()
            
            status_text.text("🤖 Testing AraBERT Model (Batch Processing)...")
            tester.test_arabert_model()
            
            status_text.text("📈 Generating insights and analysis...")
            progress_bar.progress(1.0)
            summary_df = tester.generate_insights()
            
            status_text.text("✅ Testing completed!")
            
            # Check for errors and display them
            errors = []
            for model_name, result in tester.results.items():
                if isinstance(result, dict) and 'error' in result:
                    errors.append(f"**{model_name}**: {result['error']}")
            
            if errors:
                st.error("Some models failed to run:")
                for error in errors:
                    st.markdown(f"- {error}")
            
            st.success("✅ All models tested using the same data splits and methodology as original training!")
            
            # Store results in session state
            st.session_state.test_results = tester.results
            st.session_state.summary_df = summary_df
            st.session_state.test_complete = True
            st.session_state.run_test = False
            
            # Clear the progress container
            progress_container.empty()
    
    # Display results if available
    if hasattr(st.session_state, 'test_results') and st.session_state.test_results and st.session_state.get('test_complete', False):
        display_test_results(st.session_state.test_results, st.session_state.summary_df)

def display_test_results(results, summary_df):
    st.markdown("---")
    st.subheader("📊 Test Results")
    
    # Filter out failed models for display
    successful_results = {name: result for name, result in results.items() 
                         if result is not None and not isinstance(result, dict) or 'error' not in result}
    
    if not successful_results:
        st.error("No models completed testing successfully.")
        return
    
    st.subheader("📈 Performance Summary")
    st.dataframe(summary_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = max([r['accuracy'] for r in successful_results.values()])
        best_model = [name for name, result in successful_results.items() 
                     if result['accuracy'] == best_accuracy][0]
        st.metric("🏆 Best Accuracy", f"{best_accuracy:.4f}", best_model)
    
    with col2:
        fastest_time = min([np.mean(r['times']) for r in successful_results.values()])
        fastest_model = [name for name, result in successful_results.items() 
                        if np.mean(result['times']) == fastest_time][0]
        st.metric("⚡ Fastest Model", f"{fastest_time:.4f}s", fastest_model)
    
    with col3:
        highest_conf = max([np.mean(r['confidences']) for r in successful_results.values()])
        most_confident = [name for name, result in successful_results.items() 
                         if np.mean(result['confidences']) == highest_conf][0]
        st.metric("🎯 Highest Confidence", f"{highest_conf:.4f}", most_confident)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Accuracy Comparison")
        accuracy_data = {name: result['accuracy'] for name, result in successful_results.items()}
        
        fig_acc = px.bar(
            x=list(accuracy_data.keys()),
            y=list(accuracy_data.values()),
            title="Model Accuracy",
            color=list(accuracy_data.values()),
            color_continuous_scale="RdYlGn",
            range_color=[0, 1]
        )
        fig_acc.update_layout(showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Speed Comparison")
        speed_data = {name: np.mean(result['times']) for name, result in successful_results.items()}
        
        fig_speed = px.bar(
            x=list(speed_data.keys()),
            y=list(speed_data.values()),
            title="Average Inference Time (s)",
            color=list(speed_data.values()),
            color_continuous_scale="RdYlBu_r"
        )
        fig_speed.update_layout(showlegend=False)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 F1-Score Comparison")
        f1_data = {name: result['f1_score'] for name, result in successful_results.items()}
        
        fig_f1 = px.bar(
            x=list(f1_data.keys()),
            y=list(f1_data.values()),
            title="F1-Score (Weighted)",
            color=list(f1_data.values()),
            color_continuous_scale="Viridis"
        )
        fig_f1.update_layout(showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Average Confidence")
        conf_data = {name: np.mean(result['confidences']) for name, result in successful_results.items()}
        
        fig_conf = px.bar(
            x=list(conf_data.keys()),
            y=list(conf_data.values()),
            title="Average Confidence Score",
            color=list(conf_data.values()),
            color_continuous_scale="Plasma"
        )
        fig_conf.update_layout(showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    st.subheader("📋 Detailed Performance Metrics")
    
    detailed_data = []
    for model_name, result in successful_results.items():
        detailed_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'F1-Score': result['f1_score'],
            'Avg Confidence': np.mean(result['confidences']),
            'Min Confidence': np.min(result['confidences']),
            'Max Confidence': np.max(result['confidences']),
            'Avg Time (s)': np.mean(result['times']),
            'Min Time (s)': np.min(result['times']),
            'Max Time (s)': np.max(result['times']),
            'Total Samples': len(result['predictions'])
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(detailed_df, use_container_width=True)
    
    st.subheader("💡 Key Insights")
    
    insights = []
    
    if accuracy_data:
        best_acc_model = max(accuracy_data, key=accuracy_data.get)
        worst_acc_model = min(accuracy_data, key=accuracy_data.get)
        insights.append(f"🏆 **{best_acc_model}** achieved the highest accuracy of {accuracy_data[best_acc_model]:.4f}")
        insights.append(f"📉 **{worst_acc_model}** had the lowest accuracy of {accuracy_data[worst_acc_model]:.4f}")
    
    if speed_data and len(speed_data) > 1:
        fastest_model = min(speed_data, key=speed_data.get)
        slowest_model = max(speed_data, key=speed_data.get)
        speed_ratio = speed_data[slowest_model] / speed_data[fastest_model]
        insights.append(f"⚡ **{fastest_model}** is {speed_ratio:.1f}x faster than **{slowest_model}**")
    
    if conf_data:
        most_confident_model = max(conf_data, key=conf_data.get)
        insights.append(f"🎯 **{most_confident_model}** shows highest confidence ({conf_data[most_confident_model]:.4f})")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    if len(successful_results) > 1:
        st.subheader("⚖️ Performance Trade-offs")
        
        fig_scatter = go.Figure()
        
        for model_name, result in successful_results.items():
            fig_scatter.add_trace(go.Scatter(
                x=[np.mean(result['times'])],
                y=[result['accuracy']],
                mode='markers+text',
                name=model_name,
                text=[model_name],
                textposition="top center",
                marker=dict(size=np.mean(result['confidences']) * 50)
            ))
        
        fig_scatter.update_layout(
            title="Accuracy vs Speed Trade-off (bubble size = confidence)",
            xaxis_title="Average Inference Time (s)",
            yaxis_title="Accuracy",
            showlegend=False
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True) 