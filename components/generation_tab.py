#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from .model_utils import predict_with_timing

def render_generation_tab(generator, arabert):
    st.header("✍️ Arabic Text Generation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Generation Settings")
        
        length = st.slider("Text Length (words):", min_value=10, max_value=100, value=50)
        start_word = st.text_input("Start Text (optional):", placeholder="النص المطلوب البدء به", help="You can enter a single word or multiple words to start the generation")
        
        if st.button("🎲 Generate Text", type="primary"):
            with st.spinner("Generating..."):
                if start_word.strip():
                    generated_text = generator.generate_text(length=length, start_word=start_word.strip())
                else:
                    generated_text = generator.generate_text(length=length)
                
                st.session_state.generated_text = generated_text
    
    with col2:
        st.subheader("📝 Generated Text")
        
        if hasattr(st.session_state, 'generated_text'):
            st.markdown(f'<div class="arabic-text">{st.session_state.generated_text}</div>', 
                       unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🔍 Classify Generated Text"):
                    with st.spinner("Classifying..."):
                        pred, conf, time_taken, error = predict_with_timing(
                            arabert, st.session_state.generated_text, "AraBERT"
                        )
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.success(f"**Category**: {pred}")
                            st.info(f"**Confidence**: {conf:.3f}")
                            st.caption(f"Time: {time_taken:.3f}s")
            
            with col_b:
                if st.button("🔄 Generate New Text"):
                    with st.spinner("Generating new text..."):
                        if start_word.strip():
                            generated_text = generator.generate_text(length=length, start_word=start_word.strip())
                        else:
                            generated_text = generator.generate_text(length=length)
                        
                        st.session_state.generated_text = generated_text
                        st.rerun()
        else:
            st.info("Click 'Generate Text' to create Arabic text using N-gram model")
            
            st.markdown("""
            **How it works:**
            - Uses 4-gram language model trained on the Kalimat Corpus
            - Generates coherent Arabic text based on learned patterns
            - You can specify a start word or let it choose randomly
            - Generated text can be classified to test consistency
            """)
    
    st.markdown("---")
    st.subheader("💡 Generation Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Good Start Text Examples:**
        - السلطان (Sultan)
        - الحكومة تعمل (Government works)
        - المنتخب فاز (Team won)
        - الإمام قال (Imam said)
        - الأسواق ارتفعت (Markets rose)
        - في المدينة (In the city)
        """)
    
    with col2:
        st.markdown("""
        **Text Length Guidelines:**
        - **10-20 words**: Short phrases
        - **30-50 words**: Paragraph snippets  
        - **70-100 words**: Full paragraphs
        """) 