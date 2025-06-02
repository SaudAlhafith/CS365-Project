#!/usr/bin/env python
# coding: utf-8

"""
Example usage of the text preprocessing module
Shows how to use the KalimatCorpusProcessor for different preprocessing needs
"""

from src.preprocessing.text_preprocessing import KalimatCorpusProcessor
import pandas as pd

def example_usage():
    """Demonstrate how to use the processor for different tasks"""
    
    print("Arabic Text Processing Example")
    print("=" * 40)
    
    processor = KalimatCorpusProcessor()
    
    # Load data (separate methods for classification and n-gram)
    df_classification = processor.load_data_classification()
    df_ngram = processor.load_data_ngram()
    
    print(f"\nDataset Overview:")
    print(f"Classification articles: {len(df_classification)}")
    print(f"N-gram articles: {len(df_ngram)}")
    print(f"Categories: {list(df_classification['category'].unique())}")
    
    # =============== USE CASE 1: Traditional Classification + BiLSTM ===============
    print("\nUSE CASE 1: Traditional Classification + BiLSTM")
    print("-" * 50)
    
    classification_texts = df_classification['processed_text_classification'].tolist()
    labels = df_classification['category'].tolist()
    
    print(f"Ready for classification training!")
    print(f"- {len(classification_texts)} preprocessed texts")
    print(f"- Example: '{classification_texts[0][:100]}...'")
    
    # =============== USE CASE 2: N-gram Text Generation ===============
    print("\nUSE CASE 2: N-gram Text Generation")
    print("-" * 40)
    
    ngram_texts = df_ngram['processed_text_ngram'].tolist()
    
    print(f"Ready for N-gram training!")
    print(f"- {len(ngram_texts)} preprocessed texts")
    print(f"- Example: '{ngram_texts[0][:100]}...'")
    
    # =============== USE CASE 3: AraBERT ===============
    print("\nUSE CASE 3: AraBERT Tokenization")
    print("-" * 35)
    
    sample_text = df_classification['text'].iloc[0]
    arabert_text = processor.preprocess_text_arabert(sample_text)
    tokens = processor.encode_text_transformer(arabert_text, max_len=128)
    
    print(f"AraBERT tokenization completed!")
    print(f"- Generated {len(tokens)} chunks")
    print(f"- First chunk (token IDs): {tokens[0][:20]}...")
    
    return df_classification, df_ngram, processor

def quick_example():
    """Quick example for testing individual preprocessing functions"""
    
    print("Quick Preprocessing Example")
    print("=" * 30)
    
    sample_text = "مرحباً بكم في جامعة السلطان قابوس، هذا نص تجريبي يحتوي على أرقام 123 وعلامات ترقيم!"
    
    processor = KalimatCorpusProcessor()
    
    print(f"\nOriginal text:\n{sample_text}\n")
    
    print("1. Classification preprocessing:")
    classification_result = processor.preprocess_text_classification(sample_text)
    print(f"   {classification_result}\n")
    
    print("2. N-gram preprocessing:")
    ngram_result = processor.preprocess_text_ngram(sample_text)
    print(f"   {ngram_result}\n")
    
    print("3. AraBERT preprocessing:")
    arabert_result = processor.preprocess_text_arabert(sample_text)
    print(f"   {arabert_result}\n")
    
    print("4. AraBERT tokenization:")
    tokens = processor.encode_text_transformer(arabert_result, max_len=50)
    print(f"   Tokens: {tokens[0]}\n")
    
    decoded = processor.decode_text_transformer(tokens[0])
    print(f"   Decoded: {decoded}")

if __name__ == "__main__":
    quick_example()
    
    print("\nTo run the full dataset processing:")
    print("# df_classification, df_ngram, processor = example_usage()") 