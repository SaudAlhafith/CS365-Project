#!/usr/bin/env python
# coding: utf-8

"""
Example usage of the text preprocessing module
Shows how to use the KalimatCorpusProcessor for different preprocessing needs
"""

from text_preprocessing import KalimatCorpusProcessor
import pandas as pd

def example_usage():
    """Demonstrate how to use the processor for different tasks"""
    
    print("Arabic Text Processing Example")
    print("=" * 40)
    
    # Initialize the processor
    processor = KalimatCorpusProcessor()
    
    # Load data (will use CSV if available)
    df = processor.load_data()
    
    print(f"\nDataset Overview:")
    print(f"Total articles: {len(df)}")
    print(f"Categories: {list(df['category'].unique())}")
    print(f"Available columns: {list(df.columns)}")
    
    # =============== USE CASE 1: Traditional Classification + BiLSTM ===============
    print("\nUSE CASE 1: Traditional Classification + BiLSTM")
    print("-" * 50)
    
    # Get preprocessed text for classification
    classification_texts = df['processed_text_classification'].tolist()
    labels = df['category'].tolist()
    
    print(f"Ready for classification training!")
    print(f"- {len(classification_texts)} preprocessed texts")
    print(f"- Example: '{classification_texts[0][:100]}...'")
    
    # =============== USE CASE 2: N-gram Text Generation ===============
    print("\nUSE CASE 2: N-gram Text Generation")
    print("-" * 40)
    
    # Need to preprocess for N-gram on demand
    sample_text = df['text'].iloc[0]
    ngram_text = processor.preprocess_text_ngram(sample_text)
    
    print(f"N-gram preprocessing (preserving natural flow)")
    print(f"- Example: '{ngram_text[:100]}...'")
    
    # =============== USE CASE 3: AraBERT ===============
    print("\nUSE CASE 3: AraBERT Tokenization")
    print("-" * 35)
    
    # Need to preprocess for AraBERT on demand
    arabert_text = processor.preprocess_text_arabert(sample_text)
    tokens = processor.encode_text_transformer(arabert_text, max_len=128)
    
    print(f"AraBERT tokenization completed!")
    print(f"- Generated {len(tokens)} chunks")
    print(f"- First chunk (token IDs): {tokens[0][:20]}...")
    
    return df, processor

def quick_example():
    """Quick example for testing individual preprocessing functions"""
    
    print("Quick Preprocessing Example")
    print("=" * 30)
    
    # Sample Arabic text
    sample_text = "مرحباً بكم في جامعة السلطان قابوس، هذا نص تجريبي يحتوي على أرقام 123 وعلامات ترقيم!"
    
    processor = KalimatCorpusProcessor()
    
    print(f"\nOriginal text:\n{sample_text}\n")
    
    # Test all preprocessing methods
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
    # Run quick example first (no data loading)
    quick_example()
    
    # Then run full example
    # df, processor = example_usage()
    
    print("\nTo run the full dataset processing, uncomment the line:")
    print("# df, processor = example_usage()") 