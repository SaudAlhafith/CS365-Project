#!/usr/bin/env python
# coding: utf-8

"""
Text Preprocessing and Data Loading for Arabic Text Classification
Extracted from CS365-Project-Phase2-saud.py

This module provides:
1. Data loading from Kalimat Corpus
2. Preprocessing for traditional classification + BiLSTM
3. Preprocessing for N-gram generation
4. AraBERT preprocessing and tokenization
"""

import os
import multiprocessing.dummy as mp
import pandas as pd
import nltk
from nltk.corpus import stopwords
import regex as re
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

# Download required NLTK data
nltk.download('stopwords', quiet=True)

class KalimatCorpusProcessor:
    def __init__(self, kalimat_base="data/KalimatCorpus-2.0", 
                 classification_file="processed_classification_data.csv",
                 ngram_file="processed_ngram_data.csv"):
        self.kalimat_base = kalimat_base
        self.classification_file = classification_file
        self.ngram_file = ngram_file
        self.expected_dirs = os.listdir(kalimat_base)
        
        # Setup Arabic stopwords and stemmer
        self.arabic_stopwords = set(stopwords.words('arabic'))
        self.stemmer = nltk.stem.ISRIStemmer()
        for word in ['في', 'ان', 'ان', 'الى', 'او', 'فى']: 
            self.arabic_stopwords.add(word)
        
        # Setup AraBERT
        self.model_name = "aubmindlab/bert-base-arabertv02"
        self.arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def check_kalimat_structure_os(self):
        """Check if Kalimat Corpus structure is correct"""
        missing = [d for d in self.expected_dirs if not os.path.isdir(os.path.join(self.kalimat_base, d))]
        
        if missing:
            print(f"Missing folders: {missing}")
            return False
        else:
            count = 0
            for d in self.expected_dirs:
                folder_path = os.path.join(self.kalimat_base, d)
                count += len([f for f in os.listdir(folder_path)])
            print(f"Kalimat Corpus ready with {count} .txt files")
            return True
    
    def load_kalimat_articles(self, category):
        """Load articles from a specific category"""
        category_path = os.path.join(self.kalimat_base, category)
        if not os.path.isdir(category_path):
            print(f"Category '{category}' does not exist in the Kalimat Corpus.")
            return []

        articles = []
        for filename in os.listdir(category_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                    text = " ".join(words)
                    articles.append({
                        "category": category.replace("articles", "").upper(),
                        "filename": filename,
                        "text": text,
                        "text_length": len(text),
                        "word_count": len(words)
                    })

        print(f"Loaded {len(articles)} articles from category '{category}'")
        return articles
    
    def load_all_articles_parallel(self):
        """Load all articles using parallel processing"""
        with mp.Pool(processes=min(len(self.expected_dirs), int(os.cpu_count() / 2))) as pool:
            results = pool.map(self.load_kalimat_articles, self.expected_dirs)
        
        dataset = [article for category_articles in results for article in category_articles]
        return dataset
    
    def load_data_classification(self):
        """Main function to load and return the dataset as DataFrame"""
        if os.path.exists(self.classification_file):
            # print(f"Loading classification data from {self.classification_file}")  # Comment out
            df = pd.read_csv(self.classification_file)
            # print(f"Loaded {len(df)} preprocessed articles")  # Comment out
            return df
        
        print("Loading from source files...")
        if not self.check_kalimat_structure_os():
            raise FileNotFoundError("Kalimat Corpus data not found!")
        
        dataset = self.load_all_articles_parallel()
        print(f"Dataset loaded with {len(dataset)} articles")
        
        df = pd.DataFrame(dataset)
        original_count = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"After removing duplicates: {len(df)} articles (removed {original_count - len(df)})")
        
        print("Processing texts for classification...")
        df['processed_text_classification'] = df['text'].apply(self.preprocess_text_classification)
        
        df[['category', 'processed_text_classification']].to_csv(self.classification_file, index=False)
        print(f"Saved classification data to {self.classification_file}")
        
        return df
    
    def load_data_ngram(self):
        if os.path.exists(self.ngram_file):
            # print(f"Loading N-gram data from {self.ngram_file}")  # Comment out
            df = pd.read_csv(self.ngram_file)
            # print(f"Loaded {len(df)} preprocessed articles")  # Comment out
            return df
        
        print("Loading from source files...")
        if not self.check_kalimat_structure_os():
            raise FileNotFoundError("Kalimat Corpus data not found!")
        
        dataset = self.load_all_articles_parallel()
        print(f"Dataset loaded with {len(dataset)} articles")
        
        df = pd.DataFrame(dataset)
        original_count = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"After removing duplicates: {len(df)} articles (removed {original_count - len(df)})")
        
        print("Processing texts for N-gram...")
        df['processed_text_ngram'] = df['text'].apply(self.preprocess_text_ngram)
        
        df[['category', 'processed_text_ngram']].to_csv(self.ngram_file, index=False)
        print(f"Saved N-gram data to {self.ngram_file}")
        
        return df
    
    def load_data(self):
        """Main function to load and return the dataset as DataFrame"""
        return self.load_data_classification()
    
    # =========== PREPROCESSING METHODS ===========
    
    def preprocess_text_classification(self, text):
        """
        Preprocessing for traditional classification and BiLSTM
        - Removes punctuation, digits, English letters
        - Normalizes Arabic letters
        - Removes diacritics
        - Applies stemming and stopword removal
        """
        text = re.sub(r'\p{P}+|\$', '', text)  # remove all punctuation (English + Arabic)
        text = re.sub(r'[0-9٠-٩]', '', text)  # remove Arabic and English digits
        text = re.sub(r'[a-zA-Z]', '', text)  # remove English letters
        text = re.sub(r'[اآإأ]', 'ا', text)  # replace Arabic letter with hamza with 'ا'
        text = re.sub(r'[\u064B-\u0652]', '', text)  # remove Arabic diacritics
        text = re.sub(r'\s+', ' ', text).strip()  # clean extra spaces

        tokens = text.split()
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.arabic_stopwords]

        return ' '.join(tokens)
    
    def preprocess_text_ngram(self, text):
        """
        Preprocessing for N-gram generation
        - Keeps only Arabic characters
        - Removes diacritics
        - Normalizes Arabic letters
        - No stemming or stopword removal (to preserve natural flow)
        """
        # Remove non-Arabic characters and normalize whitespace
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text) # keep only Arabic characters
        text = re.sub(r'[\u064B-\u0652]', '', text)  # remove Arabic diacritics
        text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
        text = re.sub(r'[اآإأ]', 'ا', text)  # replace Arabic letter with hamza with 'ا'

        return text
    
    def preprocess_text_arabert(self, text):
        """
        Preprocessing for AraBERT
        Uses AraBERT's built-in preprocessor
        """
        return self.arabert_prep.preprocess(text)
    
    def encode_text_transformer(self, text, max_len=128):
        """
        AraBERT tokenization method that chunks text and adds special tokens
        """
        assert max_len < 512, "Max length for BERT should be less than 512 tokens."
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        for i in range(0, len(tokens), max_len - 2):
            chunk = tokens[i:i + (max_len - 2)]

            chunk = [2] + chunk + [3]  # Add [CLS] and [SEP] tokens

            padding_length = max_len - len(chunk)
            chunk += [0] * padding_length  # Pad with zeros
            chunks.append(chunk)
        
        return chunks
    
    def decode_text_transformer(self, encoded_article):
        """Decode AraBERT tokens back to text"""
        decoded = self.tokenizer.decode(encoded_article, skip_special_tokens=True)
        return decoded.replace('  ', ' ').strip()  # Clean up double spaces


def main():
    """Main function to demonstrate usage"""
    # Initialize processor
    processor = KalimatCorpusProcessor()
    
    # Load classification data
    df_classification = processor.load_data_classification()
    print(f"\nClassification dataset shape: {df_classification.shape}")
    
    # Load N-gram data
    df_ngram = processor.load_data_ngram()
    print(f"N-gram dataset shape: {df_ngram.shape}")
    
    # Display results
    print(f"\nClassification categories: {df_classification['category'].value_counts()}")
    
    # Test individual preprocessing methods
    sample_text = "مرحباً بكم في جامعة السلطان قابوس، هذا نص تجريبي يحتوي على أرقام 123 وعلامات ترقيم!"
    
    print(f"\nOriginal text: {sample_text}")
    print(f"Classification: {processor.preprocess_text_classification(sample_text)}")
    print(f"N-gram: {processor.preprocess_text_ngram(sample_text)}")
    print(f"AraBERT: {processor.preprocess_text_arabert(sample_text)}")
    
    # Test AraBERT tokenization
    arabert_tokens = processor.encode_text_transformer(processor.preprocess_text_arabert(sample_text), max_len=50)
    print(f"AraBERT tokens: {arabert_tokens[0]}")
    
    return df_classification, df_ngram, processor


if __name__ == "__main__":
    df_classification, df_ngram, processor = main() 