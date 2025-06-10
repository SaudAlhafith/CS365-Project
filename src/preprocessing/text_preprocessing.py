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
from nltk.stem import ISRIStemmer
import pickle

# Download required NLTK data
nltk.download('stopwords', quiet=True)

class KalimatCorpusProcessor:
    def __init__(self, kalimat_base="data/KalimatCorpus-2.0", 
                 classification_file="data/processed/classification_data.csv",
                 ngram_file="data/processed/ngram_data.csv"):
        self.kalimat_base = kalimat_base
        self.classification_file = classification_file
        self.ngram_file = ngram_file
        self.expected_dirs = os.listdir(kalimat_base) if os.path.exists(kalimat_base) else []
        
        # Create directories if they don't exist
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        
        # Setup Arabic stopwords and stemmer
        self.arabic_stopwords = set(stopwords.words('arabic'))
        self.stemmer = ISRIStemmer()
        for word in ['في', 'ان', 'ان', 'الى', 'او', 'فى']: 
            self.arabic_stopwords.add(word)
        
        # Setup AraBERT
        self.model_name = "aubmindlab/bert-base-arabertv02"
        self.arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def load_kalimat_articles(self, category):
        """Load articles from a specific category"""
        category_path = os.path.join(self.kalimat_base, category)
        if not os.path.isdir(category_path):
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
            return pd.read_csv(self.classification_file)
        
        dataset = self.load_all_articles_parallel()
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['text'])
        df['processed_text_classification'] = df['text'].apply(self.preprocess_text_classification)
        
        df[['category', 'processed_text_classification']].to_csv(self.classification_file, index=False)
        return df
    
    def load_data_ngram(self):
        if os.path.exists(self.ngram_file):
            return pd.read_csv(self.ngram_file)
        
        dataset = self.load_all_articles_parallel()
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['text'])
        df['processed_text_ngram'] = df['text'].apply(self.preprocess_text_ngram)
        
        df[['category', 'processed_text_ngram']].to_csv(self.ngram_file, index=False)
        return df
    
    def load_data_arabert(self):
        """Load data specifically for AraBERT - preserves original text"""
        arabert_file = "data/processed/arabert_data.csv"
        
        if os.path.exists(arabert_file):
            return pd.read_csv(arabert_file)
        
        dataset = self.load_all_articles_parallel()
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['text'])
        # For AraBERT, we keep the original text and just preprocess it
        df['processed_text_arabert'] = df['text'].apply(self.preprocess_text_arabert)
        
        df[['category', 'text', 'processed_text_arabert']].to_csv(arabert_file, index=False)
        return df
    
    def encode_text(self, article, vocab, max_len=500):
        """
        BiLSTM text encoding - chunks article into sequences of max_len
        Same as used in training
        """
        tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in article.split()]
        
        chunks = []
        for i in range(0, len(tokens), max_len):
            chunk = tokens[i:i + max_len]
            
            # Pad chunk to max_len
            if len(chunk) < max_len:
                chunk += [vocab.get('<PAD>', 0)] * (max_len - len(chunk))
            
            chunks.append(chunk)
        
        return chunks
    
    def encode_text_transformer(self, text, max_len=500):
        """
        AraBERT text encoding - chunks text into sequences for transformer
        Same as used in training
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
    
    def load_data_bilstm_chunked(self, max_len=500):
        """Load chunked data for BiLSTM training/testing - matches original training setup"""
        bilstm_chunked_file = "data/cache/bilstm_chunked_data.pkl"  # Use pickle for better performance
        
        if os.path.exists(bilstm_chunked_file):
            with open(bilstm_chunked_file, 'rb') as f:
                return pickle.load(f)
        
        # Load base classification data
        df = self.load_data_classification()
        
        # Create vocabulary (same as BiLSTM model)
        tokenized_text = [text.split() for text in df['processed_text_classification']]
        from collections import Counter
        word_counts = Counter(word for article in tokenized_text for word in article)
        
        vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.items())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        
        # Create label encoder (same as BiLSTM model)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['category'])
        
        # Chunk the data exactly as in training
        chunked_texts = []
        chunked_labels = []
        
        for article, label in zip(df['processed_text_classification'], encoded_labels):
            chunks = self.encode_text(article, vocab, max_len)
            chunked_texts.extend(chunks)
            chunked_labels.extend([label] * len(chunks))
        
        # Create chunked DataFrame with actual lists (not strings)
        chunked_df = pd.DataFrame({
            'chunked_text': chunked_texts,  # Keep as actual lists
            'encoded_label': chunked_labels,
            'category': [label_encoder.inverse_transform([label])[0] for label in chunked_labels]
        })
        
        # Save as pickle for better performance with lists
        with open(bilstm_chunked_file, 'wb') as f:
            pickle.dump(chunked_df, f)
        print(f"Created BiLSTM chunked dataset with {len(chunked_df)} chunks from {len(df)} articles")
        
        return chunked_df
    
    def load_data_arabert_chunked(self, max_len=500):
        """Load chunked data for AraBERT training/testing - matches original training setup"""
        arabert_chunked_file = "data/cache/arabert_chunked_data.pkl"  # Use pickle for better performance
        
        if os.path.exists(arabert_chunked_file):
            with open(arabert_chunked_file, 'rb') as f:
                return pickle.load(f)
        
        # Load base data and apply AraBERT preprocessing
        dataset = self.load_all_articles_parallel()
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['text'])
        
        # Apply AraBERT preprocessing (same as training)
        df['arabert_text'] = df['text'].apply(self.preprocess_text_arabert)
        
        # Create label encoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['category'])
        
        # Chunk the data exactly as in training
        chunked_texts = []
        chunked_labels = []
        
        for article, label in zip(df['arabert_text'], encoded_labels):
            chunks = self.encode_text_transformer(article, max_len)
            chunked_texts.extend(chunks)
            chunked_labels.extend([label] * len(chunks))
        
        # Create chunked DataFrame with actual lists (not strings)
        chunked_df = pd.DataFrame({
            'chunked_text': chunked_texts,  # Keep as actual lists
            'encoded_label': chunked_labels,
            'category': [label_encoder.inverse_transform([label])[0] for label in chunked_labels]
        })
        
        # Save as pickle for better performance with lists
        with open(arabert_chunked_file, 'wb') as f:
            pickle.dump(chunked_df, f)
        print(f"Created AraBERT chunked dataset with {len(chunked_df)} chunks from {len(df)} articles")
        
        return chunked_df
    
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
    
    # Load N-gram data
    df_ngram = processor.load_data_ngram()
    
    # Load AraBERT data
    df_arabert = processor.load_data_arabert()
    
    # Load chunked datasets (same as used in training)
    df_bilstm_chunked = processor.load_data_bilstm_chunked(max_len=500)
    df_arabert_chunked = processor.load_data_arabert_chunked(max_len=500)
    
    return df_classification, df_ngram, df_arabert, df_bilstm_chunked, df_arabert_chunked, processor


if __name__ == "__main__":
    df_classification, df_ngram, df_arabert, df_bilstm_chunked, df_arabert_chunked, processor = main() 