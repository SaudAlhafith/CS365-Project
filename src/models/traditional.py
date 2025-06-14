#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict, Counter
import random
from ..preprocessing.text_preprocessing import KalimatCorpusProcessor

class TraditionalClassifier:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.svm_classifier = LinearSVC(random_state=42, C=1.0)
        self.nb_classifier = MultinomialNB(alpha=0.01)
        self.processor = KalimatCorpusProcessor()
        self.is_trained = False
    
    def train(self):
        df = self.processor.load_data_classification()
        
        X = self.tfidf_vectorizer.fit_transform(df['processed_text_classification'])
        y = df['category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.svm_classifier.fit(X_train, y_train)
        self.nb_classifier.fit(X_train, y_train)
        
        self.is_trained = True
        
    def predict(self, text, model='svm'):
        if not self.is_trained:
            self.train()
            
        processed_text = self.processor.preprocess_text_classification(text)
        X_new = self.tfidf_vectorizer.transform([processed_text])
        
        if model == 'svm':
            return self.svm_classifier.predict(X_new)[0]
        else:
            return self.nb_classifier.predict(X_new)[0]
    
    def predict_with_confidence(self, text, model='svm'):
        if not self.is_trained:
            self.train()
            
        processed_text = self.processor.preprocess_text_classification(text)
        X_new = self.tfidf_vectorizer.transform([processed_text])
        
        if model == 'svm':
            decision_scores = self.svm_classifier.decision_function(X_new)[0]
            prediction = self.svm_classifier.predict(X_new)[0]
            
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            confidence = np.max(probabilities)
            
            return prediction, float(confidence)
        else:
            probabilities = self.nb_classifier.predict_proba(X_new)[0]
            prediction = self.nb_classifier.predict(X_new)[0]
            confidence = np.max(probabilities)
            
            return prediction, float(confidence)

class NGramGenerator:
    def __init__(self, n=3):
        self.n = n
        self.model = None
        self.all_words = None
        self.processor = KalimatCorpusProcessor()
        self.is_trained = False
    
    def build_model(self):
        df = self.processor.load_data_ngram()
        texts = df['processed_text_ngram'].tolist()
        
        model = defaultdict(list)
        all_words = []
        
        for text in texts:
            words = text.split()
            all_words.extend(words)
            
            for i in range(len(words) - self.n + 1):
                prefix = tuple(words[i:i+self.n-1])
                suffix = words[i+self.n-1]
                model[prefix].append(suffix)
        
        self.model = model
        self.all_words = list(set(all_words))
        self.is_trained = True
    
    def generate_text(self, length=100, start_word=None):
        if not self.is_trained:
            self.build_model()
            
        if start_word is None:
            # No start word specified, pick a random prefix
            current = random.choice(list(self.model.keys()))
            result = list(current)
        else:
            # Start word specified, ensure the text begins with it
            start_words = start_word.strip().split()
            
            if len(start_words) >= self.n - 1:
                # If start text has enough words, use last n-1 words as prefix
                prefix = tuple(start_words[-(self.n-1):])
                result = start_words
            else:
                # If start text is shorter, find prefixes that start with it
                if len(start_words) == 1:
                    # Single start word - find prefixes that begin with this word
                    valid_prefixes = [prefix for prefix in self.model.keys() if prefix[0] == start_words[0]]
                else:
                    # Multiple start words - find prefixes that start with these words
                    valid_prefixes = [prefix for prefix in self.model.keys() 
                                    if prefix[:len(start_words)] == tuple(start_words)]
                
                if valid_prefixes:
                    # Found valid prefixes, choose one randomly
                    current_prefix = random.choice(valid_prefixes)
                    result = list(current_prefix)
                    prefix = current_prefix
                else:
                    # No valid prefixes found, start with user's words and find next best prefix
                    result = start_words[:]
                    # Try to find a prefix starting with the last word
                    last_word_prefixes = [prefix for prefix in self.model.keys() if prefix[0] == start_words[-1]]
                    if last_word_prefixes:
                        prefix = random.choice(last_word_prefixes)
                        result.extend(list(prefix)[1:])  # Add the rest of the prefix (excluding first word which we already have)
                    else:
                        # Fallback to random prefix
                        prefix = random.choice(list(self.model.keys()))
                        result.extend(list(prefix))
            
            # Set current context for generation
            if len(result) >= self.n - 1:
                current = tuple(result[-(self.n-1):])
            else:
                current = random.choice(list(self.model.keys()))
                result.extend(list(current))
                current = tuple(result[-(self.n-1):])
        
        # Generate the rest of the text
        for _ in range(length - len(result)):
            if current in self.model:
                next_word = random.choice(self.model[current])
                result.append(next_word)
                current = tuple(result[-(self.n-1):])
            else:
                # Current context not found, pick a random prefix and continue
                current = random.choice(list(self.model.keys()))
                result.extend(list(current))
                current = tuple(result[-(self.n-1):])
        
        return ' '.join(result)

def main():
    classifier = TraditionalClassifier()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني"
    ]
    
    for text in test_texts:
        prediction = classifier.predict(text)
    
    generator = NGramGenerator(n=4)
    
    generated = generator.generate_text(length=50)

if __name__ == "__main__":
    main() 