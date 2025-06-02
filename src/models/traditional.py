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
        
        # Test accuracy - comment out to reduce clutter
        # svm_acc = accuracy_score(y_test, self.svm_classifier.predict(X_test))
        # nb_acc = accuracy_score(y_test, self.nb_classifier.predict(X_test))
        # print(f"SVM Accuracy: {svm_acc:.4f}")
        # print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
        
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
        # print(f"N-gram model built with n={self.n} ({len(model)} prefixes)")  # Comment out
    
    def generate_text(self, length=100, start_word=None):
        if not self.is_trained:
            self.build_model()
            
        if start_word is None:
            start_word = random.choice(self.all_words)
        
        valid_prefixes = [prefix for prefix in self.model.keys() if start_word in prefix]
        
        if valid_prefixes:
            current = random.choice(valid_prefixes)
        else:
            current = random.choice(list(self.model.keys()))
        
        result = list(current)
        
        for _ in range(length):
            if current in self.model:
                next_word = random.choice(self.model[current])
                result.append(next_word)
                current = tuple(result[-(self.n-1):])
            else:
                current = random.choice(list(self.model.keys()))
                result.extend(current)
        
        return ' '.join(result)

def main():
    # Test classification
    classifier = TraditionalClassifier()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني"
    ]
    
    print("Classification Results:")
    for text in test_texts:
        prediction = classifier.predict(text)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {prediction}\n")
    
    # Test generation
    generator = NGramGenerator(n=4)
    
    print("Generated Text:")
    generated = generator.generate_text(length=50)
    print(generated)

if __name__ == "__main__":
    main() 