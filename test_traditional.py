#!/usr/bin/env python
# coding: utf-8

"""Test script for traditional models"""

from src.models.traditional import TraditionalClassifier, NGramGenerator

def main():
    print("Testing Traditional Models")
    print("=" * 30)
    
    # Test classification
    classifier = TraditionalClassifier()
    
    test_text = "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة"
    
    print(f"Text: {test_text}")
    print(f"SVM Prediction: {classifier.predict(test_text, model='svm')}")
    print(f"Naive Bayes Prediction: {classifier.predict(test_text, model='nb')}")
    
    print("\nTesting N-gram Generation:")
    generator = NGramGenerator(n=3)
    generated = generator.generate_text(length=30)
    print(f"Generated: {generated}")

if __name__ == "__main__":
    main() 