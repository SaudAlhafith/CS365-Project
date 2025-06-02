#!/usr/bin/env python
# coding: utf-8

"""Test script for BiLSTM model"""

from src.models.bilstm import BiLSTMPredictor

def main():
    print("Testing BiLSTM Model")
    print("=" * 25)
    
    predictor = BiLSTMPredictor()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة"
    ]
    
    for text in test_texts:
        prediction = predictor.predict(text)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {prediction}\n")

if __name__ == "__main__":
    main() 