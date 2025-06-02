#!/usr/bin/env python
# coding: utf-8

"""Test script for AraBERT model"""

from src.models.arabert import AraBERTPredictor

def main():
    print("Testing AraBERT Model")
    print("=" * 25)
    
    predictor = AraBERTPredictor()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي من خلال دعم الفنون التقليدية",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم السعودية بعد إعلان الحكومة عن خطة تنموية جديدة",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة انتهت بنتيجة ٣-٢"
    ]
    
    for text in test_texts:
        prediction, confidence = predictor.predict_with_confidence(text)
        print(f"Text: {text[:60]}...")
        print(f"Prediction: {prediction} (Confidence: {confidence:.3f})\n")

if __name__ == "__main__":
    main() 