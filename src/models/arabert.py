#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.text_preprocessing import KalimatCorpusProcessor

class AraBERTPredictor:
    def __init__(self, model_path="results/checkpoint-288-best"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "aubmindlab/bert-base-arabertv02"
        self.model_path = model_path
        self.processor = KalimatCorpusProcessor()
        
        self.arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = None
        self.label_encoder = None
        self._setup_labels()
        self._load_model()
    
    def _setup_labels(self):
        df = self.processor.load_data_classification()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df['category'])
        
    def _load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        # print(f"AraBERT model loaded from {self.model_path}")  # Comment out to reduce clutter
    
    def _encode_text(self, text, max_len=512):
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length')
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        return tokens, attention_mask
    
    def predict(self, text):
        processed_text = self.arabert_prep.preprocess(text)
        tokens, attention_mask = self._encode_text(processed_text)
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
            
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def predict_with_confidence(self, text):
        processed_text = self.arabert_prep.preprocess(text)
        tokens, attention_mask = self._encode_text(processed_text)
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_label, float(confidence)

def main():
    predictor = AraBERTPredictor()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي من خلال دعم الفنون التقليدية",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم السعودية بعد إعلان الحكومة عن خطة تنموية جديدة",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة انتهت بنتيجة ٣-٢"
    ]
    
    print("AraBERT Predictions:")
    for text in test_texts:
        prediction, confidence = predictor.predict_with_confidence(text)
        print(f"Text: {text[:60]}...")
        print(f"Prediction: {prediction} (Confidence: {confidence:.3f})\n")

if __name__ == "__main__":
    main() 