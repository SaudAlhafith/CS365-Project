#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import time
from src.models.traditional import TraditionalClassifier, NGramGenerator
from src.models.bilstm import BiLSTMPredictor
from src.models.arabert import AraBERTPredictor

@st.cache_resource
def load_models():
    try:
        traditional = TraditionalClassifier()
        traditional.train()
        bilstm = BiLSTMPredictor()
        arabert = AraBERTPredictor()
        generator = NGramGenerator(n=4)
        return traditional, bilstm, arabert, generator
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def predict_with_timing(model, text, model_name):
    try:
        start_time = time.time()
        
        if model_name == "AraBERT":
            prediction, confidence = model.predict_with_confidence(text)
            inference_time = time.time() - start_time
            return prediction, confidence, inference_time, None
        elif model_name == "BiLSTM":
            prediction, confidence = model.predict_with_confidence(text)
            inference_time = time.time() - start_time
            return prediction, confidence, inference_time, None
        elif model_name == "SVM":
            prediction, confidence = model.predict_with_confidence(text, model='svm')
            inference_time = time.time() - start_time
            return prediction, confidence, inference_time, None
        elif model_name == "Naive Bayes":
            prediction, confidence = model.predict_with_confidence(text, model='nb')
            inference_time = time.time() - start_time
            return prediction, confidence, inference_time, None
        else:
            prediction = model.predict(text)
            inference_time = time.time() - start_time
            return prediction, None, inference_time, None
            
    except Exception as e:
        return None, None, None, str(e)

SAMPLE_TEXTS = {
    "Culture": "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي من خلال دعم الفنون التقليدية",
    "Economy": "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم السعودية بعد إعلان الحكومة عن خطة تنموية جديدة",
    "Sports": "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة انتهت بنتيجة ٣-٢",
    "International": "اجتمع زعماء الدول الكبرى في نيويورك لمناقشة قضايا دولية مهمة مثل الصراعات الإقليمية",
    "Local": "بدأت أمانة المدينة بتنفيذ مشروع توسعة الطرق الداخلية بهدف تخفيف الازدحام المروري",
    "Religion": "حثّ إمام المسجد خلال خطبة الجمعة على التمسك بالقيم الإسلامية ونشر التسامح بين أفراد المجتمع مشيرًا إلى أهمية الصدق والأمانة"
}

CATEGORIES = ["CULTURE", "ECONOMY", "INTERNATIONAL", "LOCAL", "RELIGION", "SPORTS"] 