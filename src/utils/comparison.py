#!/usr/bin/env python
# coding: utf-8

from ..models.traditional import TraditionalClassifier, NGramGenerator
from ..models.bilstm import BiLSTMPredictor
from ..models.arabert import AraBERTPredictor

def compare_models():
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي من خلال دعم الفنون التقليدية والمهرجانات المحلية",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم السعودية بعد إعلان الحكومة عن خطة تنموية جديدة تركز على التنوع الاقتصادي",
        "اجتمع زعماء الدول الكبرى في نيويورك لمناقشة قضايا دولية مهمة مثل الصراعات الإقليمية وتغير المناخ والأمن العالمي",
        "بدأت أمانة المدينة بتنفيذ مشروع توسعة الطرق الداخلية بهدف تخفيف الازدحام المروري كما تم الإعلان عن إنشاء ممرات مشاة",
        "حثّ إمام المسجد خلال خطبة الجمعة على التمسك بالقيم الإسلامية ونشر التسامح بين أفراد المجتمع مشيرًا إلى أهمية الصدق والأمانة",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة انتهت بنتيجة ٣-٢ ليضمن التأهل إلى نهائي كأس آسيا"
    ]
    
    expected_categories = ["CULTURE", "ECONOMY", "INTERNATIONAL", "LOCAL", "RELIGION", "SPORTS"]
    
    # print("Loading models...")  # Comment out to reduce clutter
    traditional = TraditionalClassifier()
    bilstm = BiLSTMPredictor()
    arabert = AraBERTPredictor()
    
    print("\nModel Comparison Results:")
    print("=" * 120)
    print(f"{'Text':<50} {'Expected':<12} {'SVM':<12} {'BiLSTM':<12} {'AraBERT':<12}")
    print("-" * 120)
    
    for i, text in enumerate(test_texts):
        expected = expected_categories[i]
        
        svm_pred = traditional.predict(text, model='svm')
        bilstm_pred = bilstm.predict(text)
        arabert_pred, confidence = arabert.predict_with_confidence(text)
        
        text_short = text[:47] + "..." if len(text) > 50 else text
        
        print(f"{text_short:<50} {expected:<12} {svm_pred:<12} {bilstm_pred:<12} {arabert_pred:<12}")
    
    print("\nText Generation Example:")
    print("-" * 50)
    generator = NGramGenerator(n=4)
    generated = generator.generate_text(length=30)
    print(f"Generated: {generated}")

def quick_inference_demo():
    
    print("Quick Inference Demo")
    print("=" * 40)
    
    user_text = "تمكن الفريق المحلي من تحقيق فوز كبير في المباراة الأخيرة أمام الفريق الزائر بنتيجة واضحة"
    
    print(f"Input text: {user_text}\n")
    
    # Traditional
    traditional = TraditionalClassifier()
    svm_result = traditional.predict(user_text, model='svm')
    nb_result = traditional.predict(user_text, model='nb')
    print(f"SVM: {svm_result}")
    print(f"Naive Bayes: {nb_result}")
    
    # BiLSTM
    bilstm = BiLSTMPredictor()
    bilstm_result = bilstm.predict(user_text)
    print(f"BiLSTM: {bilstm_result}")
    
    # AraBERT
    arabert = AraBERTPredictor()
    arabert_result, confidence = arabert.predict_with_confidence(user_text)
    print(f"AraBERT: {arabert_result} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    # compare_models()
    quick_inference_demo() 