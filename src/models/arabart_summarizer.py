#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ..preprocessing.text_preprocessing import KalimatCorpusProcessor

class AraBARTSummarizer:
    def __init__(self, model_path="results/summarization_model"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "moussaKam/AraBART"
        self.model_path = model_path
        self.processor = KalimatCorpusProcessor()
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned AraBART model"""
        try:
            # Try to load fine-tuned model first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            print(f"Loaded fine-tuned AraBART model from {self.model_path}")
            
        except Exception as e:
            print(f"Could not load fine-tuned model from {self.model_path}: {str(e)}")
            try:
                # Fallback to base model
                print("Falling back to base AraBART model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                print("Loaded base AraBART model")
                
            except Exception as e2:
                print(f"Could not load base AraBART model: {str(e2)}")
                self.model = None
                self.tokenizer = None
    
    def generate_summary(self, text, max_length=128, min_length=30, num_beams=4):
        """Generate summary for given text"""
        if self.model is None or self.tokenizer is None:
            return "AraBART model not available"
        
        try:
            # Preprocess text
            text = self.processor.normalize_arabic_summarization(text)
            
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate summary
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode generated summary
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary.strip()
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_summary_with_confidence(self, text, max_length=128, min_length=30, num_beams=4):
        """Generate summary with confidence score"""
        if self.model is None or self.tokenizer is None:
            return "AraBART model not available", 0.0
        
        try:
            # Preprocess text
            text = self.processor.normalize_arabic_summarization(text)
            
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate summary with scores
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Decode generated summary
                summary = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                
                # Calculate confidence score (simplified)
                if hasattr(outputs, 'sequences_scores'):
                    confidence = float(torch.exp(outputs.sequences_scores[0]))
                else:
                    confidence = 0.8  # Default confidence if scores not available
                
                return summary.strip(), confidence
        
        except Exception as e:
            return f"Error generating summary: {str(e)}", 0.0
    
    def generate_batch_summaries(self, texts, max_length=128, min_length=30, num_beams=4, batch_size=4):
        """Generate summaries for multiple texts in batches"""
        if self.model is None or self.tokenizer is None:
            return ["AraBART model not available"] * len(texts)
        
        summaries = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess batch
                processed_texts = [self.processor.normalize_arabic_summarization(text) for text in batch_texts]
                
                with torch.no_grad():
                    # Tokenize batch
                    inputs = self.tokenizer(
                        processed_texts,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate summaries
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        do_sample=False
                    )
                    
                    # Decode summaries
                    batch_summaries = [
                        self.tokenizer.decode(summary_id, skip_special_tokens=True).strip()
                        for summary_id in summary_ids
                    ]
                    summaries.extend(batch_summaries)
        
        except Exception as e:
            error_msg = f"Error in batch summarization: {str(e)}"
            summaries.extend([error_msg] * (len(texts) - len(summaries)))
        
        return summaries
    
    def is_available(self):
        """Check if model is available for inference"""
        return self.model is not None and self.tokenizer is not None

def main():
    summarizer = AraBARTSummarizer()
    
    if summarizer.is_available():
        test_texts = [
            "هذا نص تجريبي باللغة العربية لاختبار نموذج التلخيص الآلي باستخدام نموذج أرابارت المتقدم للذكاء الاصطناعي",
            "تطور التكنولوجيا في العالم العربي يشهد نموا كبيرا في السنوات الأخيرة مع زيادة الاستثمار في مجال الذكاء الاصطناعي والتعلم الآلي"
        ]
        
        for text in test_texts:
            summary, confidence = summarizer.generate_summary_with_confidence(text)
            print(f"Original: {text}")
            print(f"Summary: {summary}")
            print(f"Confidence: {confidence:.3f}")
            print("-" * 50)
    else:
        print("AraBART summarizer not available")

if __name__ == "__main__":
    main() 