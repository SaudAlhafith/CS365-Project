#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from ..preprocessing.text_preprocessing import KalimatCorpusProcessor
from ..models.traditional import TraditionalClassifier
from ..models.bilstm import BiLSTMPredictor
from ..models.arabert import AraBERTPredictor

class ModelTester:
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.processor = KalimatCorpusProcessor()
        self.results = {}
        self.progress_callback = None
        
        # Files to store train/test indices for consistency
        self.indices_file = "test_indices.pkl"
        self.bilstm_indices_file = "bilstm_chunked_test_indices.pkl"
        self.arabert_indices_file = "arabert_chunked_test_indices.pkl"
        
        # Prepare different datasets for different models
        self.X_test_classification = None  # For Traditional models
        self.X_test_bilstm_chunked = None  # For BiLSTM (chunked data)
        self.X_test_arabert_chunked = None # For AraBERT (chunked data)
        self.y_test = None                 # Labels for traditional models
        self.y_test_bilstm = None         # Labels for BiLSTM chunks
        self.y_test_arabert = None        # Labels for AraBERT chunks
        
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def _update_progress(self, current, total, message):
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    def _get_chunked_test_indices(self, chunked_df, indices_file, model_name):
        """Get or create test indices for chunked data using same methodology as training"""
        if os.path.exists(indices_file):
            with open(indices_file, 'rb') as f:
                saved_data = pickle.load(f)
                if (saved_data['total_size'] == len(chunked_df) and 
                    saved_data['random_state'] == self.random_state):
                    print(f"Using saved {model_name} chunked test indices from {indices_file}")
                    return saved_data['train_indices'], saved_data['test_indices']
        
        # Recreate the exact same split as in training:
        # 1. First split: 80% train, 20% devtest
        # 2. Second split: 50% dev, 50% test (from the 20% devtest)
        
        chunked_labels = chunked_df['encoded_label'].values
        
        # First split: train vs devtest (same as training)
        train_indices, devtest_indices = train_test_split(
            range(len(chunked_df)), 
            test_size=0.2, 
            stratify=chunked_labels, 
            random_state=self.random_state
        )
        
        # Second split: dev vs test (same as training)
        devtest_labels = chunked_labels[devtest_indices]
        dev_indices_relative, test_indices_relative = train_test_split(
            range(len(devtest_indices)), 
            test_size=0.5, 
            stratify=devtest_labels, 
            random_state=self.random_state
        )
        
        # Convert relative indices back to absolute indices
        test_indices = [devtest_indices[i] for i in test_indices_relative]
        
        # Save for consistency
        save_data = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'total_size': len(chunked_df),
            'random_state': self.random_state
        }
        
        with open(indices_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Created and saved {model_name} chunked test indices to {indices_file}")
        return train_indices, test_indices
    
    def prepare_test_data(self):
        # Load traditional classification data (non-chunked)
        df_classification = self.processor.load_data_classification()
        
        # Get traditional model test indices
        train_indices, test_indices = self._get_or_create_test_indices(
            len(df_classification), 
            df_classification['category']
        )
        
        self.X_test_classification = df_classification['processed_text_classification'].iloc[test_indices].reset_index(drop=True)
        self.y_test = df_classification['category'].iloc[test_indices].reset_index(drop=True)
        
        # Load chunked datasets (as used in training)
        df_bilstm_chunked = self.processor.load_data_bilstm_chunked(max_len=500)
        df_arabert_chunked = self.processor.load_data_arabert_chunked(max_len=500)
        
        # Get BiLSTM chunked test indices (recreate exact training split)
        bilstm_train_indices, bilstm_test_indices = self._get_chunked_test_indices(
            df_bilstm_chunked, self.bilstm_indices_file, "BiLSTM"
        )
        
        # Get AraBERT chunked test indices (recreate exact training split)
        arabert_train_indices, arabert_test_indices = self._get_chunked_test_indices(
            df_arabert_chunked, self.arabert_indices_file, "AraBERT"
        )
        
        # Extract test sets for chunked models
        self.X_test_bilstm_chunked = df_bilstm_chunked.iloc[bilstm_test_indices]['chunked_text'].reset_index(drop=True)
        self.y_test_bilstm = df_bilstm_chunked.iloc[bilstm_test_indices]['category'].reset_index(drop=True)
        
        self.X_test_arabert_chunked = df_arabert_chunked.iloc[arabert_test_indices]['chunked_text'].reset_index(drop=True)
        self.y_test_arabert = df_arabert_chunked.iloc[arabert_test_indices]['category'].reset_index(drop=True)
        
        print(f"Traditional models test set: {len(self.y_test)} samples")
        print(f"BiLSTM chunked test set: {len(self.y_test_bilstm)} chunks")
        print(f"AraBERT chunked test set: {len(self.y_test_arabert)} chunks")
        
        return self.X_test_classification, self.y_test
    
    def _get_or_create_test_indices(self, total_size, y):
        """Get existing test indices or create new ones and save them for traditional models"""
        if os.path.exists(self.indices_file):
            with open(self.indices_file, 'rb') as f:
                saved_data = pickle.load(f)
                if saved_data['total_size'] == total_size and saved_data['random_state'] == self.random_state:
                    print(f"Using saved test indices from {self.indices_file}")
                    return saved_data['train_indices'], saved_data['test_indices']
        
        # Create new split for traditional models (simple train/test)
        train_indices, test_indices = train_test_split(
            range(total_size), 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Save indices for consistency
        save_data = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'total_size': total_size,
            'random_state': self.random_state
        }
        
        with open(self.indices_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Created and saved new test indices to {self.indices_file}")
        return train_indices, test_indices
    
    def test_traditional_models(self):
        # Create a fresh classifier and train only on training data
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC  
        from sklearn.naive_bayes import MultinomialNB
        
        # Load full data and get train indices
        df_classification = self.processor.load_data_classification()
        train_indices, test_indices = self._get_or_create_test_indices(
            len(df_classification), 
            df_classification['category']
        )
        
        # Use only training data for fitting
        X_train_text = df_classification['processed_text_classification'].iloc[train_indices]
        y_train = df_classification['category'].iloc[train_indices]
        
        # Fit vectorizer and models on training data only
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        
        svm_classifier = LinearSVC(random_state=42, C=1.0)
        nb_classifier = MultinomialNB(alpha=0.01)
        
        svm_classifier.fit(X_train_tfidf, y_train)
        nb_classifier.fit(X_train_tfidf, y_train)
        
        total_samples = len(self.X_test_classification)
        
        # Test SVM
        svm_predictions = []
        svm_confidences = []
        svm_times = []
        
        for i, text in enumerate(self.X_test_classification):
            self._update_progress(i, total_samples, f"Testing SVM model ({i+1}/{total_samples})")
            start_time = time.time()
            
            X_new = tfidf_vectorizer.transform([text])
            pred = svm_classifier.predict(X_new)[0]
            
            # Get confidence from decision function
            decision_scores = svm_classifier.decision_function(X_new)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            conf = np.max(probabilities)
            
            inference_time = time.time() - start_time
            
            svm_predictions.append(pred)
            svm_confidences.append(conf)
            svm_times.append(inference_time)
        
        # Test Naive Bayes
        nb_predictions = []
        nb_confidences = []
        nb_times = []
        
        for i, text in enumerate(self.X_test_classification):
            self._update_progress(i, total_samples, f"Testing Naive Bayes model ({i+1}/{total_samples})")
            start_time = time.time()
            
            X_new = tfidf_vectorizer.transform([text])
            pred = nb_classifier.predict(X_new)[0]
            
            # Get confidence from predict_proba
            probabilities = nb_classifier.predict_proba(X_new)[0]
            conf = np.max(probabilities)
            
            inference_time = time.time() - start_time
            
            nb_predictions.append(pred)
            nb_confidences.append(conf)
            nb_times.append(inference_time)
        
        self.results['SVM'] = {
            'predictions': svm_predictions,
            'confidences': svm_confidences,
            'times': svm_times,
            'accuracy': accuracy_score(self.y_test, svm_predictions),
            'f1_score': f1_score(self.y_test, svm_predictions, average='weighted')
        }
        
        self.results['Naive_Bayes'] = {
            'predictions': nb_predictions,
            'confidences': nb_confidences,
            'times': nb_times,
            'accuracy': accuracy_score(self.y_test, nb_predictions),
            'f1_score': f1_score(self.y_test, nb_predictions, average='weighted')
        }
    
    def test_bilstm_model(self):
        try:
            print("Testing BiLSTM with proper chunked data (same as training)")
            
            bilstm = BiLSTMPredictor()
            
            # Use the chunked test data (already as lists)
            chunked_texts = list(self.X_test_bilstm_chunked)
            total_samples = len(chunked_texts)
            batch_size = 32
            
            self._update_progress(0, total_samples, "Starting BiLSTM batch prediction on chunked data...")
            
            start_time = time.time()
            
            # Process in batches with progress updates
            all_predictions = []
            all_confidences = []
            
            for i in range(0, len(chunked_texts), batch_size):
                batch_end = min(i + batch_size, len(chunked_texts))
                batch_texts = chunked_texts[i:batch_end]
                
                # Update progress for this batch
                self._update_progress(
                    batch_end, 
                    total_samples, 
                    f"Processing BiLSTM batch {(i//batch_size)+1}/{(total_samples-1)//batch_size + 1} ({batch_end}/{total_samples} chunks)"
                )
                
                # Process this batch
                batch_predictions, batch_confidences = bilstm.predict_batch_chunked_with_confidence(
                    batch_texts, batch_size=batch_size
                )
                
                all_predictions.extend(batch_predictions)
                all_confidences.extend(batch_confidences)
            
            total_time = time.time() - start_time
            avg_time_per_chunk = total_time / total_samples
            all_times = [avg_time_per_chunk] * total_samples
            
            self._update_progress(total_samples, total_samples, f"BiLSTM completed: {total_samples} chunks in {total_time:.2f}s")
            
            self.results['BiLSTM'] = {
                'predictions': all_predictions,
                'confidences': all_confidences,
                'times': all_times,
                'accuracy': accuracy_score(self.y_test_bilstm, all_predictions),
                'f1_score': f1_score(self.y_test_bilstm, all_predictions, average='weighted')
            }
            
        except Exception as e:
            print(f"BiLSTM Error: {str(e)}")
            self.results['BiLSTM'] = {'error': str(e)}
    
    def test_arabert_model(self):
        try:
            print("Testing AraBERT with proper chunked data (same as training)")
            
            arabert = AraBERTPredictor()
            
            # Use the chunked test data (already as lists)
            chunked_texts = list(self.X_test_arabert_chunked)
            total_samples = len(chunked_texts)
            batch_size = 32
            
            self._update_progress(0, total_samples, "Starting AraBERT batch prediction on chunked data...")
            
            start_time = time.time()
            
            # Process in batches with progress updates
            all_predictions = []
            all_confidences = []
            
            for i in range(0, len(chunked_texts), batch_size):
                batch_end = min(i + batch_size, len(chunked_texts))
                batch_texts = chunked_texts[i:batch_end]
                
                # Update progress for this batch
                self._update_progress(
                    batch_end, 
                    total_samples, 
                    f"Processing AraBERT batch {(i//batch_size)+1}/{(total_samples-1)//batch_size + 1} ({batch_end}/{total_samples} chunks)"
                )
                
                # Process this batch
                batch_predictions, batch_confidences = arabert.predict_batch_chunked_with_confidence(
                    batch_texts, batch_size=batch_size
                )
                
                all_predictions.extend(batch_predictions)
                all_confidences.extend(batch_confidences)
            
            total_time = time.time() - start_time
            avg_time_per_chunk = total_time / total_samples
            all_times = [avg_time_per_chunk] * total_samples
            
            self._update_progress(total_samples, total_samples, f"AraBERT completed: {total_samples} chunks in {total_time:.2f}s")
            
            self.results['AraBERT'] = {
                'predictions': all_predictions,
                'confidences': all_confidences,
                'times': all_times,
                'accuracy': accuracy_score(self.y_test_arabert, all_predictions),
                'f1_score': f1_score(self.y_test_arabert, all_predictions, average='weighted')
            }
            
        except Exception as e:
            print(f"AraBERT Error: {str(e)}")
            self.results['AraBERT'] = {'error': str(e)}
    
    def run_comprehensive_test(self):
        self.prepare_test_data()
        self.test_traditional_models()
        self.test_bilstm_model()
        self.test_arabert_model()
        self.generate_insights()
        
        return self.results
    
    def generate_insights(self):
        summary_data = []
        for model_name, result in self.results.items():
            if result is not None and isinstance(result, dict) and 'error' not in result:
                # Only process successful results that have the required keys
                if all(key in result for key in ['accuracy', 'f1_score', 'confidences', 'times', 'predictions']):
                    summary_data.append({
                        'Model': model_name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'F1-Score': f"{result['f1_score']:.4f}",
                        'Avg Confidence': f"{np.mean(result['confidences']):.4f}",
                        'Avg Time (s)': f"{np.mean(result['times']):.4f}",
                        'Total Samples': len(result['predictions'])
                    })
                else:
                    print(f"Warning: {model_name} result missing required keys: {result.keys()}")
            elif isinstance(result, dict) and 'error' in result:
                print(f"Skipping {model_name} due to error: {result['error']}")
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def get_confusion_matrices(self):
        matrices = {}
        for model_name, result in self.results.items():
            if result is not None:
                cm = confusion_matrix(self.y_test, result['predictions'])
                matrices[model_name] = cm
        return matrices
    
    def get_classification_reports(self):
        reports = {}
        for model_name, result in self.results.items():
            if result is not None:
                report = classification_report(self.y_test, result['predictions'], 
                                             output_dict=True)
                reports[model_name] = report
        return reports

def main():
    tester = ModelTester(random_state=42, test_size=0.2)
    results = tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 