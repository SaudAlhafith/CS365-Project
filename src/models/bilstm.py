#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.text_preprocessing import KalimatCorpusProcessor

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, pad_idx):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=max(2, num_layers//2), dropout=0.4, bidirectional=True, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=max(2, num_layers//2), dropout=0.4, bidirectional=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        x = self.embedding(x)
        x = self.dropout1(x)

        packed1 = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out1, _ = self.lstm1(packed1)
        lstm1_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out1, batch_first=True)

        lstm1_out = self.norm1(lstm1_out)
        fc1_out = self.fc1(lstm1_out)

        packed2 = nn.utils.rnn.pack_padded_sequence(fc1_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out2, (hidden, _) = self.lstm2(packed2)
        lstm2_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out2, batch_first=True)
        
        lstm2_out = self.norm2(lstm2_out + lstm1_out)

        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout2(out)
        return self.fc2(out)

class BiLSTMPredictor:
    def __init__(self, model_path="bilstm_classifier_model.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = KalimatCorpusProcessor()
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.model_path = model_path
        self._setup_vocab_and_labels()
        self._load_model()
    
    def _setup_vocab_and_labels(self):
        df = self.processor.load_data_classification()
        
        tokenized_text = [text.split() for text in df['processed_text_classification']]
        word_counts = Counter(word for article in tokenized_text for word in article)
        
        self.vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.items())}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df['category'])
        
    def _load_model(self):
        vocab_size = len(self.vocab)
        embedding_dim = 300
        hidden_dim = 256
        num_layers = 6
        num_classes = len(self.label_encoder.classes_)
        pad_idx = self.vocab['<PAD>']
        
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            pad_idx=pad_idx
        ).to(self.device)
        
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def _encode_text(self, text, max_len=500):
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
        
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [self.vocab['<PAD>']] * (max_len - len(tokens))
            
        length = len([t for t in tokens if t != self.vocab['<PAD>']])
        return tokens, length
    
    def predict(self, text):
        processed_text = self.processor.preprocess_text_classification(text)
        encoded_text, length = self._encode_text(processed_text)
        
        x = torch.tensor([encoded_text], dtype=torch.long).to(self.device)
        lengths = torch.tensor([length], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, lengths)
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def predict_with_confidence(self, text):
        processed_text = self.processor.preprocess_text_classification(text)
        encoded_text, length = self._encode_text(processed_text)
        
        x = torch.tensor([encoded_text], dtype=torch.long).to(self.device)
        lengths = torch.tensor([length], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, lengths)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        return predicted_label, float(confidence)

    def predict_batch_with_confidence(self, texts, batch_size=32):
        """Predict multiple texts in batches for faster inference"""
        all_predictions = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_encoded = []
            batch_lengths = []
            
            # Encode all texts in batch
            for text in batch_texts:
                processed_text = self.processor.preprocess_text_classification(text)
                encoded_text, length = self._encode_text(processed_text)
                batch_encoded.append(encoded_text)
                batch_lengths.append(length)
            
            # Convert to tensors
            x = torch.tensor(batch_encoded, dtype=torch.long).to(self.device)
            lengths = torch.tensor(batch_lengths, dtype=torch.long).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(x, lengths)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                confidences = probabilities[range(len(predictions)), predictions]
                
                # Convert predictions to labels
                predicted_labels = self.label_encoder.inverse_transform(predictions)
                
                all_predictions.extend(predicted_labels)
                all_confidences.extend(confidences.astype(float))
        
        return all_predictions, all_confidences

    def predict_batch_chunked_with_confidence(self, chunked_texts, batch_size=32):
        """Predict multiple pre-chunked texts (integer lists) in batches for faster inference"""
        all_predictions = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(chunked_texts), batch_size):
            batch_chunks = chunked_texts[i:i + batch_size]
            
            # Convert to tensors (chunks are already encoded as integer lists)
            x = torch.tensor(batch_chunks, dtype=torch.long).to(self.device)
            lengths = torch.tensor([len([t for t in chunk if t != 0]) for chunk in batch_chunks], 
                                 dtype=torch.long).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(x, lengths)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                confidences = probabilities[range(len(predictions)), predictions]
                
                # Convert predictions to labels
                predicted_labels = self.label_encoder.inverse_transform(predictions)
                
                all_predictions.extend(predicted_labels)
                all_confidences.extend(confidences.astype(float))
        
        return all_predictions, all_confidences

def main():
    predictor = BiLSTMPredictor()
    
    test_texts = [
        "أطلقت وزارة الثقافة برنامجًا وطنيًا يهدف إلى إحياء التراث الشعبي",
        "شهدت الأسواق المالية ارتفاعًا ملحوظًا في قيمة الأسهم",
        "تمكن المنتخب الوطني من الفوز على نظيره الإيراني في مباراة مثيرة"
    ]
    
    for text in test_texts:
        prediction = predictor.predict(text)

if __name__ == "__main__":
    main() 