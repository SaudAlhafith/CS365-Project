#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ..preprocessing.text_preprocessing import KalimatCorpusProcessor

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        device = src.device
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        _, (hidden, cell) = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

class Seq2SeqSummarizer:
    def __init__(self, model_path="arabic_summarizer.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = KalimatCorpusProcessor()
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        # Model hyperparameters (same as training)
        self.max_text_len = 128
        self.max_summary_len = 32
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.vocab_size = None
        
        self._setup_tokenizer()
        self._load_model()
    
    def _setup_tokenizer(self):
        """Setup tokenizer - in practice, this should be loaded from saved file"""
        # This is a placeholder - in practice, you'd load the tokenizer used during training
        # For now, we'll create a basic setup
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            self.tokenizer = Tokenizer(filters='', oov_token=None)
            # In practice, load the fitted tokenizer from training
            print("Warning: Using placeholder tokenizer. Load the actual training tokenizer for proper inference.")
        except ImportError:
            print("TensorFlow not available. Cannot create tokenizer.")
    
    def _load_model(self):
        """Load the trained Seq2Seq model"""
        try:
            # In practice, you'd need to know the vocab size from training
            # For now, using a default value
            self.vocab_size = 20000  # This should match training vocab size
            
            self.model = Seq2Seq(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading Seq2Seq model: {str(e)}")
            self.model = None
    
    def generate_summary(self, text, max_length=128):
        """Generate summary for given text"""
        if self.model is None or self.tokenizer is None:
            return "Model or tokenizer not available"
        
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Preprocess text
            text = self.processor.normalize_arabic_summarization(text)
            
            # Convert to sequence (placeholder implementation)
            # In practice, use the same tokenizer from training
            text_seq = self.tokenizer.texts_to_sequences([text])[0] if hasattr(self.tokenizer, 'texts_to_sequences') else [1, 2, 3]
            text_padded = pad_sequences([text_seq], maxlen=self.max_text_len, padding='post')
            text_tensor = torch.tensor(text_padded, dtype=torch.long).to(self.device)
            
            # Get special token indices
            sos_idx = getattr(self.tokenizer.word_index, '<sos>', 2) if hasattr(self.tokenizer, 'word_index') else 2
            eos_idx = getattr(self.tokenizer.word_index, '<eos>', 3) if hasattr(self.tokenizer, 'word_index') else 3
            
            decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                _, (hidden, cell) = self.model.encoder(text_tensor)
            
            summary = []
            for _ in range(max_length):
                output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)
                predicted = output.argmax(2)
                pred_idx = predicted.item()
                
                if pred_idx == eos_idx:
                    break
                
                summary.append(pred_idx)
                decoder_input = predicted
            
            # Convert indices back to words
            if hasattr(self.tokenizer, 'word_index'):
                idx2word = {v: k for k, v in self.tokenizer.word_index.items()}
                summary_words = [idx2word.get(idx, '') for idx in summary]
                return ' '.join(summary_words)
            else:
                return "Summary generated but tokenizer not properly loaded"
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def is_available(self):
        """Check if model is available for inference"""
        return self.model is not None

def main():
    summarizer = Seq2SeqSummarizer()
    
    if summarizer.is_available():
        test_text = "هذا نص تجريبي باللغة العربية لاختبار نموذج التلخيص الآلي باستخدام الشبكات العصبية"
        summary = summarizer.generate_summary(test_text)
        print(f"Original: {test_text}")
        print(f"Summary: {summary}")
    else:
        print("Seq2Seq summarizer not available")

if __name__ == "__main__":
    main() 