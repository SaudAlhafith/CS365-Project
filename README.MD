# CS365 Arabic Text Classification Project

**Arabic NLP using Traditional vs Modern Approaches**  
Course: CS365 - Natural Language Processing

## 🎯 Project Overview

This project compares traditional machine learning vs modern deep learning approaches for Arabic text classification using the KalimatCorpus-2.0 dataset (Omani news articles).

**Categories:** Culture, Economy, International, Local, Religion, Sports

## 🤖 Results Summary

| Model | Accuracy | Speed |
|-------|----------|-------|
| **AraBERT** | **94.48%** | Slow (GPU recommended) |
| **SVM + TF-IDF** | **92%** | Very Fast |
| **BiLSTM** | **88.78%** | Medium |
| **Naive Bayes** | **88%** | Very Fast |

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download KalimatCorpus-2.0 from [SourceForge](https://sourceforge.net/projects/kalimat/files/kalimat/document-collection/) and extract to:
```
data/KalimatCorpus-2.0/
├── articlesInternational/
├── articlesEconomy/
├── articlesCulture/
├── articlesLocal/
├── articlesReligion/
└── articlesSports/
```

### 3. Run the Project
**Main Notebook:** Open `CS365-Project-Phase2-saud.ipynb` in Jupyter and run all cells

**Web Demo:** 
```bash
streamlit run app.py
```

## 📁 Project Files

```
├── CS365-Project-Phase2-saud.ipynb    # Main notebook with all code
├── app.py                             # Streamlit web demo
├── bilstm_best_model.pth             # Trained BiLSTM model
├── results/checkpoint-288-best/       # AraBERT model
├── src/                              # Source code modules
├── data/KalimatCorpus-2.0/           # Dataset (download manually)
├── requirements.txt                   # Python dependencies
└── README.MD                         # This file
```

## 🔧 What the Notebook Contains

1. **Data Loading & Preprocessing** - Arabic text cleaning and normalization
2. **Traditional Models** - SVM, Naive Bayes with TF-IDF + N-gram text generation
3. **Modern Models** - BiLSTM and AraBERT training and evaluation
4. **Comparison** - Performance analysis across all approaches
5. **Interactive Demo** - Test all models on sample texts

## 🎮 Web Demo

Run `streamlit run app.py` for an interactive web interface where you can:
- Test all models with custom Arabic text
- Compare predictions side-by-side
- See confidence scores

## 📚 Key Learning Points

- Arabic text preprocessing challenges
- Traditional ML: TF-IDF, SVM, Naive Bayes
- Modern NLP: BiLSTM, Transformers, BERT
- Model comparison and evaluation
- Real-world Arabic NLP pipeline

## 🆘 Troubleshooting

**Dataset issues:** Make sure KalimatCorpus-2.0 is extracted correctly in `data/` folder  
**CUDA errors:** Set device to 'cpu' if no GPU available  
**Memory issues:** Reduce batch sizes in training sections

---
*Student project for CS365 - Educational purposes*
