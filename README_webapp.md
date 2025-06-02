# 🔤 Arabic Text Classification Web Demo

Interactive web application for CS365 Arabic Text Classification project featuring real-time model comparison and text generation.

## ✨ Features

### 🔍 Text Classification
- **Real-time prediction** with Traditional ML (SVM, Naive Bayes), BiLSTM, and AraBERT models
- **Sample texts** for quick testing across all 6 categories
- **Custom text input** with Arabic text support
- **Performance metrics** including inference time and confidence scores
- **Interactive comparison** with side-by-side results

### 📊 Model Comparison Dashboard
- **Comprehensive testing** on predefined test cases
- **Accuracy visualization** with interactive charts
- **Speed comparison** showing inference times
- **Detailed results table** with all predictions

### ✍️ Text Generation
- **N-gram based** Arabic text generation
- **Customizable length** and optional start word
- **Generated text classification** to test model consistency
- **Arabic text display** with proper RTL formatting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- All model files in place:
  - `bilstm_best_model.pth`
  - `results/checkpoint-288-best/`
  - `processed_classification_data.csv`
  - `processed_ngram_data.csv`

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

### Alternative: Run with specific port
```bash
streamlit run app.py --server.port 8501
```

## 📱 Usage

1. **Open your browser** to `http://localhost:8501`
2. **Select models** to compare in the sidebar
3. **Choose a tab**:
   - **Text Classification**: Input text and get real-time predictions
   - **Model Comparison**: Run comprehensive tests and view analytics  
   - **Text Generation**: Generate and classify Arabic text

## 🎯 Categories Supported
- **CULTURE** - Cultural and heritage content
- **ECONOMY** - Economic and financial news
- **INTERNATIONAL** - International affairs and global news
- **LOCAL** - Local and domestic news
- **RELIGION** - Religious content and teachings
- **SPORTS** - Sports news and events

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit (Pure Python)
- **Models**: SVM, Naive Bayes, BiLSTM, AraBERT
- **Visualization**: Plotly for interactive charts
- **Caching**: Streamlit's built-in caching for model loading

### Performance
- **Model Loading**: Cached on first run
- **Real-time Inference**: Sub-second response times
- **Batch Processing**: Progress tracking for comprehensive tests

## 🌐 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from repository

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📋 Requirements
- Streamlit >= 1.28.0
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- Plotly >= 5.15.0
- All other dependencies in `requirements.txt`

## 🔧 Troubleshooting

### Common Issues
1. **Model files not found**: Ensure all model files are in correct locations
2. **CUDA out of memory**: Models will automatically fallback to CPU
3. **Arabic text display**: Modern browsers support Arabic text natively

### Performance Tips
- Models are cached on first load
- Use smaller batch sizes for memory-constrained environments
- GPU acceleration available if CUDA is installed

## 📊 Demo Screenshots

The web app includes:
- Clean, modern interface with Arabic text support
- Interactive charts and visualizations
- Real-time processing with progress indicators
- Responsive design for different screen sizes

## 🎉 Next Steps

- **Share the demo** with stakeholders
- **Collect feedback** from users
- **Deploy to cloud** for wider accessibility
- **Add more features** based on requirements

---

**CS365 Project - Phase 2**  
Interactive Arabic Text Classification Demo 