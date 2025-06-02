"""
Arabic Text Processing and Classification Package
CS365 Project - Phase 2
"""

__version__ = "1.0.0"
__author__ = "CS365 Project"

from .preprocessing import KalimatCorpusProcessor
from .models import TraditionalClassifier, NGramGenerator, BiLSTMPredictor, AraBERTPredictor

__all__ = [
    'KalimatCorpusProcessor',
    'TraditionalClassifier', 
    'NGramGenerator',
    'BiLSTMPredictor',
    'AraBERTPredictor'
] 