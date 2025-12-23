"""Detection model modules for empirical study."""

from .base import BaseDetector
from .hate_speech import HateBERTDetector, RoBERTaHatEvalDetector
from .sarcasm import TwitterRoBERTaIronyDetector
from .gpt4 import GPT4Detector

__all__ = [
    "BaseDetector",
    "HateBERTDetector",
    "RoBERTaHatEvalDetector",
    "TwitterRoBERTaIronyDetector",
    "GPT4Detector",
]
