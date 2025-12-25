"""Data loading modules for empirical study datasets."""

from .dataset_loader import DatasetLoader
from .hate_speech import (
    HateXplainLoader,
    ImplicitHateLoader,
    SBICLoader,
    NonHateLoader,
    load_hate_speech_data,
)
from .sarcasm import SemEvalSarcasmLoader, ISarcasmLoader
from .dialectal import SapDialectalLoader

__all__ = [
    "DatasetLoader",
    # Hate speech loaders
    "HateXplainLoader",      # Explicit hate (clear slurs/threats)
    "ImplicitHateLoader",    # SALT-NLP implicit hate
    "SBICLoader",            # SBIC implicit hate (truly implicit - no slurs)
    "NonHateLoader",         # Non-hate for balancing
    "load_hate_speech_data", # Convenience function
    # Sarcasm loaders
    "SemEvalSarcasmLoader",
    "ISarcasmLoader",
    # Dialectal loader
    "SapDialectalLoader",
]
