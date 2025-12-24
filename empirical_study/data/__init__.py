"""Data loading modules for empirical study datasets."""

from .dataset_loader import DatasetLoader
from .hate_speech import HateXplainLoader, ImplicitHateLoader, NonHateLoader, load_hate_speech_data
from .sarcasm import SemEvalSarcasmLoader, ISarcasmLoader
from .dialectal import SapDialectalLoader

__all__ = [
    "DatasetLoader",
    # Hate speech loaders
    "HateXplainLoader",      # Explicit hate (replaces HatEvalLoader)
    "ImplicitHateLoader",    # Implicit hate (replaces LatentHatredLoader)
    "NonHateLoader",         # Non-hate for balancing
    "load_hate_speech_data", # Convenience function
    # Sarcasm loaders
    "SemEvalSarcasmLoader",
    "ISarcasmLoader",
    # Dialectal loader
    "SapDialectalLoader",
]
