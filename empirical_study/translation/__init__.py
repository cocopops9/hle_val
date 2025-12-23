"""Translation modules for empirical study."""

from .base import BaseTranslator, TranslationResult
from .google_translate import GoogleTranslator
from .deepl_translate import DeepLTranslator
from .nllb_translate import NLLBTranslator

__all__ = [
    "BaseTranslator",
    "TranslationResult",
    "GoogleTranslator",
    "DeepLTranslator",
    "NLLBTranslator",
]
