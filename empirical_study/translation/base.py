"""Base translator class and utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class TranslationResult:
    """Stores result of a translation operation."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    back_translated_text: Optional[str] = None
    translator_name: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def is_round_trip(self) -> bool:
        """Check if back-translation was performed."""
        return self.back_translated_text is not None

    def text_changed(self) -> bool:
        """Check if round-trip translation changed the text."""
        if not self.is_round_trip:
            return False
        return self.original_text.lower().strip() != self.back_translated_text.lower().strip()


class BaseTranslator(ABC):
    """Abstract base class for translation systems."""

    def __init__(self, source_lang: str = "en", target_lang: str = "es"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.name = self.__class__.__name__

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source: Source language code
            target: Target language code

        Returns:
            Translated text
        """
        pass

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate a batch of texts.

        Default implementation translates one by one.
        Subclasses may override for batch API support.
        """
        return [self.translate(text, source, target) for text in texts]

    def round_trip_translate(
        self,
        text: str,
        source: str = "en",
        target: str = "es",
    ) -> TranslationResult:
        """
        Perform round-trip translation (source -> target -> source).

        Args:
            text: Original text
            source: Source language code
            target: Target language code

        Returns:
            TranslationResult with original, translated, and back-translated text
        """
        # Forward translation
        translated = self.translate(text, source, target)

        # Back translation
        back_translated = self.translate(translated, target, source)

        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source,
            target_language=target,
            back_translated_text=back_translated,
            translator_name=self.name,
        )

    def round_trip_batch(
        self,
        texts: List[str],
        source: str = "en",
        target: str = "es",
    ) -> List[TranslationResult]:
        """
        Perform round-trip translation on a batch of texts.

        Args:
            texts: List of texts to translate
            source: Source language code
            target: Target language code

        Returns:
            List of TranslationResult objects
        """
        logger.info(
            f"Round-trip translating {len(texts)} texts "
            f"({source} -> {target} -> {source}) using {self.name}"
        )

        # Forward translation
        translated = self.translate_batch(texts, source, target)

        # Back translation
        back_translated = self.translate_batch(translated, target, source)

        results = []
        for orig, trans, back in zip(texts, translated, back_translated):
            result = TranslationResult(
                original_text=orig,
                translated_text=trans,
                source_language=source,
                target_language=target,
                back_translated_text=back,
                translator_name=self.name,
            )
            results.append(result)

        return results

    def __repr__(self) -> str:
        return f"{self.name}(source={self.source_lang}, target={self.target_lang})"


# Language code mappings for different translation services
LANGUAGE_CODES = {
    "google": {
        "en": "en",
        "es": "es",
        "nl": "nl",
        "it": "it",
        "de": "de",
        "fr": "fr",
        "pt": "pt",
        "zh": "zh-CN",
        "ja": "ja",
        "ko": "ko",
        "ar": "ar",
    },
    "deepl": {
        "en": "EN",
        "es": "ES",
        "nl": "NL",
        "it": "IT",
        "de": "DE",
        "fr": "FR",
        "pt": "PT-PT",
        "zh": "ZH",
        "ja": "JA",
        "ko": "KO",
    },
    "nllb": {
        "en": "eng_Latn",
        "es": "spa_Latn",
        "nl": "nld_Latn",
        "it": "ita_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "pt": "por_Latn",
        "zh": "zho_Hans",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "ar": "arb_Arab",
    },
}


def get_language_code(lang: str, service: str) -> str:
    """
    Get language code for a specific translation service.

    Args:
        lang: Standard language code (e.g., 'en', 'es')
        service: Translation service name

    Returns:
        Service-specific language code
    """
    service_codes = LANGUAGE_CODES.get(service.lower(), {})
    return service_codes.get(lang.lower(), lang)
