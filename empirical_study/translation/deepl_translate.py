"""DeepL Translate implementation."""

import os
import time
from typing import List, Optional

from loguru import logger

from .base import BaseTranslator, get_language_code


class DeepLTranslator(BaseTranslator):
    """
    DeepL Translator using the official DeepL API.

    Requires DEEPL_API_KEY environment variable to be set.
    Known for high-quality translations that preserve style and nuance.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        api_key: Optional[str] = None,
        formality: str = "default",  # 'default', 'more', 'less'
    ):
        super().__init__(source_lang, target_lang)
        self.api_key = api_key or os.environ.get("DEEPL_API_KEY")
        self.formality = formality
        self._translator = None

    def _get_translator(self):
        """Lazy initialization of DeepL translator."""
        if self._translator is None:
            if not self.api_key:
                raise ValueError(
                    "DeepL API key not provided. Set DEEPL_API_KEY "
                    "environment variable or pass api_key parameter."
                )

            try:
                import deepl

                self._translator = deepl.Translator(self.api_key)
                logger.info("DeepL translator initialized")
            except ImportError:
                raise ImportError(
                    "deepl package not installed. "
                    "Install with: pip install deepl"
                )
        return self._translator

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text using DeepL API.

        Args:
            text: Text to translate
            source: Source language code
            target: Target language code

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        translator = self._get_translator()

        # Convert to DeepL's language codes
        src_code = get_language_code(source, "deepl")
        tgt_code = get_language_code(target, "deepl")

        try:
            # DeepL uses target_lang, source_lang is optional
            result = translator.translate_text(
                text,
                source_lang=src_code,
                target_lang=tgt_code,
                formality=self.formality if self.formality != "default" else None,
            )
            return result.text

        except Exception as e:
            logger.error(f"DeepL translation error: {e}")
            return text

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate batch of texts using DeepL.

        DeepL API supports batch translation natively.
        """
        if not texts:
            return []

        translator = self._get_translator()

        src_code = get_language_code(source, "deepl")
        tgt_code = get_language_code(target, "deepl")

        try:
            # DeepL handles batches efficiently
            results = translator.translate_text(
                texts,
                source_lang=src_code,
                target_lang=tgt_code,
                formality=self.formality if self.formality != "default" else None,
            )

            # Results may be a single result or list
            if isinstance(results, list):
                return [r.text for r in results]
            else:
                return [results.text]

        except Exception as e:
            logger.error(f"DeepL batch translation error: {e}")
            # Fall back to one-by-one
            return [self.translate(t, source, target) for t in texts]

    def get_usage(self) -> dict:
        """Get API usage statistics."""
        translator = self._get_translator()
        try:
            usage = translator.get_usage()
            return {
                "character_count": usage.character.count,
                "character_limit": usage.character.limit,
                "remaining": usage.character.limit - usage.character.count,
            }
        except Exception as e:
            logger.warning(f"Could not get DeepL usage: {e}")
            return {}

    def get_supported_languages(self) -> dict:
        """Get supported languages."""
        translator = self._get_translator()
        try:
            source_langs = translator.get_source_languages()
            target_langs = translator.get_target_languages()
            return {
                "source": [(l.code, l.name) for l in source_langs],
                "target": [(l.code, l.name) for l in target_langs],
            }
        except Exception as e:
            logger.warning(f"Could not get supported languages: {e}")
            return {"source": [], "target": []}


class DeepLFreeTranslator(DeepLTranslator):
    """
    DeepL Free API translator.

    Uses the free tier of DeepL with rate limits.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rate_limit_delay = 1.0  # Seconds between requests

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate with rate limiting for free tier.
        """
        results = []
        for i, text in enumerate(texts):
            translated = self.translate(text, source, target)
            results.append(translated)

            # Rate limiting for free tier
            if i < len(texts) - 1:
                time.sleep(self.rate_limit_delay)

            if (i + 1) % 10 == 0:
                logger.info(f"DeepL translated {i + 1}/{len(texts)} texts")

        return results
