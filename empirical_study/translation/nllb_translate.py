"""NLLB-200 (No Language Left Behind) translation implementation."""

from typing import List, Optional

import torch
from loguru import logger

from .base import BaseTranslator, get_language_code


class NLLBTranslator(BaseTranslator):
    """
    NLLB-200 Translator using Hugging Face Transformers.

    Meta's No Language Left Behind model supporting 200 languages.
    Runs locally, no API key required.

    Reference: Costa-jussà et al. (2022). No Language Left Behind:
    Scaling Human-Centered Machine Translation.
    """

    # Available model sizes
    MODELS = {
        "small": "facebook/nllb-200-distilled-600M",
        "medium": "facebook/nllb-200-1.3B",
        "large": "facebook/nllb-200-3.3B",
        "xl": "facebook/nllb-200-distilled-1.3B",
    }

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        model_size: str = "small",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        super().__init__(source_lang, target_lang)

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.MODELS.get(model_size, self.MODELS["small"])

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.batch_size = batch_size

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load NLLB model and tokenizer."""
        if self._model is not None:
            return

        logger.info(f"Loading NLLB model: {self.model_name}")

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            logger.info(f"NLLB model loaded on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers package not installed. "
                "Install with: pip install transformers"
            )

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text using NLLB-200.

        Args:
            text: Text to translate
            source: Source language code
            target: Target language code

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        self._load_model()

        # Convert to NLLB language codes
        src_code = get_language_code(source, "nllb")
        tgt_code = get_language_code(target, "nllb")

        try:
            # Set source language
            self._tokenizer.src_lang = src_code

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=self.max_length,
                    num_beams=5,
                    early_stopping=True,
                )

            # Decode
            translated = self._tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            return translated

        except Exception as e:
            logger.error(f"NLLB translation error: {e}")
            return text

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate batch of texts using NLLB.

        Supports efficient batch processing on GPU.
        """
        if not texts:
            return []

        self._load_model()

        src_code = get_language_code(source, "nllb")
        tgt_code = get_language_code(target, "nllb")

        all_translations = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                self._tokenizer.src_lang = src_code

                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                ).to(self.device)

                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                            tgt_code
                        ),
                        max_length=self.max_length,
                        num_beams=5,
                        early_stopping=True,
                    )

                translations = self._tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                all_translations.extend(translations)

            except Exception as e:
                logger.error(f"NLLB batch error: {e}")
                # Fall back to one-by-one for this batch
                for text in batch:
                    all_translations.append(self.translate(text, source, target))

            if (i + self.batch_size) % 50 == 0:
                logger.info(f"NLLB translated {i + self.batch_size}/{len(texts)} texts")

        return all_translations

    def get_supported_languages(self) -> List[str]:
        """Get list of supported NLLB language codes."""
        self._load_model()
        # Return common language codes
        return list(get_language_code.__code__.co_consts)


def create_translator(
    translator_type: str,
    source_lang: str = "en",
    target_lang: str = "es",
    **kwargs,
) -> BaseTranslator:
    """
    Factory function to create translators.

    Args:
        translator_type: One of 'google', 'deepl', 'nllb'
        source_lang: Source language code
        target_lang: Target language code
        **kwargs: Additional arguments for specific translators

    Returns:
        Initialized translator
    """
    from .google_translate import GoogleTranslator
    from .deepl_translate import DeepLTranslator

    translators = {
        "google": GoogleTranslator,
        "deepl": DeepLTranslator,
        "nllb": NLLBTranslator,
    }

    if translator_type.lower() not in translators:
        raise ValueError(
            f"Unknown translator type: {translator_type}. "
            f"Available: {list(translators.keys())}"
        )

    return translators[translator_type.lower()](
        source_lang=source_lang, target_lang=target_lang, **kwargs
    )
