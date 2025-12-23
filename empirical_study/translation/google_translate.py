"""Google Translate implementation."""

import time
from typing import List, Optional

from loguru import logger

from .base import BaseTranslator, get_language_code


class GoogleTranslator(BaseTranslator):
    """
    Google Translate translator using googletrans library.

    Note: Uses the unofficial googletrans library. For production use,
    consider using the official Google Cloud Translation API.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(source_lang, target_lang)
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._translator = None

    def _get_translator(self):
        """Lazy initialization of translator."""
        if self._translator is None:
            try:
                from googletrans import Translator

                self._translator = Translator(
                    service_urls=["translate.google.com"],
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "googletrans package not installed. "
                    "Install with: pip install googletrans==4.0.0-rc1"
                )
        return self._translator

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text using Google Translate.

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

        # Convert to Google's language codes
        src_code = get_language_code(source, "google")
        tgt_code = get_language_code(target, "google")

        for attempt in range(self.retry_count):
            try:
                result = translator.translate(
                    text, src=src_code, dest=tgt_code
                )
                return result.text

            except Exception as e:
                logger.warning(
                    f"Google Translate error (attempt {attempt + 1}): {e}"
                )
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    # Reinitialize translator on error
                    self._translator = None

        logger.error(f"Google Translate failed after {self.retry_count} attempts")
        return text  # Return original on failure

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate batch of texts.

        Google Translate doesn't have native batch support in googletrans,
        so we translate one by one with rate limiting.
        """
        results = []
        for i, text in enumerate(texts):
            translated = self.translate(text, source, target)
            results.append(translated)

            # Rate limiting
            if i < len(texts) - 1:
                time.sleep(0.5)  # 500ms between requests

            if (i + 1) % 10 == 0:
                logger.info(f"Translated {i + 1}/{len(texts)} texts")

        return results


class GoogleCloudTranslator(BaseTranslator):
    """
    Google Cloud Translation API translator.

    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable
    to be set with path to service account key.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        project_id: Optional[str] = None,
    ):
        super().__init__(source_lang, target_lang)
        self.project_id = project_id
        self._client = None

    def _get_client(self):
        """Lazy initialization of Google Cloud client."""
        if self._client is None:
            try:
                from google.cloud import translate_v2 as translate

                self._client = translate.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-translate package not installed. "
                    "Install with: pip install google-cloud-translate"
                )
        return self._client

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text using Google Cloud Translation API.

        Args:
            text: Text to translate
            source: Source language code
            target: Target language code

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        client = self._get_client()

        src_code = get_language_code(source, "google")
        tgt_code = get_language_code(target, "google")

        try:
            result = client.translate(
                text,
                source_language=src_code,
                target_language=tgt_code,
            )
            return result["translatedText"]

        except Exception as e:
            logger.error(f"Google Cloud Translation error: {e}")
            return text

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        """
        Translate batch using Google Cloud API.

        Google Cloud supports batch translation natively.
        """
        if not texts:
            return []

        client = self._get_client()

        src_code = get_language_code(source, "google")
        tgt_code = get_language_code(target, "google")

        try:
            # Google Cloud can handle batches
            results = client.translate(
                texts,
                source_language=src_code,
                target_language=tgt_code,
            )
            return [r["translatedText"] for r in results]

        except Exception as e:
            logger.error(f"Google Cloud batch translation error: {e}")
            # Fall back to one-by-one
            return [self.translate(t, source, target) for t in texts]
