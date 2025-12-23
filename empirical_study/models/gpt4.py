"""GPT-4 zero-shot detector for hate speech and sarcasm detection."""

import json
import os
import time
from typing import Dict, List, Optional

from loguru import logger

from .base import BaseDetector, PredictionResult


class GPT4Detector(BaseDetector):
    """
    GPT-4 zero-shot detector for text classification.

    Uses OpenAI's GPT-4 API with task-specific prompts for
    hate speech and sarcasm detection.
    """

    def __init__(
        self,
        task: str = "hate_speech",  # 'hate_speech' or 'sarcasm'
        model_name: str = "gpt-4",
        device: Optional[str] = None,  # Not used for API
        batch_size: int = 10,  # Smaller batch for API rate limits
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(model_name, device, batch_size)
        self.task = task
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None

        # Task-specific prompts
        self.prompts = {
            "hate_speech": self._get_hate_speech_prompt(),
            "sarcasm": self._get_sarcasm_prompt(),
        }

    def _get_hate_speech_prompt(self) -> str:
        """Get zero-shot prompt for hate speech detection."""
        return """You are an expert at detecting hate speech in text. Analyze the following text and determine if it contains hate speech.

Hate speech is defined as content that:
- Attacks or demeans a group based on attributes like race, ethnicity, gender, religion, sexual orientation, or disability
- Contains slurs, threats, or dehumanizing language
- Promotes violence or discrimination against a group

Respond with a JSON object containing:
- "label": 1 if the text contains hate speech, 0 if it does not
- "confidence": a float between 0 and 1 indicating your confidence
- "reasoning": a brief explanation of your decision

Text to analyze: "{text}"

Respond only with the JSON object, no other text."""

    def _get_sarcasm_prompt(self) -> str:
        """Get zero-shot prompt for sarcasm detection."""
        return """You are an expert at detecting sarcasm and irony in text. Analyze the following text and determine if it is sarcastic.

Sarcasm typically involves:
- Saying the opposite of what is meant
- Using exaggeration or understatement for effect
- Expressing mockery or contempt through ironic statements
- A mismatch between literal meaning and intended meaning

Respond with a JSON object containing:
- "label": 1 if the text is sarcastic/ironic, 0 if it is not
- "confidence": a float between 0 and 1 indicating your confidence
- "reasoning": a brief explanation of your decision

Text to analyze: "{text}"

Respond only with the JSON object, no other text."""

    def load_model(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it to use GPT-4 detection."
                )

            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")

        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def _call_api(self, text: str) -> Dict:
        """Make a single API call with retry logic."""
        if self.client is None:
            self.load_model()

        prompt = self.prompts[self.task].format(text=text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that analyzes text for classification. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=200,
                )

                content = response.choices[0].message.content.strip()

                # Parse JSON response
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                result = json.loads(content)
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                logger.debug(f"Raw response: {content}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # Return default if all retries failed
        return {"label": 0, "confidence": 0.5, "reasoning": "API call failed"}

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make predictions using GPT-4 API."""
        results = []

        for i, text in enumerate(texts):
            api_result = self._call_api(text)

            label = int(api_result.get("label", 0))
            confidence = float(api_result.get("confidence", 0.5))
            reasoning = api_result.get("reasoning", "")

            # Create probability dict
            if label == 1:
                prob_dict = {0: 1 - confidence, 1: confidence}
            else:
                prob_dict = {0: confidence, 1: 1 - confidence}

            result = PredictionResult(
                sample_id=f"gpt4_{self.task}_{i}",
                text=text,
                predicted_label=label,
                confidence=confidence,
                probabilities=prob_dict,
            )
            result.metadata = {"reasoning": reasoning}
            results.append(result)

            # Small delay to avoid rate limiting
            if i < len(texts) - 1:
                time.sleep(0.1)

        return results

    def predict_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """Make predictions in smaller batches for API rate limits."""
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(texts))]

        all_results = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = sample_ids[i : i + self.batch_size]

            logger.info(
                f"Processing batch {i // self.batch_size + 1}/"
                f"{(len(texts) + self.batch_size - 1) // self.batch_size}"
            )

            batch_results = self.predict(batch_texts)

            for result, sid in zip(batch_results, batch_ids):
                result.sample_id = sid

            all_results.extend(batch_results)

            # Delay between batches
            if i + self.batch_size < len(texts):
                time.sleep(1)

        return all_results


class GPT4HateSpeechDetector(GPT4Detector):
    """GPT-4 detector specifically for hate speech."""

    def __init__(self, **kwargs):
        super().__init__(task="hate_speech", **kwargs)


class GPT4SarcasmDetector(GPT4Detector):
    """GPT-4 detector specifically for sarcasm."""

    def __init__(self, **kwargs):
        super().__init__(task="sarcasm", **kwargs)


def create_gpt4_detector(
    task: str = "hate_speech",
    model_name: str = "gpt-4",
    batch_size: int = 10,
) -> GPT4Detector:
    """
    Factory function to create GPT-4 detectors.

    Args:
        task: 'hate_speech' or 'sarcasm'
        model_name: OpenAI model name
        batch_size: Batch size for API calls

    Returns:
        Initialized GPT-4 detector
    """
    return GPT4Detector(task=task, model_name=model_name, batch_size=batch_size)
