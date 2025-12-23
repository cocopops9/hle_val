"""Base detector class for all detection models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


@dataclass
class PredictionResult:
    """Stores prediction result for a single sample."""

    sample_id: str
    text: str
    predicted_label: int
    confidence: float
    probabilities: Dict[int, float]
    true_label: Optional[int] = None


class BaseDetector(ABC):
    """Abstract base class for all detection models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """
        Make predictions on a list of texts.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of PredictionResult objects.
        """
        pass

    def predict_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        Make predictions in batches.

        Args:
            texts: List of text strings.
            sample_ids: Optional list of sample IDs.

        Returns:
            List of PredictionResult objects.
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(texts))]

        all_results = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = sample_ids[i : i + self.batch_size]

            batch_results = self.predict(batch_texts)

            # Update sample IDs
            for result, sid in zip(batch_results, batch_ids):
                result.sample_id = sid

            all_results.extend(batch_results)

        return all_results

    def get_predictions_array(
        self, results: List[PredictionResult]
    ) -> np.ndarray:
        """Convert results to numpy array of predictions."""
        return np.array([r.predicted_label for r in results])

    def get_probabilities_array(
        self, results: List[PredictionResult], class_label: int = 1
    ) -> np.ndarray:
        """Get probabilities for a specific class."""
        return np.array([r.probabilities.get(class_label, 0.0) for r in results])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"


class TransformerDetector(BaseDetector):
    """Base class for transformer-based detectors."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        super().__init__(model_name, device, batch_size)
        self.max_length = max_length

    def load_model(self) -> None:
        """Load transformer model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make predictions using transformer model."""
        if self.model is None:
            self.load_model()

        results = []

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Convert to predictions
            predictions = torch.argmax(logits, dim=-1)

            for i, text in enumerate(texts):
                pred_label = predictions[i].item()
                prob_dict = {
                    j: probs[i, j].item() for j in range(probs.shape[1])
                }
                confidence = prob_dict[pred_label]

                result = PredictionResult(
                    sample_id=f"sample_{i}",
                    text=text,
                    predicted_label=pred_label,
                    confidence=confidence,
                    probabilities=prob_dict,
                )
                results.append(result)

        return results
