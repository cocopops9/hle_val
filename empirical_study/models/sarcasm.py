"""Sarcasm/Irony detection models: Twitter-RoBERTa-Irony."""

from typing import Dict, List, Optional

import torch
from loguru import logger

from .base import PredictionResult, TransformerDetector


class TwitterRoBERTaIronyDetector(TransformerDetector):
    """
    Twitter-RoBERTa model trained for irony detection.

    Reference: Barbieri et al. (2020). TweetEval: Unified Benchmark and
    Comparative Evaluation for Tweet Classification.
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-irony"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
    ):
        model_name = model_name or self.DEFAULT_MODEL
        super().__init__(model_name, device, batch_size, max_length)
        self.label_map = {0: "not_irony", 1: "irony"}

    def load_model(self) -> None:
        """Load Twitter-RoBERTa-Irony model."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info(f"Loading Twitter-RoBERTa-Irony model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # Update label map from config if available
        if hasattr(self.model.config, "id2label"):
            self.label_map = self.model.config.id2label

        logger.info(f"Twitter-RoBERTa-Irony model loaded")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for Twitter-trained models."""
        # Handle user mentions
        words = text.split()
        new_words = []
        for word in words:
            if word.startswith("@"):
                new_words.append("@user")
            elif word.startswith("http"):
                new_words.append("http")
            else:
                new_words.append(word)
        return " ".join(new_words)

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make sarcasm/irony predictions."""
        if self.model is None:
            self.load_model()

        # Preprocess texts
        processed_texts = [self.preprocess_text(t) for t in texts]

        results = []

        with torch.no_grad():
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            for i, text in enumerate(texts):
                pred_label = predictions[i].item()
                prob_dict = {j: probs[i, j].item() for j in range(2)}

                result = PredictionResult(
                    sample_id=f"twitter_irony_{i}",
                    text=text,  # Store original text
                    predicted_label=pred_label,
                    confidence=prob_dict[pred_label],
                    probabilities=prob_dict,
                )
                results.append(result)

        return results


class SarcasmEnsembleDetector:
    """
    Ensemble detector combining multiple sarcasm detection approaches.
    """

    def __init__(
        self,
        models: Optional[List[TransformerDetector]] = None,
        device: Optional[str] = None,
        voting: str = "soft",  # 'hard' or 'soft'
    ):
        self.device = device
        self.voting = voting

        if models is None:
            # Default to Twitter-RoBERTa-Irony
            self.models = [TwitterRoBERTaIronyDetector(device=device)]
        else:
            self.models = models

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make ensemble predictions."""
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            preds = model.predict(texts)
            all_predictions.append(preds)

        # Combine predictions
        results = []
        for i, text in enumerate(texts):
            if self.voting == "soft":
                # Average probabilities
                avg_probs = {}
                for model_preds in all_predictions:
                    for label, prob in model_preds[i].probabilities.items():
                        avg_probs[label] = avg_probs.get(label, 0) + prob

                for label in avg_probs:
                    avg_probs[label] /= len(self.models)

                pred_label = max(avg_probs, key=avg_probs.get)
                confidence = avg_probs[pred_label]

            else:
                # Hard voting
                votes = [mp[i].predicted_label for mp in all_predictions]
                pred_label = max(set(votes), key=votes.count)
                confidence = votes.count(pred_label) / len(votes)
                avg_probs = {0: 1 - confidence, 1: confidence}

            result = PredictionResult(
                sample_id=f"ensemble_{i}",
                text=text,
                predicted_label=pred_label,
                confidence=confidence,
                probabilities=avg_probs,
            )
            results.append(result)

        return results


def create_sarcasm_detector(
    model_type: str = "twitter_roberta",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> TransformerDetector:
    """
    Factory function to create sarcasm detectors.

    Args:
        model_type: Type of detector to create
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        Initialized detector
    """
    if model_type.lower() in ["twitter_roberta", "twitter_roberta_irony"]:
        return TwitterRoBERTaIronyDetector(device=device, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
