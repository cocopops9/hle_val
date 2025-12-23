"""Hate speech detection models: HateBERT and RoBERTa-HatEval."""

from typing import Dict, List, Optional

import torch
from loguru import logger

from .base import PredictionResult, TransformerDetector


class HateBERTDetector(TransformerDetector):
    """
    HateBERT detector for hate speech detection.

    HateBERT is a BERT model pre-trained on Reddit hate communities.
    Reference: Caselli et al. (2021). HateBERT: Retraining BERT for
    Abusive Language Detection in English.
    """

    DEFAULT_MODEL = "GroNLP/hateBERT"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
    ):
        model_name = model_name or self.DEFAULT_MODEL
        super().__init__(model_name, device, batch_size, max_length)
        self.label_map = {0: "not_hate", 1: "hate"}

    def load_model(self) -> None:
        """Load HateBERT model."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BertForSequenceClassification,
        )

        logger.info(f"Loading HateBERT model: {self.model_name}")

        try:
            # Try loading fine-tuned version for hate speech
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
            logger.info("Loading base HateBERT and setting up for classification")

            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info("HateBERT model loaded successfully")

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make hate speech predictions."""
        if self.model is None:
            self.load_model()

        results = []

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
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
                    sample_id=f"hatebert_{i}",
                    text=text,
                    predicted_label=pred_label,
                    confidence=prob_dict[pred_label],
                    probabilities=prob_dict,
                )
                results.append(result)

        return results


class RoBERTaHatEvalDetector(TransformerDetector):
    """
    RoBERTa fine-tuned on HatEval training data.

    This model uses RoBERTa-base fine-tuned on the HatEval dataset
    for hate speech detection.
    """

    DEFAULT_MODEL = "facebook/roberta-base"
    HATEVAL_FINETUNED = "cardiffnlp/twitter-roberta-base-offensive"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
        use_hateval_finetuned: bool = True,
    ):
        # Use pre-fine-tuned offensive detection model as proxy
        if use_hateval_finetuned:
            model_name = self.HATEVAL_FINETUNED
        else:
            model_name = model_name or self.DEFAULT_MODEL

        super().__init__(model_name, device, batch_size, max_length)
        self.use_hateval_finetuned = use_hateval_finetuned

    def load_model(self) -> None:
        """Load RoBERTa model."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info(f"Loading RoBERTa model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # Get label mapping if available
        if hasattr(self.model.config, "id2label"):
            self.label_map = self.model.config.id2label
        else:
            self.label_map = {0: "not_offensive", 1: "offensive"}

        logger.info(f"RoBERTa model loaded with labels: {self.label_map}")

    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Make hate speech predictions."""
        if self.model is None:
            self.load_model()

        results = []

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
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

            num_labels = logits.shape[1]

            for i, text in enumerate(texts):
                pred_label = predictions[i].item()
                prob_dict = {j: probs[i, j].item() for j in range(num_labels)}

                # Map to binary hate/not_hate if needed
                # For offensive detection, offensive=1 maps to hate=1
                if num_labels > 2:
                    # Aggregate non-neutral classes
                    hate_prob = sum(
                        prob_dict[j] for j in range(1, num_labels)
                    )
                    prob_dict = {0: prob_dict[0], 1: hate_prob}
                    pred_label = 1 if hate_prob > 0.5 else 0

                result = PredictionResult(
                    sample_id=f"roberta_hateval_{i}",
                    text=text,
                    predicted_label=pred_label,
                    confidence=prob_dict[pred_label],
                    probabilities=prob_dict,
                )
                results.append(result)

        return results


def create_hate_speech_detector(
    model_type: str,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> TransformerDetector:
    """
    Factory function to create hate speech detectors.

    Args:
        model_type: One of 'hatebert', 'roberta_hateval'
        device: Device to use for inference
        batch_size: Batch size for inference

    Returns:
        Initialized detector
    """
    if model_type.lower() == "hatebert":
        return HateBERTDetector(device=device, batch_size=batch_size)
    elif model_type.lower() in ["roberta_hateval", "roberta"]:
        return RoBERTaHatEvalDetector(device=device, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
