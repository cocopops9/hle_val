"""
Cross-Domain Hate Speech Detection Study

This module implements the cross-domain evaluation:
1. Fine-tune BERT on HateXplain (explicit) → Test on HateXplain + SBIC
2. Fine-tune BERT on SBIC (implicit) → Test on HateXplain + SBIC
3. Fine-tune RoBERTa on HateXplain → Test on HateXplain + SBIC
4. Fine-tune RoBERTa on SBIC → Test on HateXplain + SBIC

This tests the hypothesis that models trained on explicit hate
struggle with implicit hate, and vice versa.
"""

# Disable TensorFlow to avoid Keras 3 conflicts with transformers
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Delay transformers import to avoid TensorFlow/Keras conflicts
HAS_TRANSFORMERS = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not installed. Install with: pip install torch")

from .data import HateXplainLoader, SBICLoader
from .data.dataset_loader import Sample


@dataclass
class CrossDomainResult:
    """Results from a cross-domain evaluation."""
    model_name: str
    train_dataset: str
    test_dataset: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    fpr: float  # False positive rate
    fnr: float  # False negative rate
    n_train: int
    n_test: int
    confusion_matrix: List[List[int]]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "confusion_matrix": self.confusion_matrix,
        }


def create_dataset_class():
    """Create HateSpeechDataset class with torch imports."""
    from torch.utils.data import Dataset

    class HateSpeechDataset(Dataset):
        """PyTorch Dataset for hate speech samples."""

        def __init__(
            self,
            samples: List[Sample],
            tokenizer,
            max_length: int = 128,
        ):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            encoding = self.tokenizer(
                sample.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(sample.label, dtype=torch.long),
            }

    return HateSpeechDataset


class CrossDomainStudy:
    """
    Cross-domain hate speech detection study.

    Compares performance when training on explicit (HateXplain) vs
    implicit (SBIC) hate speech datasets.
    """

    # Available models for fine-tuning
    MODELS = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
    }

    def __init__(
        self,
        output_dir: str = "./results/cross_domain",
        cache_dir: str = "./data/cache",
        device: Optional[str] = None,
        n_samples_per_class: int = 1000,  # For balanced training
        test_size: float = 0.2,
        random_seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.n_samples_per_class = n_samples_per_class
        self.test_size = test_size
        self.random_seed = random_seed

        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        # Store results
        self.results: List[CrossDomainResult] = []

    def _balanced_split(
        self,
        samples: List[Sample],
        test_ratio: float = 0.2,
    ) -> Tuple[List[Sample], List[Sample]]:
        """Split samples into train/test with balanced classes."""
        np.random.seed(self.random_seed)

        positive = [s for s in samples if s.label == 1]
        negative = [s for s in samples if s.label == 0]

        # Limit to n_samples_per_class
        if len(positive) > self.n_samples_per_class:
            idx = np.random.choice(len(positive), self.n_samples_per_class, replace=False)
            positive = [positive[i] for i in idx]
        if len(negative) > self.n_samples_per_class:
            idx = np.random.choice(len(negative), self.n_samples_per_class, replace=False)
            negative = [negative[i] for i in idx]

        # Split each class
        n_test_pos = int(len(positive) * test_ratio)
        n_test_neg = int(len(negative) * test_ratio)

        np.random.shuffle(positive)
        np.random.shuffle(negative)

        test_samples = positive[:n_test_pos] + negative[:n_test_neg]
        train_samples = positive[n_test_pos:] + negative[n_test_neg:]

        np.random.shuffle(test_samples)
        np.random.shuffle(train_samples)

        return train_samples, test_samples

    def load_datasets(self) -> Dict[str, Dict[str, List[Sample]]]:
        """Load and prepare HateXplain and SBIC datasets."""
        logger.info("Loading datasets...")

        # Load HateXplain (explicit hate)
        hatexplain_loader = HateXplainLoader(cache_dir=str(self.cache_dir))
        hatexplain_samples = hatexplain_loader.load()
        hatexplain_train, hatexplain_test = self._balanced_split(hatexplain_samples)

        # Load SBIC (implicit hate)
        sbic_loader = SBICLoader(cache_dir=str(self.cache_dir), implicit_only=True)
        sbic_samples = sbic_loader.load()
        sbic_train, sbic_test = self._balanced_split(sbic_samples)

        logger.info(f"HateXplain: {len(hatexplain_train)} train, {len(hatexplain_test)} test")
        logger.info(f"SBIC: {len(sbic_train)} train, {len(sbic_test)} test")

        return {
            "hatexplain": {"train": hatexplain_train, "test": hatexplain_test},
            "sbic": {"train": sbic_train, "test": sbic_test},
        }

    def compute_metrics(self, predictions, labels) -> dict:
        """Compute evaluation metrics."""
        preds = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "fpr": fpr,
            "fnr": fnr,
            "confusion_matrix": cm.tolist(),
        }

    def fine_tune_and_evaluate(
        self,
        model_type: str,
        train_samples: List[Sample],
        test_datasets: Dict[str, List[Sample]],
        train_dataset_name: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> List[CrossDomainResult]:
        """Fine-tune a model and evaluate on multiple test sets."""

        # Import transformers here to avoid TensorFlow/Keras conflicts at module load
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
        except ImportError as e:
            logger.error(f"transformers library not installed or import error: {e}")
            logger.error("Install with: pip install transformers torch")
            return []

        if not HAS_TORCH:
            logger.error("PyTorch not installed. Install with: pip install torch")
            return []

        model_path = self.MODELS.get(model_type, model_type)
        logger.info(f"Fine-tuning {model_path} on {train_dataset_name}...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
        )

        # Create dataset class (delayed import for torch)
        HateSpeechDataset = create_dataset_class()

        # Create training dataset
        train_dataset = HateSpeechDataset(train_samples, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"{model_type}_{train_dataset_name}"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_strategy="no",  # Don't save checkpoints to save space
            report_to="none",  # Disable wandb/tensorboard
            seed=self.random_seed,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Train
        logger.info("Training...")
        trainer.train()

        # Evaluate on all test sets
        results = []
        for test_name, test_samples in test_datasets.items():
            logger.info(f"Evaluating on {test_name}...")

            test_dataset = HateSpeechDataset(test_samples, tokenizer)

            # Get predictions
            predictions = trainer.predict(test_dataset)
            preds = predictions.predictions
            labels = np.array([s.label for s in test_samples])

            # Compute metrics
            metrics = self.compute_metrics(preds, labels)

            result = CrossDomainResult(
                model_name=model_type,
                train_dataset=train_dataset_name,
                test_dataset=test_name,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                macro_f1=metrics["macro_f1"],
                fpr=metrics["fpr"],
                fnr=metrics["fnr"],
                n_train=len(train_samples),
                n_test=len(test_samples),
                confusion_matrix=metrics["confusion_matrix"],
            )
            results.append(result)

            logger.info(f"  {test_name}: F1={metrics['f1']:.3f}, Macro-F1={metrics['macro_f1']:.3f}")

        # Clean up to free memory
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    def run_study(
        self,
        model_types: List[str] = ["bert", "roberta"],
        num_epochs: int = 3,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Run the complete cross-domain study.

        For each model type:
        1. Train on HateXplain → Test on HateXplain + SBIC
        2. Train on SBIC → Test on HateXplain + SBIC
        """
        logger.info("=" * 60)
        logger.info("Cross-Domain Hate Speech Detection Study")
        logger.info("=" * 60)

        # Load datasets
        datasets = self.load_datasets()

        all_results = []

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_type.upper()}")
            logger.info("=" * 60)

            # Test sets for evaluation
            test_datasets = {
                "hatexplain": datasets["hatexplain"]["test"],
                "sbic": datasets["sbic"]["test"],
            }

            # Train on HateXplain (explicit)
            results_explicit = self.fine_tune_and_evaluate(
                model_type=model_type,
                train_samples=datasets["hatexplain"]["train"],
                test_datasets=test_datasets,
                train_dataset_name="hatexplain",
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            all_results.extend(results_explicit)

            # Train on SBIC (implicit)
            results_implicit = self.fine_tune_and_evaluate(
                model_type=model_type,
                train_samples=datasets["sbic"]["train"],
                test_datasets=test_datasets,
                train_dataset_name="sbic",
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            all_results.extend(results_implicit)

        # Store results
        self.results = all_results

        # Create DataFrame
        df = pd.DataFrame([r.to_dict() for r in all_results])

        # Save results
        self._save_results(df)

        return df

    def _save_results(self, df: pd.DataFrame) -> None:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = self.output_dir / f"cross_domain_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")

        # JSON
        json_path = self.output_dir / f"cross_domain_results_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)

        # Summary table
        summary = self._create_summary_table(df)
        summary_path = self.output_dir / f"cross_domain_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info(f"Saved summary to {summary_path}")

    def _create_summary_table(self, df: pd.DataFrame) -> str:
        """Create a summary table of results."""
        lines = []
        lines.append("=" * 80)
        lines.append("CROSS-DOMAIN HATE SPEECH DETECTION STUDY RESULTS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Hypothesis: Models trained on explicit hate struggle with implicit hate,")
        lines.append("           and models trained on implicit hate struggle with explicit hate.")
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{'Model':<12} {'Train':<12} {'Test':<12} {'F1':>8} {'Macro-F1':>10} {'FPR':>8}")
        lines.append("-" * 80)

        for _, row in df.iterrows():
            lines.append(
                f"{row['model_name']:<12} "
                f"{row['train_dataset']:<12} "
                f"{row['test_dataset']:<12} "
                f"{row['f1']:>8.3f} "
                f"{row['macro_f1']:>10.3f} "
                f"{row['fpr']:>8.3f}"
            )

        lines.append("-" * 80)
        lines.append("")

        # Compute cross-domain performance drops
        lines.append("CROSS-DOMAIN PERFORMANCE ANALYSIS:")
        lines.append("")

        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]

            # HateXplain trained
            hx_on_hx = model_df[(model_df['train_dataset'] == 'hatexplain') &
                                (model_df['test_dataset'] == 'hatexplain')]['f1'].values
            hx_on_sbic = model_df[(model_df['train_dataset'] == 'hatexplain') &
                                  (model_df['test_dataset'] == 'sbic')]['f1'].values

            # SBIC trained
            sbic_on_sbic = model_df[(model_df['train_dataset'] == 'sbic') &
                                    (model_df['test_dataset'] == 'sbic')]['f1'].values
            sbic_on_hx = model_df[(model_df['train_dataset'] == 'sbic') &
                                  (model_df['test_dataset'] == 'hatexplain')]['f1'].values

            if len(hx_on_hx) > 0 and len(hx_on_sbic) > 0:
                drop1 = hx_on_hx[0] - hx_on_sbic[0]
                lines.append(f"{model.upper()}:")
                lines.append(f"  Trained on HateXplain (explicit):")
                lines.append(f"    → Test on HateXplain: F1 = {hx_on_hx[0]:.3f}")
                lines.append(f"    → Test on SBIC:       F1 = {hx_on_sbic[0]:.3f} (drop: {drop1:.3f})")

            if len(sbic_on_sbic) > 0 and len(sbic_on_hx) > 0:
                drop2 = sbic_on_sbic[0] - sbic_on_hx[0]
                lines.append(f"  Trained on SBIC (implicit):")
                lines.append(f"    → Test on SBIC:       F1 = {sbic_on_sbic[0]:.3f}")
                lines.append(f"    → Test on HateXplain: F1 = {sbic_on_hx[0]:.3f} (drop: {drop2:.3f})")
                lines.append("")

        return "\n".join(lines)


def main():
    """Run cross-domain study from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-Domain Hate Speech Detection Study"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/cross_domain",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./data/cache",
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["bert", "roberta"],
        choices=["bert", "roberta", "distilbert"],
        help="Models to fine-tune"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of samples per class for training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(
        "cross_domain_study_{time}.log",
        rotation="100 MB",
        level="INFO",
    )

    # Run study
    study = CrossDomainStudy(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        n_samples_per_class=args.n_samples,
    )

    results = study.run_study(
        model_types=args.models,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results.to_string())


if __name__ == "__main__":
    main()
