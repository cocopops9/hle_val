"""
Functional Validation of Data Limitations Study

This study validates whether data limitations (class imbalance, annotation subjectivity,
domain specificity) cause specific model failures, and tests whether targeted
interventions can mitigate these weaknesses.

Phase 1: Diagnostic Evaluation
- Train ELECTRA/RoBERTa on HateXplain (explicit) and SBIC (implicit)
- Evaluate on HateCheck functional tests targeting specific data limitations

Phase 2: Targeted Interventions
- Focal loss for class rebalancing
- Multi-dataset training for domain coverage
- Targeted augmentation for specific gaps

Reference: Röttger et al. (2021). HateCheck: Functional Tests for Hate Speech Detection Models.
"""

# Disable TensorFlow to avoid Keras 3 conflicts
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data import (
    HateXplainLoader,
    SBICLoader,
    HateCheckLoader,
    FUNCTIONAL_TESTS,
    DATA_LIMITATION_TESTS,
)
from .data.dataset_loader import Sample

# Delayed imports
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not installed")


@dataclass
class FunctionalTestResult:
    """Result for a single functional test."""
    test_id: str
    test_name: str
    n_samples: int
    accuracy: float
    hateful_expected: bool
    predictions_hateful: int
    predictions_non_hateful: int

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "hateful_expected": self.hateful_expected,
            "predictions_hateful": self.predictions_hateful,
            "predictions_non_hateful": self.predictions_non_hateful,
        }


@dataclass
class DiagnosticResult:
    """Results from diagnostic evaluation."""
    model_name: str
    train_dataset: str
    test_results: List[FunctionalTestResult] = field(default_factory=list)
    macro_accuracy_hateful: float = 0.0
    macro_accuracy_nonhateful: float = 0.0
    overall_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "train_dataset": self.train_dataset,
            "test_results": [r.to_dict() for r in self.test_results],
            "macro_accuracy_hateful": self.macro_accuracy_hateful,
            "macro_accuracy_nonhateful": self.macro_accuracy_nonhateful,
            "overall_accuracy": self.overall_accuracy,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Reference: Lin et al. (2017). Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


def create_dataset_class():
    """Create HateSpeechDataset class with torch imports."""
    from torch.utils.data import Dataset

    class HateSpeechDataset(Dataset):
        def __init__(self, samples: List[Sample], tokenizer, max_length: int = 128):
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


class FocalLossTrainer:
    """Custom trainer using focal loss."""

    def __init__(self, model, tokenizer, device, gamma: float = 2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma

    def train(
        self,
        train_samples: List[Sample],
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
    ):
        from torch.utils.data import DataLoader
        from torch.optim import AdamW

        HateSpeechDataset = create_dataset_class()
        train_dataset = HateSpeechDataset(train_samples, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Compute class weights
        labels = [s.label for s in train_samples]
        class_counts = [labels.count(0), labels.count(1)]
        total = sum(class_counts)
        alpha = torch.tensor([total / (2 * c) for c in class_counts]).to(self.device)

        focal_loss = FocalLoss(alpha=alpha, gamma=self.gamma)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = focal_loss(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, samples: List[Sample], batch_size: int = 16) -> np.ndarray:
        from torch.utils.data import DataLoader

        HateSpeechDataset = create_dataset_class()
        dataset = HateSpeechDataset(samples, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)


class FunctionalValidationStudy:
    """
    Functional validation study using HateCheck as diagnostic instrument.

    Phase 1: Diagnostic evaluation to validate data limitation effects
    Phase 2: Test targeted interventions
    """

    MODELS = {
        "electra": "google/electra-base-discriminator",
        "roberta": "roberta-base",
        "bert": "bert-base-cased",
    }

    # Target functional tests by data limitation
    TARGET_TESTS = {
        "class_imbalance": ["F9", "F15"],
        "identity_bias": ["F18", "F19"],
        "domain_specificity": ["F20", "F21"],
        "lexical_coverage": ["F25", "F26", "F27", "F28", "F29"],
    }

    def __init__(
        self,
        output_dir: str = "./results/functional_validation",
        cache_dir: str = "./data/cache",
        device: Optional[str] = None,
        n_train_samples: int = 1000,
        random_seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.n_train_samples = n_train_samples
        self.random_seed = random_seed

        if device:
            self.device = device
        elif HAS_TORCH and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        self.results: List[DiagnosticResult] = []

    def _balanced_sample(self, samples: List[Sample], n_total: int) -> List[Sample]:
        """Create balanced sample with equal positive/negative."""
        np.random.seed(self.random_seed)

        positive = [s for s in samples if s.label == 1]
        negative = [s for s in samples if s.label == 0]

        n_per_class = n_total // 2

        if len(positive) >= n_per_class:
            idx = np.random.choice(len(positive), n_per_class, replace=False)
            positive = [positive[i] for i in idx]

        if len(negative) >= n_per_class:
            idx = np.random.choice(len(negative), n_per_class, replace=False)
            negative = [negative[i] for i in idx]

        combined = positive + negative
        np.random.shuffle(combined)
        return combined

    def load_training_data(self) -> Dict[str, List[Sample]]:
        """Load HateXplain and SBIC training data."""
        logger.info("Loading training datasets...")

        # HateXplain (explicit hate)
        hx_loader = HateXplainLoader(cache_dir=str(self.cache_dir))
        hx_samples = hx_loader.load()
        hx_train = self._balanced_sample(hx_samples, self.n_train_samples)

        # SBIC (implicit hate)
        sbic_loader = SBICLoader(cache_dir=str(self.cache_dir), implicit_only=True)
        sbic_samples = sbic_loader.load()
        sbic_train = self._balanced_sample(sbic_samples, self.n_train_samples)

        logger.info(f"HateXplain train: {len(hx_train)} samples")
        logger.info(f"SBIC train: {len(sbic_train)} samples")

        return {
            "hatexplain": hx_train,
            "sbic": sbic_train,
        }

    def load_hatecheck(self) -> Dict[str, List[Sample]]:
        """Load HateCheck test samples grouped by functional test."""
        logger.info("Loading HateCheck diagnostic data...")

        loader = HateCheckLoader(cache_dir=str(self.cache_dir))
        loader.load()

        return loader.get_samples_by_category()

    def train_model(
        self,
        model_type: str,
        train_samples: List[Sample],
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        use_focal_loss: bool = False,
    ):
        """Train a model on the given samples."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = self.MODELS.get(model_type, model_type)
        logger.info(f"Training {model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

        if use_focal_loss:
            trainer = FocalLossTrainer(model, tokenizer, self.device)
            trainer.train(train_samples, num_epochs, batch_size, learning_rate)
            return model, tokenizer, trainer
        else:
            from transformers import TrainingArguments, Trainer

            HateSpeechDataset = create_dataset_class()
            train_dataset = HateSpeechDataset(train_samples, tokenizer)

            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_steps=100,
                save_strategy="no",
                report_to="none",
                seed=self.random_seed,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            return model, tokenizer, trainer

    def evaluate_on_hatecheck(
        self,
        model,
        tokenizer,
        trainer,
        hatecheck_samples: Dict[str, List[Sample]],
        use_focal_loss: bool = False,
    ) -> List[FunctionalTestResult]:
        """Evaluate model on HateCheck functional tests."""
        results = []

        for test_id, samples in hatecheck_samples.items():
            if not samples:
                continue

            # Get predictions
            if use_focal_loss:
                preds = trainer.predict(samples)
            else:
                HateSpeechDataset = create_dataset_class()
                test_dataset = HateSpeechDataset(samples, tokenizer)
                predictions = trainer.predict(test_dataset)
                preds = np.argmax(predictions.predictions, axis=1)

            labels = np.array([s.label for s in samples])
            accuracy = accuracy_score(labels, preds)

            # Get expected label for this test
            hateful_expected = FUNCTIONAL_TESTS.get(test_id, {}).get("hateful", None)

            result = FunctionalTestResult(
                test_id=test_id,
                test_name=FUNCTIONAL_TESTS.get(test_id, {}).get("name", ""),
                n_samples=len(samples),
                accuracy=accuracy,
                hateful_expected=hateful_expected,
                predictions_hateful=int(sum(preds == 1)),
                predictions_non_hateful=int(sum(preds == 0)),
            )
            results.append(result)

        return results

    def run_phase1_diagnostic(
        self,
        model_types: List[str] = ["electra", "roberta"],
        num_epochs: int = 3,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Phase 1: Diagnostic evaluation.

        Train models on HateXplain and SBIC, evaluate on HateCheck.
        """
        logger.info("=" * 60)
        logger.info("Phase 1: Diagnostic Evaluation")
        logger.info("=" * 60)

        # Load data
        train_data = self.load_training_data()
        hatecheck_samples = self.load_hatecheck()

        all_results = []

        for model_type in model_types:
            for train_name, train_samples in train_data.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Model: {model_type.upper()}, Train: {train_name}")
                logger.info("=" * 60)

                # Train model
                model, tokenizer, trainer = self.train_model(
                    model_type=model_type,
                    train_samples=train_samples,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                )

                # Evaluate on HateCheck
                test_results = self.evaluate_on_hatecheck(
                    model, tokenizer, trainer, hatecheck_samples
                )

                # Compute macro accuracies
                hateful_tests = [r for r in test_results if r.hateful_expected]
                nonhateful_tests = [r for r in test_results if not r.hateful_expected]

                macro_hateful = np.mean([r.accuracy for r in hateful_tests]) if hateful_tests else 0
                macro_nonhateful = np.mean([r.accuracy for r in nonhateful_tests]) if nonhateful_tests else 0
                overall = np.mean([r.accuracy for r in test_results]) if test_results else 0

                result = DiagnosticResult(
                    model_name=model_type,
                    train_dataset=train_name,
                    test_results=test_results,
                    macro_accuracy_hateful=macro_hateful,
                    macro_accuracy_nonhateful=macro_nonhateful,
                    overall_accuracy=overall,
                )

                all_results.append(result)
                self.results.append(result)

                # Log key results
                logger.info(f"\nResults for {model_type}/{train_name}:")
                logger.info(f"  Macro accuracy (hateful tests): {macro_hateful:.3f}")
                logger.info(f"  Macro accuracy (non-hateful tests): {macro_nonhateful:.3f}")
                logger.info(f"  Overall accuracy: {overall:.3f}")

                # Log data limitation tests
                for limitation, test_ids in self.TARGET_TESTS.items():
                    limitation_results = [r for r in test_results if r.test_id in test_ids]
                    if limitation_results:
                        avg_acc = np.mean([r.accuracy for r in limitation_results])
                        logger.info(f"  {limitation}: {avg_acc:.3f}")

                # Cleanup
                del model, trainer
                if HAS_TORCH:
                    torch.cuda.empty_cache()

        # Create summary DataFrame
        df = self._create_results_dataframe(all_results)
        self._save_results(df, "phase1_diagnostic")

        return df

    def run_phase2_interventions(
        self,
        model_type: str = "electra",
        num_epochs: int = 3,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Phase 2: Test targeted interventions.

        1. Focal loss for class rebalancing
        2. Multi-dataset training
        3. Targeted augmentation
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Targeted Interventions")
        logger.info("=" * 60)

        train_data = self.load_training_data()
        hatecheck_samples = self.load_hatecheck()

        intervention_results = []

        # Baseline (standard training on HateXplain)
        logger.info("\n--- Baseline (HateXplain, standard loss) ---")
        model, tokenizer, trainer = self.train_model(
            model_type=model_type,
            train_samples=train_data["hatexplain"],
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_focal_loss=False,
        )
        baseline_results = self.evaluate_on_hatecheck(
            model, tokenizer, trainer, hatecheck_samples
        )
        intervention_results.append(("baseline", baseline_results))
        del model, trainer

        # Intervention 1: Focal Loss
        logger.info("\n--- Intervention 1: Focal Loss ---")
        model, tokenizer, trainer = self.train_model(
            model_type=model_type,
            train_samples=train_data["hatexplain"],
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_focal_loss=True,
        )
        focal_results = self.evaluate_on_hatecheck(
            model, tokenizer, trainer, hatecheck_samples, use_focal_loss=True
        )
        intervention_results.append(("focal_loss", focal_results))
        del model, trainer

        # Intervention 2: Multi-Dataset Training
        logger.info("\n--- Intervention 2: Multi-Dataset Training ---")
        combined_train = train_data["hatexplain"] + train_data["sbic"]
        np.random.seed(self.random_seed)
        np.random.shuffle(combined_train)

        model, tokenizer, trainer = self.train_model(
            model_type=model_type,
            train_samples=combined_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_focal_loss=False,
        )
        multi_results = self.evaluate_on_hatecheck(
            model, tokenizer, trainer, hatecheck_samples
        )
        intervention_results.append(("multi_dataset", multi_results))
        del model, trainer

        # Create comparison DataFrame
        df = self._create_intervention_comparison(intervention_results)
        self._save_results(df, "phase2_interventions")

        return df

    def _create_results_dataframe(self, results: List[DiagnosticResult]) -> pd.DataFrame:
        """Create DataFrame from diagnostic results."""
        rows = []
        for result in results:
            for test_result in result.test_results:
                rows.append({
                    "model": result.model_name,
                    "train_data": result.train_dataset,
                    "test_id": test_result.test_id,
                    "test_name": test_result.test_name,
                    "n_samples": test_result.n_samples,
                    "accuracy": test_result.accuracy,
                    "hateful_expected": test_result.hateful_expected,
                })

        return pd.DataFrame(rows)

    def _create_intervention_comparison(
        self,
        results: List[Tuple[str, List[FunctionalTestResult]]],
    ) -> pd.DataFrame:
        """Create comparison DataFrame for interventions."""
        rows = []
        baseline_acc = {}

        # Get baseline accuracies
        for intervention, test_results in results:
            if intervention == "baseline":
                for r in test_results:
                    baseline_acc[r.test_id] = r.accuracy

        # Compute deltas
        for intervention, test_results in results:
            for r in test_results:
                delta = r.accuracy - baseline_acc.get(r.test_id, 0)
                rows.append({
                    "intervention": intervention,
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "accuracy": r.accuracy,
                    "delta": delta if intervention != "baseline" else 0,
                    "hateful_expected": r.hateful_expected,
                })

        return pd.DataFrame(rows)

    def _save_results(self, df: pd.DataFrame, prefix: str) -> None:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = self.output_dir / f"{prefix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")

        json_path = self.output_dir / f"{prefix}_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)

    def run_full_study(
        self,
        model_types: List[str] = ["electra", "roberta"],
        num_epochs: int = 3,
        batch_size: int = 16,
    ) -> Dict[str, pd.DataFrame]:
        """Run complete study (Phase 1 + Phase 2)."""
        results = {}

        # Phase 1
        results["phase1"] = self.run_phase1_diagnostic(
            model_types=model_types,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Phase 2
        results["phase2"] = self.run_phase2_interventions(
            model_type=model_types[0],  # Use first model for interventions
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        return results


def main():
    """Run functional validation study from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Functional Validation of Data Limitations Study"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/functional_validation",
        help="Output directory"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./data/cache",
        help="Cache directory"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["electra", "roberta"],
        choices=["electra", "roberta", "bert"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Training samples per class"
    )
    parser.add_argument(
        "--phase", type=str, choices=["1", "2", "all"], default="all",
        help="Which phase to run"
    )

    args = parser.parse_args()

    study = FunctionalValidationStudy(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        n_train_samples=args.n_samples,
    )

    if args.phase == "1":
        results = study.run_phase1_diagnostic(
            model_types=args.models,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.phase == "2":
        results = study.run_phase2_interventions(
            model_type=args.models[0],
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        results = study.run_full_study(
            model_types=args.models,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
