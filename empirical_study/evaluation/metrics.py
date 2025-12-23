"""Evaluation metrics for detection performance."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class ConfidenceInterval:
    """Stores confidence interval information."""

    lower: float
    upper: float
    point_estimate: float
    confidence_level: float = 0.95

    def __repr__(self) -> str:
        return f"{self.point_estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""

    model_name: str
    dataset_name: str
    content_type: str

    # Core metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float

    # Confidence intervals
    f1_ci: Optional[ConfidenceInterval] = None
    macro_f1_ci: Optional[ConfidenceInterval] = None

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Additional metrics
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    preservation_rate: Optional[float] = None

    # Statistical tests
    p_value: Optional[float] = None

    # Sample counts
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "content_type": self.content_type,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "f1_ci_lower": self.f1_ci.lower if self.f1_ci else None,
            "f1_ci_upper": self.f1_ci.upper if self.f1_ci else None,
            "fpr": self.false_positive_rate,
            "fnr": self.false_negative_rate,
            "preservation_rate": self.preservation_rate,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
        }


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    """
    Compute F1 score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'macro', 'micro', 'weighted')
        pos_label: Positive class label for binary classification

    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, pos_label=pos_label)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro F1 score (unweighted mean across classes).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average="macro")


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable = compute_macro_f1,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_fn: Function to compute metric (takes y_true, y_pred)
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_seed: Random seed for reproducibility

    Returns:
        ConfidenceInterval object
    """
    np.random.seed(random_seed)

    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_resamples):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        # Compute metric on bootstrap sample
        try:
            score = metric_fn(y_true_sample, y_pred_sample)
            bootstrap_scores.append(score)
        except Exception:
            continue  # Skip failed samples

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(y_true, y_pred)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=point_estimate,
        confidence_level=confidence_level,
    )


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    correction: bool = True,
) -> Tuple[float, float]:
    """
    Perform McNemar's test for paired comparison of classifiers.

    Tests whether two classifiers have significantly different error rates.

    Args:
        y_true: Ground truth labels
        y_pred_1: Predictions from classifier 1
        y_pred_2: Predictions from classifier 2
        correction: Apply continuity correction

    Returns:
        Tuple of (test statistic, p-value)
    """
    from scipy import stats

    # Build contingency table
    # b = model 1 correct, model 2 wrong
    # c = model 1 wrong, model 2 correct
    correct_1 = y_pred_1 == y_true
    correct_2 = y_pred_2 == y_true

    b = np.sum(correct_1 & ~correct_2)  # 1 right, 2 wrong
    c = np.sum(~correct_1 & correct_2)  # 1 wrong, 2 right

    # McNemar's test
    if b + c == 0:
        return 0.0, 1.0  # No disagreement

    if correction:
        # With continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)

    # Chi-square with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


def compute_preservation_rate(
    original_predictions: np.ndarray,
    backtranslated_predictions: np.ndarray,
) -> float:
    """
    Compute pragmatic preservation rate.

    The proportion of examples where model predictions match between
    original and back-translated text, independent of correctness.

    Args:
        original_predictions: Predictions on original text
        backtranslated_predictions: Predictions on back-translated text

    Returns:
        Preservation rate (0 to 1)
    """
    if len(original_predictions) == 0:
        return 0.0

    matches = original_predictions == backtranslated_predictions
    return np.mean(matches)


def compute_false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Compute false positive rate (FPR).

    FPR = FP / (FP + TN) = proportion of negatives incorrectly classified

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        False positive rate
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape[0] < 2:
        return 0.0

    # For binary: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    tn = cm[0, 0]
    fp = cm[0, 1]

    if (fp + tn) == 0:
        return 0.0

    return fp / (fp + tn)


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Compute false negative rate (FNR).

    FNR = FN / (FN + TP) = proportion of positives incorrectly classified

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        False negative rate
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape[0] < 2:
        return 0.0

    fn = cm[1, 0]
    tp = cm[1, 1]

    if (fn + tp) == 0:
        return 0.0

    return fn / (fn + tp)


def evaluate_model(
    model_name: str,
    dataset_name: str,
    content_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> EvaluationResults:
    """
    Comprehensive model evaluation.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        content_type: Type of content (explicit, implicit, etc.)
        y_true: Ground truth labels
        y_pred: Predicted labels
        compute_ci: Whether to compute confidence intervals
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        EvaluationResults object
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = compute_macro_f1(y_true, y_pred)

    # Compute confidence intervals
    f1_ci = None
    macro_f1_ci = None

    if compute_ci and len(y_true) > 10:
        f1_ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            metric_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
        )
        macro_f1_ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            metric_fn=compute_macro_f1,
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
        )

    # Compute confusion matrix and derived metrics
    cm = confusion_matrix(y_true, y_pred)
    fpr = compute_false_positive_rate(y_true, y_pred)
    fnr = compute_false_negative_rate(y_true, y_pred)

    return EvaluationResults(
        model_name=model_name,
        dataset_name=dataset_name,
        content_type=content_type,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        macro_f1=macro_f1,
        f1_ci=f1_ci,
        macro_f1_ci=macro_f1_ci,
        confusion_matrix=cm,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        n_samples=len(y_true),
        n_positive=int(np.sum(y_true == 1)),
        n_negative=int(np.sum(y_true == 0)),
    )
