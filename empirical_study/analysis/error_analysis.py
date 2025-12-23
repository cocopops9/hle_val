"""Error analysis functions for empirical study."""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..data.dataset_loader import Sample
from ..models.base import PredictionResult
from ..translation.base import TranslationResult


def analyze_implicit_hate_errors(
    samples: List[Sample],
    predictions: List[PredictionResult],
    n_examples: int = 20,
) -> Dict:
    """
    Analyze errors on implicit hate speech samples.

    Identifies patterns in false negatives where implicit hate
    was not detected.

    Args:
        samples: List of Sample objects (implicit hate)
        predictions: Corresponding predictions
        n_examples: Number of example errors to return

    Returns:
        Dictionary with error analysis results
    """
    logger.info("Analyzing implicit hate speech errors...")

    errors = {
        "false_negatives": [],
        "false_positives": [],
        "error_categories": Counter(),
    }

    # Match samples with predictions
    pred_map = {p.sample_id: p for p in predictions}

    for sample in samples:
        if sample.sample_id not in pred_map:
            continue

        pred = pred_map[sample.sample_id]
        true_label = sample.label
        pred_label = pred.predicted_label

        if true_label == 1 and pred_label == 0:
            # False negative - missed implicit hate
            error_info = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": pred.confidence,
                "category": _categorize_implicit_error(sample.text),
            }
            errors["false_negatives"].append(error_info)
            errors["error_categories"][error_info["category"]] += 1

        elif true_label == 0 and pred_label == 1:
            # False positive
            error_info = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": pred.confidence,
            }
            errors["false_positives"].append(error_info)

    # Summary statistics
    errors["summary"] = {
        "total_fn": len(errors["false_negatives"]),
        "total_fp": len(errors["false_positives"]),
        "top_error_categories": errors["error_categories"].most_common(5),
    }

    # Limit examples
    errors["false_negatives"] = errors["false_negatives"][:n_examples]
    errors["false_positives"] = errors["false_positives"][:n_examples]

    logger.info(
        f"Found {errors['summary']['total_fn']} false negatives, "
        f"{errors['summary']['total_fp']} false positives"
    )

    return errors


def _categorize_implicit_error(text: str) -> str:
    """
    Categorize type of implicit hate that was missed.

    Categories based on common implicit hate patterns.
    """
    text_lower = text.lower()

    # Stereotyping patterns
    stereotype_markers = [
        "always", "never", "all of them", "typical", "you know how",
        "what do you expect", "no surprise",
    ]
    if any(m in text_lower for m in stereotype_markers):
        return "stereotype_invocation"

    # Historical/coded references
    coded_markers = [
        "those people", "them", "their kind", "certain groups",
        "you know who", "the usual suspects",
    ]
    if any(m in text_lower for m in coded_markers):
        return "coded_language"

    # Dehumanizing metaphors
    dehumanizing = [
        "animals", "vermin", "plague", "disease", "infestation",
        "swarm", "horde",
    ]
    if any(m in text_lower for m in dehumanizing):
        return "dehumanizing_metaphor"

    # Dog whistles / in-group signals
    dogwhistle = [
        "based", "redpilled", "clown world", "honk", "jogger",
    ]
    if any(m in text_lower for m in dogwhistle):
        return "dog_whistle"

    # Sarcastic/ironic hate
    if any(m in text_lower for m in ["obviously", "of course", "naturally", "surprise"]):
        return "ironic_hate"

    return "other"


def analyze_sarcasm_errors(
    samples: List[Sample],
    predictions: List[PredictionResult],
    n_examples: int = 20,
) -> Dict:
    """
    Analyze errors in sarcasm detection.

    Focuses on false negatives where sarcasm was missed.

    Args:
        samples: List of Sample objects
        predictions: Corresponding predictions
        n_examples: Number of example errors to return

    Returns:
        Dictionary with error analysis results
    """
    logger.info("Analyzing sarcasm detection errors...")

    errors = {
        "false_negatives": [],
        "false_positives": [],
        "error_patterns": Counter(),
    }

    pred_map = {p.sample_id: p for p in predictions}

    for sample in samples:
        if sample.sample_id not in pred_map:
            continue

        pred = pred_map[sample.sample_id]
        true_label = sample.label
        pred_label = pred.predicted_label

        if true_label == 1 and pred_label == 0:
            # False negative - missed sarcasm
            error_info = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": pred.confidence,
                "pattern": _categorize_sarcasm_error(sample.text),
            }
            errors["false_negatives"].append(error_info)
            errors["error_patterns"][error_info["pattern"]] += 1

        elif true_label == 0 and pred_label == 1:
            # False positive
            error_info = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": pred.confidence,
            }
            errors["false_positives"].append(error_info)

    errors["summary"] = {
        "total_fn": len(errors["false_negatives"]),
        "total_fp": len(errors["false_positives"]),
        "top_error_patterns": errors["error_patterns"].most_common(5),
    }

    errors["false_negatives"] = errors["false_negatives"][:n_examples]
    errors["false_positives"] = errors["false_positives"][:n_examples]

    return errors


def _categorize_sarcasm_error(text: str) -> str:
    """Categorize type of sarcasm that was missed."""
    text_lower = text.lower()

    # Polarity contrast markers
    positive_words = ["great", "wonderful", "amazing", "perfect", "love", "best"]
    negative_words = ["worst", "terrible", "awful", "hate", "bad", "fail"]

    has_positive = any(w in text_lower for w in positive_words)
    has_negative = any(w in text_lower for w in negative_words)

    if has_positive and has_negative:
        return "mixed_polarity"
    elif has_positive:
        return "positive_only_context_dependent"

    # Rhetorical questions
    if text.strip().endswith("?") and any(w in text_lower for w in ["really", "seriously", "right"]):
        return "rhetorical_question"

    # Hyperbole
    hyperbole = ["so", "such", "absolutely", "completely", "totally", "literally"]
    if any(w in text_lower for w in hyperbole):
        return "hyperbole"

    # Understatement
    understatement = ["just", "only", "merely", "slightly", "a bit"]
    if any(w in text_lower for w in understatement):
        return "understatement"

    return "context_dependent"


def analyze_mt_induced_errors(
    original_texts: List[str],
    translations: List[TranslationResult],
    original_predictions: List[PredictionResult],
    bt_predictions: List[PredictionResult],
    true_labels: List[int],
    n_examples: int = 20,
) -> Dict:
    """
    Analyze errors induced by machine translation.

    Compares predictions on original vs back-translated text
    to identify translation-induced failures.

    Args:
        original_texts: Original text samples
        translations: Translation results with back-translated text
        original_predictions: Predictions on original text
        bt_predictions: Predictions on back-translated text
        true_labels: Ground truth labels
        n_examples: Number of example errors to return

    Returns:
        Dictionary with error analysis results
    """
    logger.info("Analyzing MT-induced errors...")

    errors = {
        "prediction_changes": [],
        "normalization_examples": [],
        "loss_categories": Counter(),
    }

    for i, (orig, trans, orig_pred, bt_pred, label) in enumerate(
        zip(original_texts, translations, original_predictions, bt_predictions, true_labels)
    ):
        # Check if prediction changed
        if orig_pred.predicted_label != bt_pred.predicted_label:
            change_info = {
                "original_text": orig,
                "back_translated": trans.back_translated_text,
                "original_pred": orig_pred.predicted_label,
                "bt_pred": bt_pred.predicted_label,
                "true_label": label,
                "loss_type": _categorize_mt_loss(orig, trans.back_translated_text),
            }
            errors["prediction_changes"].append(change_info)
            errors["loss_categories"][change_info["loss_type"]] += 1

        # Check for normalization
        normalization = _detect_normalization(orig, trans.back_translated_text)
        if normalization:
            errors["normalization_examples"].append({
                "original": orig,
                "back_translated": trans.back_translated_text,
                "normalization_type": normalization,
            })

    errors["summary"] = {
        "total_changes": len(errors["prediction_changes"]),
        "preservation_rate": 1 - len(errors["prediction_changes"]) / max(len(original_texts), 1),
        "top_loss_categories": errors["loss_categories"].most_common(5),
    }

    errors["prediction_changes"] = errors["prediction_changes"][:n_examples]
    errors["normalization_examples"] = errors["normalization_examples"][:n_examples]

    return errors


def _categorize_mt_loss(original: str, back_translated: str) -> str:
    """Categorize type of pragmatic loss from translation."""
    orig_lower = original.lower()
    bt_lower = back_translated.lower()

    # Emphasis loss
    emphasis_markers = ["so", "very", "really", "absolutely", "completely"]
    orig_emphasis = sum(1 for m in emphasis_markers if m in orig_lower)
    bt_emphasis = sum(1 for m in emphasis_markers if m in bt_lower)
    if orig_emphasis > bt_emphasis:
        return "emphasis_loss"

    # Punctuation loss
    if original.count("!") > back_translated.count("!"):
        return "exclamation_loss"
    if original.count("?") > back_translated.count("?"):
        return "question_loss"

    # Capitalization loss (for emphasis)
    orig_caps = sum(1 for c in original if c.isupper())
    bt_caps = sum(1 for c in back_translated if c.isupper())
    if orig_caps > bt_caps + 5:  # Threshold for significant loss
        return "capitalization_loss"

    # Cultural reference loss
    # (Would need more sophisticated detection)

    # Length change (possible elaboration or truncation)
    length_ratio = len(back_translated) / max(len(original), 1)
    if length_ratio < 0.8:
        return "truncation"
    if length_ratio > 1.2:
        return "elaboration"

    return "semantic_shift"


def _detect_normalization(original: str, back_translated: str) -> Optional[str]:
    """Detect if translation normalized marked language."""
    orig_lower = original.lower()
    bt_lower = back_translated.lower()

    # Emphatic to neutral
    emphatic_pairs = [
        ("so incredibly", "very"),
        ("absolutely amazing", "great"),
        ("totally", "completely"),
    ]
    for emphatic, neutral in emphatic_pairs:
        if emphatic in orig_lower and neutral in bt_lower and emphatic not in bt_lower:
            return "emphatic_to_neutral"

    # Colloquial to formal
    colloquial = ["gonna", "wanna", "gotta", "kinda", "sorta"]
    for word in colloquial:
        if word in orig_lower and word not in bt_lower:
            return "colloquial_to_formal"

    # Slang normalization
    if any(s in orig_lower for s in ["lol", "lmao", "omg", "wtf"]) and \
       not any(s in bt_lower for s in ["lol", "lmao", "omg", "wtf"]):
        return "slang_removal"

    return None


def generate_error_report(
    implicit_hate_errors: Dict,
    sarcasm_errors: Dict,
    mt_errors: Dict,
    output_path: str = "./error_report.md",
) -> str:
    """
    Generate a comprehensive error analysis report.

    Args:
        implicit_hate_errors: Results from analyze_implicit_hate_errors
        sarcasm_errors: Results from analyze_sarcasm_errors
        mt_errors: Results from analyze_mt_induced_errors
        output_path: Path to save report

    Returns:
        Report as markdown string
    """
    report = []
    report.append("# Error Analysis Report\n")

    # Implicit Hate Section
    report.append("## 1. Implicit Hate Speech Errors\n")
    report.append(f"Total false negatives: {implicit_hate_errors['summary']['total_fn']}\n")
    report.append(f"Total false positives: {implicit_hate_errors['summary']['total_fp']}\n")
    report.append("\n### Error Categories\n")
    for cat, count in implicit_hate_errors['summary']['top_error_categories']:
        report.append(f"- {cat}: {count}\n")

    report.append("\n### Example False Negatives\n")
    for err in implicit_hate_errors['false_negatives'][:5]:
        report.append(f"- **Text**: {err['text'][:100]}...\n")
        report.append(f"  - Category: {err['category']}\n")
        report.append(f"  - Confidence: {err['confidence']:.2f}\n")

    # Sarcasm Section
    report.append("\n## 2. Sarcasm Detection Errors\n")
    report.append(f"Total false negatives: {sarcasm_errors['summary']['total_fn']}\n")
    report.append(f"Total false positives: {sarcasm_errors['summary']['total_fp']}\n")
    report.append("\n### Error Patterns\n")
    for pat, count in sarcasm_errors['summary']['top_error_patterns']:
        report.append(f"- {pat}: {count}\n")

    # MT Section
    report.append("\n## 3. MT-Induced Errors\n")
    report.append(f"Total prediction changes: {mt_errors['summary']['total_changes']}\n")
    report.append(f"Preservation rate: {mt_errors['summary']['preservation_rate']:.2%}\n")
    report.append("\n### Loss Categories\n")
    for cat, count in mt_errors['summary']['top_loss_categories']:
        report.append(f"- {cat}: {count}\n")

    report.append("\n### Normalization Examples\n")
    for ex in mt_errors['normalization_examples'][:5]:
        report.append(f"- **Original**: {ex['original'][:80]}...\n")
        report.append(f"  **Back-translated**: {ex['back_translated'][:80]}...\n")
        report.append(f"  Type: {ex['normalization_type']}\n")

    report_text = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info(f"Error report saved to {output_path}")

    return report_text
