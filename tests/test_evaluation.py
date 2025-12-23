"""Tests for evaluation metrics."""

import numpy as np
import pytest

from empirical_study.evaluation.metrics import (
    compute_f1_score,
    compute_macro_f1,
    compute_bootstrap_ci,
    mcnemar_test,
    compute_preservation_rate,
    compute_false_positive_rate,
    evaluate_model,
)
from empirical_study.evaluation.statistical import (
    bonferroni_correction,
    paired_comparison,
    compute_effect_size,
)


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_f1_score(self):
        """Test F1 score computation."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1])

        f1 = compute_f1_score(y_true, y_pred)
        assert 0 <= f1 <= 1
        assert f1 == pytest.approx(0.666, rel=0.01)

    def test_compute_macro_f1(self):
        """Test macro F1 score computation."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        macro_f1 = compute_macro_f1(y_true, y_pred)
        assert 0 <= macro_f1 <= 1

    def test_compute_bootstrap_ci(self):
        """Test bootstrap confidence interval computation."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

        ci = compute_bootstrap_ci(
            y_true, y_pred,
            n_resamples=100,
            confidence_level=0.95,
        )

        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.confidence_level == 0.95

    def test_mcnemar_test(self):
        """Test McNemar's test."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_pred_1 = np.array([1, 1, 0, 0, 0, 1, 1, 0])
        y_pred_2 = np.array([1, 0, 0, 0, 1, 0, 1, 0])

        stat, p_value = mcnemar_test(y_true, y_pred_1, y_pred_2)

        assert stat >= 0
        assert 0 <= p_value <= 1

    def test_compute_preservation_rate(self):
        """Test pragmatic preservation rate."""
        original = np.array([1, 1, 0, 0, 1])
        backtranslated = np.array([1, 0, 0, 0, 1])

        rate = compute_preservation_rate(original, backtranslated)
        assert rate == 0.8  # 4/5 match

    def test_compute_false_positive_rate(self):
        """Test false positive rate computation."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])  # 1 FP out of 4 negatives

        fpr = compute_false_positive_rate(y_true, y_pred)
        assert fpr == 0.25

    def test_evaluate_model(self):
        """Test comprehensive model evaluation."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

        results = evaluate_model(
            model_name="test_model",
            dataset_name="test_dataset",
            content_type="test",
            y_true=y_true,
            y_pred=y_pred,
            compute_ci=False,
        )

        assert results.model_name == "test_model"
        assert 0 <= results.accuracy <= 1
        assert 0 <= results.f1 <= 1
        assert results.n_samples == 10


class TestStatistical:
    """Test statistical utilities."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        corrected_alpha, significant = bonferroni_correction(p_values, alpha=0.05)

        assert corrected_alpha == 0.01  # 0.05 / 5
        assert significant[0] == True  # 0.01 < 0.01
        assert significant[1] == False  # 0.02 > 0.01

    def test_paired_comparison_mcnemar(self):
        """Test paired comparison with McNemar's test."""
        results_1 = np.array([True, True, False, True, True])
        results_2 = np.array([True, False, False, True, True])

        result = paired_comparison(results_1, results_2, test="mcnemar")

        assert "statistic" in result
        assert "p_value" in result
        assert result["test"] == "mcnemar"

    def test_compute_effect_size(self):
        """Test effect size computation."""
        group_1 = np.array([5, 6, 7, 8, 9])
        group_2 = np.array([3, 4, 5, 6, 7])

        d = compute_effect_size(group_1, group_2, method="cohens_d")

        assert d > 0  # group_1 has higher mean


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        rate = compute_preservation_rate(y_true, y_pred)
        assert rate == 0.0

    def test_perfect_predictions(self):
        """Test perfect prediction scenario."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        f1 = compute_f1_score(y_true, y_pred)
        assert f1 == 1.0

        fpr = compute_false_positive_rate(y_true, y_pred)
        assert fpr == 0.0

    def test_all_wrong_predictions(self):
        """Test completely wrong predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        f1 = compute_f1_score(y_true, y_pred)
        assert f1 == 0.0

        fpr = compute_false_positive_rate(y_true, y_pred)
        assert fpr == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
