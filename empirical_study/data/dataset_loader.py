"""Base dataset loader class and utilities."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Sample:
    """Represents a single sample in the dataset."""

    text: str
    label: int
    sample_id: str
    source_dataset: str
    content_type: str  # 'explicit', 'implicit', 'clear', 'subtle', 'aae', 'sae'
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnnotationInfo:
    """Stores annotation agreement information."""

    annotator_1: int
    annotator_2: int
    final_label: int
    agreement: bool
    third_annotator: Optional[int] = None


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Sample] = []

    @abstractmethod
    def load(self) -> List[Sample]:
        """Load the dataset and return samples."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass

    def filter_by_agreement(
        self, samples: List[Sample], min_agreement: float = 0.5
    ) -> List[Sample]:
        """Filter samples based on annotator agreement if available."""
        filtered = []
        for sample in samples:
            if "agreement_score" in sample.metadata:
                if sample.metadata["agreement_score"] >= min_agreement:
                    filtered.append(sample)
            else:
                # If no agreement info, include the sample
                filtered.append(sample)
        return filtered

    def stratified_sample(
        self, samples: List[Sample], n: int, random_seed: int = 42
    ) -> List[Sample]:
        """Sample n examples maintaining label distribution."""
        np.random.seed(random_seed)

        # Group by label
        by_label: Dict[int, List[Sample]] = {}
        for sample in samples:
            if sample.label not in by_label:
                by_label[sample.label] = []
            by_label[sample.label].append(sample)

        # Calculate proportions
        total = len(samples)
        sampled = []

        for label, label_samples in by_label.items():
            proportion = len(label_samples) / total
            n_sample = max(1, int(n * proportion))
            n_sample = min(n_sample, len(label_samples))

            indices = np.random.choice(len(label_samples), n_sample, replace=False)
            sampled.extend([label_samples[i] for i in indices])

        # Adjust if we have too few or too many
        if len(sampled) < n:
            remaining = [s for s in samples if s not in sampled]
            extra_needed = n - len(sampled)
            if remaining:
                extra_indices = np.random.choice(
                    len(remaining), min(extra_needed, len(remaining)), replace=False
                )
                sampled.extend([remaining[i] for i in extra_indices])
        elif len(sampled) > n:
            indices = np.random.choice(len(sampled), n, replace=False)
            sampled = [sampled[i] for i in indices]

        return sampled

    def to_dataframe(self, samples: Optional[List[Sample]] = None) -> pd.DataFrame:
        """Convert samples to a pandas DataFrame."""
        if samples is None:
            samples = self.samples

        data = []
        for sample in samples:
            row = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "label": sample.label,
                "source_dataset": sample.source_dataset,
                "content_type": sample.content_type,
            }
            row.update(sample.metadata)
            data.append(row)

        return pd.DataFrame(data)


def compute_cohen_kappa(
    annotations_1: List[int], annotations_2: List[int]
) -> float:
    """Compute Cohen's Kappa for inter-annotator agreement."""
    assert len(annotations_1) == len(annotations_2)

    n = len(annotations_1)
    if n == 0:
        return 0.0

    # Get unique labels
    labels = sorted(set(annotations_1) | set(annotations_2))
    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Build confusion matrix
    confusion = np.zeros((n_labels, n_labels))
    for a1, a2 in zip(annotations_1, annotations_2):
        confusion[label_to_idx[a1], label_to_idx[a2]] += 1

    # Calculate observed agreement
    p_o = np.trace(confusion) / n

    # Calculate expected agreement
    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    p_e = np.sum(row_marginals * col_marginals)

    # Cohen's Kappa
    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)

    return kappa


def merge_datasets(datasets: List[Tuple[str, List[Sample]]]) -> pd.DataFrame:
    """Merge multiple datasets into a single DataFrame."""
    all_samples = []
    for name, samples in datasets:
        for sample in samples:
            all_samples.append(sample)

    loader = DatasetLoader.__new__(DatasetLoader)
    return loader.to_dataframe(all_samples)
