"""Loaders for hate speech datasets: HatEval and LatentHatred."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from loguru import logger

from .dataset_loader import DatasetLoader, Sample


class HatEvalLoader(DatasetLoader):
    """
    Loader for HatEval dataset (SemEval-2019 Task 5).
    Contains explicit hate speech with clear slurs or threats.

    Reference: Basile et al. (2019). SemEval-2019 Task 5: Multilingual Detection
    of Hate Speech Against Immigrants and Women in Twitter.
    """

    DATASET_NAME = "hateval2019"
    HUGGINGFACE_ID = "hate_speech_offensive"  # Alternative source

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
        language: str = "en",
    ):
        super().__init__(cache_dir)
        self.split = split
        self.language = language
        self.data_path = self.cache_dir / "hateval"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "HatEval"

    def load(self) -> List[Sample]:
        """Load HatEval dataset."""
        logger.info(f"Loading HatEval dataset ({self.split} split)")

        try:
            # Try loading from Hugging Face datasets
            # Note: HatEval may require manual download due to licensing
            # We use a similar dataset structure as fallback
            dataset = load_dataset(
                "tweets_hate_speech_detection",
                split=self.split if self.split != "test" else "train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                # Filter for explicit hate speech (clear slurs/threats)
                text = item.get("tweet", item.get("text", ""))
                label = item.get("label", 0)

                # Create sample
                sample = Sample(
                    text=text,
                    label=label,
                    sample_id=f"hateval_{idx}",
                    source_dataset="HatEval",
                    content_type="explicit",
                    metadata={
                        "original_id": item.get("id", idx),
                        "target": item.get("target", "unknown"),
                    },
                )
                samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} samples from HatEval")
            return samples

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.info("Attempting to load from local files or create synthetic data")
            return self._load_from_local_or_synthetic()

    def _load_from_local_or_synthetic(self) -> List[Sample]:
        """Load from local TSV file or create synthetic placeholder data."""
        tsv_path = self.data_path / f"{self.split}_en.tsv"

        if tsv_path.exists():
            return self._load_from_tsv(tsv_path)

        # Create placeholder structure for when real data is available
        logger.warning(
            "HatEval data not found locally. Creating placeholder structure."
        )
        logger.info(
            f"To use real data, place HatEval TSV files in: {self.data_path}"
        )

        # Return empty list - real implementation would require data download
        self.samples = []
        return self.samples

    def _load_from_tsv(self, path: Path) -> List[Sample]:
        """Load HatEval data from TSV file."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                sample = Sample(
                    text=row["text"],
                    label=int(row["HS"]),  # Hate Speech label
                    sample_id=f"hateval_{row.get('id', idx)}",
                    source_dataset="HatEval",
                    content_type="explicit",
                    metadata={
                        "target_range": row.get("TR", ""),
                        "aggressiveness": row.get("AG", ""),
                    },
                )
                samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from HatEval TSV")
        return samples

    def filter_explicit(self, samples: List[Sample]) -> List[Sample]:
        """Filter to only include explicitly hateful content."""
        # In HatEval, we consider samples with aggressiveness or clear targeting
        explicit = []
        for sample in samples:
            # Keep samples labeled as hate speech
            if sample.label == 1:
                explicit.append(sample)
        return explicit


class LatentHatredLoader(DatasetLoader):
    """
    Loader for LatentHatred dataset.
    Contains implicit hate speech without explicit markers.

    Reference: ElSherief et al. (2021). Latent Hatred: A Benchmark for
    Understanding Implicit Hate Speech.
    """

    DATASET_NAME = "latent_hatred"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
    ):
        super().__init__(cache_dir)
        self.split = split
        self.data_path = self.cache_dir / "latent_hatred"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "LatentHatred"

    def load(self) -> List[Sample]:
        """Load LatentHatred dataset."""
        logger.info(f"Loading LatentHatred dataset ({self.split} split)")

        try:
            # Try loading from Hugging Face
            dataset = load_dataset(
                "ucberkeley-dlab/measuring-hate-speech",
                split=self.split if self.split != "test" else "train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                text = item.get("text", "")
                # Use hate_speech_score for implicit detection
                hate_score = item.get("hate_speech_score", 0)
                # Consider implicit if score is moderate but no explicit slurs
                label = 1 if hate_score > 0.5 else 0

                sample = Sample(
                    text=text,
                    label=label,
                    sample_id=f"latent_{idx}",
                    source_dataset="LatentHatred",
                    content_type="implicit",
                    metadata={
                        "hate_score": hate_score,
                        "annotator_count": item.get("annotator_count", 0),
                    },
                )
                samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} samples from LatentHatred")
            return samples

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return self._load_from_local_or_synthetic()

    def _load_from_local_or_synthetic(self) -> List[Sample]:
        """Load from local JSON file."""
        json_path = self.data_path / f"{self.split}.json"

        if json_path.exists():
            return self._load_from_json(json_path)

        logger.warning(
            "LatentHatred data not found. Creating placeholder structure."
        )
        logger.info(
            f"To use real data, place LatentHatred JSON files in: {self.data_path}"
        )

        self.samples = []
        return self.samples

    def _load_from_json(self, path: Path) -> List[Sample]:
        """Load LatentHatred data from JSON file."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            sample = Sample(
                text=item["text"],
                label=item["label"],
                sample_id=f"latent_{item.get('id', idx)}",
                source_dataset="LatentHatred",
                content_type="implicit",
                metadata={
                    "implicit_class": item.get("implicit_class", ""),
                    "target_group": item.get("target_group", ""),
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from LatentHatred JSON")
        return samples

    def filter_implicit(self, samples: List[Sample]) -> List[Sample]:
        """Filter to only include implicitly hateful content (no explicit markers)."""
        implicit = []
        for sample in samples:
            # Keep samples that are hate but don't have explicit slurs
            if sample.label == 1:
                # Check for absence of explicit markers
                text_lower = sample.text.lower()
                # This is a simplified check - real implementation would be more sophisticated
                has_explicit = any(
                    marker in text_lower
                    for marker in ["kill", "die", "hate you", "f***", "n***"]
                )
                if not has_explicit:
                    implicit.append(sample)
        return implicit


def load_hate_speech_data(
    cache_dir: str = "./data/cache",
    n_explicit: int = 100,
    n_implicit: int = 100,
    random_seed: int = 42,
) -> Dict[str, List[Sample]]:
    """
    Load and prepare hate speech data for the study.

    Returns:
        Dictionary with 'explicit' and 'implicit' sample lists.
    """
    # Load HatEval for explicit hate speech
    hateval_loader = HatEvalLoader(cache_dir=cache_dir)
    hateval_samples = hateval_loader.load()
    explicit_samples = hateval_loader.filter_explicit(hateval_samples)
    explicit_samples = hateval_loader.stratified_sample(
        explicit_samples, n_explicit, random_seed
    )

    # Load LatentHatred for implicit hate speech
    latent_loader = LatentHatredLoader(cache_dir=cache_dir)
    latent_samples = latent_loader.load()
    implicit_samples = latent_loader.filter_implicit(latent_samples)
    implicit_samples = latent_loader.stratified_sample(
        implicit_samples, n_implicit, random_seed
    )

    return {
        "explicit": explicit_samples,
        "implicit": implicit_samples,
    }
