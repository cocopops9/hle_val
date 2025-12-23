"""Loader for dialectal data: Sap et al. African American English dataset."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from loguru import logger

from .dataset_loader import DatasetLoader, Sample


class SapDialectalLoader(DatasetLoader):
    """
    Loader for Sap et al. (2019) dialectal dataset.
    Contains African American English (AAE) and matched Standard American English (SAE).

    Reference: Sap et al. (2019). The Risk of Racial Bias in Hate Speech Detection.
    """

    DATASET_NAME = "sap_dialectal"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
    ):
        super().__init__(cache_dir)
        self.split = split
        self.data_path = self.cache_dir / "sap_dialectal"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "Sap et al. Dialectal"

    def load(self) -> List[Sample]:
        """Load Sap et al. dialectal dataset."""
        logger.info(f"Loading Sap et al. dialectal dataset ({self.split} split)")

        try:
            # Try loading from HuggingFace (social_bias_frames contains some of this)
            dataset = load_dataset(
                "social_bias_frames",
                split=self.split if self.split != "test" else "train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                text = item.get("post", "")
                # Infer dialect from available metadata
                dialect = self._infer_dialect(item)

                if dialect in ["aae", "sae"]:
                    # Label: whether annotators marked as offensive
                    label = 1 if item.get("offensiveYN", "") == "1.0" else 0

                    sample = Sample(
                        text=text,
                        label=label,
                        sample_id=f"sap_{dialect}_{idx}",
                        source_dataset="Sap et al.",
                        content_type=dialect,
                        metadata={
                            "dialect": dialect,
                            "target_minority": item.get("targetMinority", ""),
                            "intent": item.get("intentYN", ""),
                        },
                    )
                    samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} samples from Sap et al.")
            return samples

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return self._load_from_local()

    def _infer_dialect(self, item: dict) -> str:
        """Infer dialect from item metadata."""
        # This is a simplified inference - real implementation would use
        # the dialect annotations from the original paper
        text = item.get("post", "").lower()

        # Simple heuristics for AAE features (this would be replaced with
        # actual dialect annotations in production)
        aae_markers = [
            "finna", "tryna", "gonna", "gotta", "aint", "ain't",
            "bruh", "fam", "lowkey", "deadass", "ion", "issa",
        ]

        if any(marker in text for marker in aae_markers):
            return "aae"
        return "sae"

    def _load_from_local(self) -> List[Sample]:
        """Load from local files."""
        csv_path = self.data_path / f"{self.split}.csv"

        if csv_path.exists():
            return self._load_from_csv(csv_path)

        logger.warning("Sap et al. dialectal data not found. Creating placeholder.")
        logger.info(f"To use real data, place files in: {self.data_path}")

        self.samples = []
        return self.samples

    def _load_from_csv(self, path: Path) -> List[Sample]:
        """Load from CSV file with dialect annotations."""
        samples = []

        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            dialect = row.get("dialect", "sae").lower()
            sample = Sample(
                text=row["text"],
                label=int(row["label"]),
                sample_id=f"sap_{dialect}_{idx}",
                source_dataset="Sap et al.",
                content_type=dialect,
                metadata={
                    "dialect": dialect,
                    "race_of_author": row.get("race", ""),
                    "matched_pair_id": row.get("pair_id", ""),
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from Sap et al. CSV")
        return samples

    def get_dialect_pairs(self) -> List[Tuple[Sample, Sample]]:
        """Get matched AAE-SAE pairs for controlled comparison."""
        aae_samples = [s for s in self.samples if s.content_type == "aae"]
        sae_samples = [s for s in self.samples if s.content_type == "sae"]

        pairs = []
        for aae in aae_samples:
            pair_id = aae.metadata.get("matched_pair_id")
            if pair_id:
                matched_sae = next(
                    (s for s in sae_samples
                     if s.metadata.get("matched_pair_id") == pair_id),
                    None
                )
                if matched_sae:
                    pairs.append((aae, matched_sae))

        return pairs


def load_dialectal_data(
    cache_dir: str = "./data/cache",
    n_aae: int = 100,
    random_seed: int = 42,
) -> Dict[str, List[Sample]]:
    """
    Load and prepare dialectal data for the study.

    Returns:
        Dictionary with 'aae' and 'sae' sample lists.
    """
    loader = SapDialectalLoader(cache_dir=cache_dir)
    all_samples = loader.load()

    aae_samples = [s for s in all_samples if s.content_type == "aae"]
    sae_samples = [s for s in all_samples if s.content_type == "sae"]

    # Sample equal numbers
    aae_samples = loader.stratified_sample(aae_samples, n_aae, random_seed)
    sae_samples = loader.stratified_sample(sae_samples, n_aae, random_seed)

    return {
        "aae": aae_samples,
        "sae": sae_samples,
    }
