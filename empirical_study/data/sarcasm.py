"""Loaders for sarcasm datasets: SemEval-2018 Task 3 and iSarcasm."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from loguru import logger

from .dataset_loader import DatasetLoader, Sample


class SemEvalSarcasmLoader(DatasetLoader):
    """
    Loader for SemEval-2018 Task 3 (Irony Detection).
    Contains clear polarity contrast cases of sarcasm/irony.

    Reference: Van Hee et al. (2018). SemEval-2018 Task 3: Irony Detection
    in English Tweets.
    """

    DATASET_NAME = "semeval2018_task3"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
        subtask: str = "A",  # A: binary, B: multiclass
    ):
        super().__init__(cache_dir)
        self.split = split
        self.subtask = subtask
        self.data_path = self.cache_dir / "semeval2018_task3"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "SemEval-2018 Task 3"

    def load(self) -> List[Sample]:
        """Load SemEval-2018 Task 3 dataset."""
        logger.info(f"Loading SemEval-2018 Task 3 dataset ({self.split} split)")

        try:
            # Try loading tweet_eval irony subset from Hugging Face
            dataset = load_dataset(
                "tweet_eval",
                "irony",
                split=self.split,
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                text = item.get("text", "")
                label = item.get("label", 0)

                sample = Sample(
                    text=text,
                    label=label,
                    sample_id=f"semeval_irony_{idx}",
                    source_dataset="SemEval-2018 Task 3",
                    content_type="clear",  # Clear polarity contrast
                    metadata={
                        "original_idx": idx,
                    },
                )
                samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} samples from SemEval-2018 Task 3")
            return samples

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return self._load_from_local()

    def _load_from_local(self) -> List[Sample]:
        """Load from local files."""
        txt_path = self.data_path / f"SemEval2018-T3-{self.split}-taskA.txt"

        if txt_path.exists():
            return self._load_from_txt(txt_path)

        logger.warning(
            "SemEval-2018 Task 3 data not found. Creating placeholder."
        )
        logger.info(f"To use real data, place files in: {self.data_path}")

        self.samples = []
        return self.samples

    def _load_from_txt(self, path: Path) -> List[Sample]:
        """Load from official SemEval format."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    tweet_id, label, text = row[0], int(row[1]), row[2]
                    sample = Sample(
                        text=text,
                        label=label,
                        sample_id=f"semeval_irony_{tweet_id}",
                        source_dataset="SemEval-2018 Task 3",
                        content_type="clear",
                        metadata={"tweet_id": tweet_id},
                    )
                    samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from SemEval TXT")
        return samples

    def filter_clear_polarity(self, samples: List[Sample]) -> List[Sample]:
        """Filter to samples with clear polarity contrast."""
        clear = []
        for sample in samples:
            # Clear polarity contrast typically involves sentiment words
            text_lower = sample.text.lower()
            has_positive = any(
                w in text_lower
                for w in ["great", "love", "amazing", "wonderful", "perfect", "best"]
            )
            has_negative_context = any(
                w in text_lower
                for w in ["not", "but", "however", "unfortunately", "fail", "worst"]
            )

            # Include if labeled ironic or has clear contrast markers
            if sample.label == 1 or (has_positive and has_negative_context):
                clear.append(sample)
            elif sample.label == 0:
                # Include some non-ironic for balance
                clear.append(sample)

        return clear


class ISarcasmLoader(DatasetLoader):
    """
    Loader for iSarcasm dataset.
    Contains subtle/contextual sarcasm cases requiring deeper understanding.

    Reference: Oprea and Magdy (2020). iSarcasm: A Dataset of Intended Sarcasm.
    """

    DATASET_NAME = "isarcasm"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
    ):
        super().__init__(cache_dir)
        self.split = split
        self.data_path = self.cache_dir / "isarcasm"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "iSarcasm"

    def load(self) -> List[Sample]:
        """Load iSarcasm dataset."""
        logger.info(f"Loading iSarcasm dataset ({self.split} split)")

        try:
            # iSarcasm is available on HuggingFace
            dataset = load_dataset(
                "isarcasm",
                split=self.split if self.split != "test" else "train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                text = item.get("tweet", item.get("text", ""))
                label = item.get("label", 0)
                # iSarcasm has: sarcastic (1), not_sarcastic (0)

                sample = Sample(
                    text=text,
                    label=label,
                    sample_id=f"isarcasm_{idx}",
                    source_dataset="iSarcasm",
                    content_type="subtle",  # Subtle/contextual sarcasm
                    metadata={
                        "sarcasm_type": item.get("sarcasm_type", ""),
                        "rephrase": item.get("rephrase", ""),
                    },
                )
                samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} samples from iSarcasm")
            return samples

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return self._load_from_local()

    def _load_from_local(self) -> List[Sample]:
        """Load from local CSV file."""
        csv_path = self.data_path / f"{self.split}.csv"

        if csv_path.exists():
            return self._load_from_csv(csv_path)

        logger.warning("iSarcasm data not found. Creating placeholder.")
        logger.info(f"To use real data, place files in: {self.data_path}")

        self.samples = []
        return self.samples

    def _load_from_csv(self, path: Path) -> List[Sample]:
        """Load from CSV file."""
        samples = []

        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            sample = Sample(
                text=row["tweet"],
                label=int(row["label"]),
                sample_id=f"isarcasm_{idx}",
                source_dataset="iSarcasm",
                content_type="subtle",
                metadata={
                    "sarcasm_type": row.get("sarcasm_type", ""),
                    "dialect": row.get("dialect", ""),
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from iSarcasm CSV")
        return samples

    def filter_subtle(self, samples: List[Sample]) -> List[Sample]:
        """Filter to samples with subtle/contextual sarcasm."""
        subtle = []
        for sample in samples:
            # Subtle sarcasm lacks obvious markers
            text_lower = sample.text.lower()
            has_obvious_markers = any(
                marker in text_lower
                for marker in ["yeah right", "sure...", "/s", "#sarcasm", "not!"]
            )

            if sample.label == 1 and not has_obvious_markers:
                subtle.append(sample)
            elif sample.label == 0:
                subtle.append(sample)

        return subtle


def load_sarcasm_data(
    cache_dir: str = "./data/cache",
    n_clear: int = 100,
    n_subtle: int = 100,
    random_seed: int = 42,
) -> Dict[str, List[Sample]]:
    """
    Load and prepare sarcasm data for the study.

    Returns:
        Dictionary with 'clear' and 'subtle' sample lists.
    """
    # Load SemEval-2018 Task 3 for clear sarcasm
    semeval_loader = SemEvalSarcasmLoader(cache_dir=cache_dir)
    semeval_samples = semeval_loader.load()
    clear_samples = semeval_loader.filter_clear_polarity(semeval_samples)
    clear_samples = semeval_loader.stratified_sample(
        clear_samples, n_clear, random_seed
    )

    # Load iSarcasm for subtle sarcasm
    isarcasm_loader = ISarcasmLoader(cache_dir=cache_dir)
    isarcasm_samples = isarcasm_loader.load()
    subtle_samples = isarcasm_loader.filter_subtle(isarcasm_samples)
    subtle_samples = isarcasm_loader.stratified_sample(
        subtle_samples, n_subtle, random_seed
    )

    return {
        "clear": clear_samples,
        "subtle": subtle_samples,
    }
