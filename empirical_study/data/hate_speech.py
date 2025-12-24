"""Loaders for hate speech datasets: HateXplain (explicit) and ImplicitHate (implicit)."""

import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger

from .dataset_loader import DatasetLoader, Sample


class HateXplainLoader(DatasetLoader):
    """
    Loader for HateXplain dataset (explicit hate speech).

    Contains explicit hate speech with clear hateful content, slurs, or offensive language.
    Uses majority voting across 3 annotators for final labels.

    Reference: Mathew et al. (2021). HateXplain: A Benchmark Dataset for
    Explainable Hate Speech Detection. AAAI 2021.

    GitHub: https://github.com/hate-alert/HateXplain
    """

    GITHUB_RAW_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        split: str = "test",
        use_binary_labels: bool = True,  # hate+offensive vs normal
    ):
        super().__init__(cache_dir)
        self.split = split
        self.use_binary_labels = use_binary_labels
        self.data_path = self.cache_dir / "hatexplain"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "HateXplain"

    def load(self) -> List[Sample]:
        """Load HateXplain dataset."""
        logger.info(f"Loading HateXplain dataset ({self.split} split)")

        try:
            # Try loading from Hugging Face first
            dataset = load_dataset(
                "hatexplain",
                split=self.split,
                cache_dir=str(self.cache_dir),
            )
            return self._process_hf_dataset(dataset)

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.info("Attempting to load from local files...")
            return self._load_from_local()

    def _process_hf_dataset(self, dataset) -> List[Sample]:
        """Process HuggingFace dataset format."""
        samples = []

        for idx, item in enumerate(dataset):
            # Get post text (join tokens)
            if "post_tokens" in item:
                text = " ".join(item["post_tokens"])
            else:
                text = item.get("text", "")

            # Get majority label from annotators
            if "annotators" in item:
                labels = []
                for ann in item["annotators"]:
                    if isinstance(ann, dict):
                        labels.append(ann.get("label", 0))
                    else:
                        labels.append(ann)
                if labels:
                    majority_label = Counter(labels).most_common(1)[0][0]
                else:
                    majority_label = 0
            else:
                majority_label = item.get("label", 0)

            # Convert to binary if requested
            # In HateXplain HF: 0=hatespeech, 1=normal, 2=offensive
            # We want: 1=hate/offensive, 0=normal
            if self.use_binary_labels:
                if isinstance(majority_label, int):
                    binary_label = 0 if majority_label == 1 else 1
                else:
                    # String labels
                    binary_label = 0 if majority_label == "normal" else 1
            else:
                binary_label = majority_label

            # Get target groups if available
            target_groups = []
            if "annotators" in item:
                for ann in item["annotators"]:
                    if isinstance(ann, dict) and "target" in ann:
                        targets = ann["target"]
                        if isinstance(targets, list):
                            target_groups.extend(targets)
                        else:
                            target_groups.append(targets)

            sample = Sample(
                text=text,
                label=binary_label,
                sample_id=f"hatexplain_{item.get('id', idx)}",
                source_dataset="HateXplain",
                content_type="explicit",
                metadata={
                    "original_label": majority_label,
                    "target_groups": list(set(target_groups)) if target_groups else [],
                    "annotator_count": len(item.get("annotators", [])),
                },
            )
            samples.append(sample)

        self.samples = samples

        # Log distribution
        pos = sum(1 for s in samples if s.label == 1)
        neg = sum(1 for s in samples if s.label == 0)
        logger.info(f"Loaded {len(samples)} samples: {pos} hate/offensive, {neg} normal")

        return samples

    def _load_from_local(self) -> List[Sample]:
        """Load from local JSON file."""
        json_path = self.data_path / "dataset.json"

        if not json_path.exists():
            # Try to download
            logger.info("Downloading HateXplain dataset...")
            self._download_dataset()

        if json_path.exists():
            return self._load_from_json(json_path)

        logger.warning("HateXplain data not found. Please download from GitHub.")
        logger.info(f"Expected location: {json_path}")
        self.samples = []
        return self.samples

    def _download_dataset(self):
        """Download dataset from GitHub."""
        import urllib.request

        url = f"{self.GITHUB_RAW_URL}/dataset.json"
        json_path = self.data_path / "dataset.json"

        try:
            logger.info(f"Downloading from {url}")
            urllib.request.urlretrieve(url, json_path)
            logger.info(f"Downloaded to {json_path}")
        except Exception as e:
            logger.error(f"Failed to download: {e}")

    def _load_from_json(self, path: Path) -> List[Sample]:
        """Load from local JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load split IDs if available
        split_ids = self._load_split_ids()

        samples = []
        for post_id, item in data.items():
            # Filter by split if we have split info
            if split_ids and post_id not in split_ids.get(self.split, set()):
                continue

            # Get text from tokens
            text = " ".join(item.get("post_tokens", []))

            # Get majority label
            labels = [ann.get("label") for ann in item.get("annotators", [])]
            if labels:
                majority_label = Counter(labels).most_common(1)[0][0]
            else:
                continue

            # Map labels: hatespeech->1, offensive->1, normal->0
            label_map = {"hatespeech": 1, "offensive": 1, "normal": 0}
            binary_label = label_map.get(majority_label, 0)

            sample = Sample(
                text=text,
                label=binary_label,
                sample_id=f"hatexplain_{post_id}",
                source_dataset="HateXplain",
                content_type="explicit",
                metadata={
                    "original_label": majority_label,
                    "post_id": post_id,
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from local JSON")
        return samples

    def _load_split_ids(self) -> Optional[Dict[str, set]]:
        """Load train/val/test split IDs."""
        split_path = self.data_path / "post_id_divisions.json"

        if not split_path.exists():
            return None

        with open(split_path, "r") as f:
            splits = json.load(f)

        return {
            "train": set(splits.get("train", [])),
            "val": set(splits.get("val", [])),
            "test": set(splits.get("test", [])),
        }


class ImplicitHateLoader(DatasetLoader):
    """
    Loader for Implicit Hate Corpus (IHC).

    Contains implicit hate speech without explicit slurs or obvious markers.
    Categorized into: grievance, incitement, inferiority, irony, stereotypes, threats.

    Reference: ElSherief et al. (2021). Latent Hatred: A Benchmark for
    Understanding Implicit Hate Speech. EMNLP 2021.

    GitHub: https://github.com/SALT-NLP/implicit-hate
    HuggingFace: https://huggingface.co/datasets/SALT-NLP/ImplicitHate
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        local_path: Optional[str] = None,  # Path to local data file or zip
        split: str = "train",  # IHC doesn't have official splits
    ):
        super().__init__(cache_dir)
        self.split = split
        self.local_path = local_path
        self.data_path = self.cache_dir / "implicit_hate"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "ImplicitHate"

    def load(self) -> List[Sample]:
        """Load Implicit Hate Corpus."""
        logger.info("Loading Implicit Hate Corpus...")

        # First try local path if provided
        if self.local_path:
            local = Path(self.local_path)
            if local.exists():
                if local.suffix == ".zip":
                    return self._load_from_zip(self.local_path)
                else:
                    return self._load_from_local_file(local)

        # Check data directory for any data files
        for pattern in ["*.tsv", "*.csv", "*.json"]:
            files = list(self.data_path.glob(pattern)) + list(self.data_path.glob(f"**/{pattern}"))
            if files:
                return self._load_from_local_file(files[0])

        # Try HuggingFace
        try:
            logger.info("Loading from HuggingFace: SALT-NLP/ImplicitHate")
            dataset = load_dataset(
                "SALT-NLP/ImplicitHate",
                split=self.split,
                cache_dir=str(self.cache_dir),
            )
            return self._process_hf_dataset(dataset)

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.info(f"Please place data files in: {self.data_path}")
            self.samples = []
            return self.samples

    def _process_hf_dataset(self, dataset) -> List[Sample]:
        """Process HuggingFace dataset format."""
        samples = []

        for idx, item in enumerate(dataset):
            text = item.get("post", "")
            implicit_class = item.get("implicit_class", "")

            # All items in ImplicitHate are implicit hate (label=1)
            sample = Sample(
                text=text,
                label=1,  # All are implicit hate
                sample_id=f"implicit_{idx}",
                source_dataset="ImplicitHate",
                content_type="implicit",
                metadata={
                    "implicit_class": implicit_class,
                    "extra_class": item.get("extra_implicit_class", ""),
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} implicit hate samples")

        # Log class distribution
        class_dist = Counter(s.metadata.get("implicit_class", "") for s in samples)
        for cls, count in class_dist.most_common():
            logger.info(f"  {cls}: {count}")

        return samples

    def _load_from_zip(self, zip_path: str) -> List[Sample]:
        """Load from zip file."""
        logger.info(f"Extracting {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)

        # Find the data file
        for pattern in ["*.tsv", "*.csv", "*.json"]:
            files = list(self.data_path.rglob(pattern))
            if files:
                return self._load_from_local_file(files[0])

        logger.warning("No data files found in zip")
        return []

    def _load_from_local_file(self, file_path: Path) -> List[Sample]:
        """Load from local TSV/CSV/JSON file."""
        logger.info(f"Loading from {file_path}")

        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            # TSV or CSV
            sep = "\t" if file_path.suffix == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)

        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Shape: {df.shape}")

        samples = []

        # Try to identify text column
        text_col = None
        for col in ["post", "text", "tweet", "content", "sentence"]:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[0]

        # Check for implicit class column
        class_col = None
        for col in ["implicit_class", "class", "label", "category", "type"]:
            if col in df.columns:
                class_col = col
                break

        for idx, row in df.iterrows():
            text = str(row[text_col])
            implicit_class = str(row[class_col]) if class_col else "unknown"

            sample = Sample(
                text=text,
                label=1,  # All are implicit hate
                sample_id=f"implicit_{idx}",
                source_dataset="ImplicitHate",
                content_type="implicit",
                metadata={
                    "implicit_class": implicit_class,
                },
            )
            samples.append(sample)

        self.samples = samples
        logger.info(f"Loaded {len(samples)} samples from local file")
        return samples


class NonHateLoader(DatasetLoader):
    """
    Loader for non-hate speech samples to balance the dataset.

    Uses neutral tweets from Davidson et al. or TweetEval.
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        source: str = "davidson",  # "davidson" or "tweeteval"
    ):
        super().__init__(cache_dir)
        self.source = source
        self.data_path = self.cache_dir / "non_hate"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "NonHate"

    def load(self) -> List[Sample]:
        """Load non-hate samples."""
        logger.info(f"Loading non-hate samples from {self.source}")

        if self.source == "davidson":
            return self._load_davidson_normal()
        elif self.source == "tweeteval":
            return self._load_tweeteval_neutral()
        else:
            return self._load_davidson_normal()

    def _load_davidson_normal(self) -> List[Sample]:
        """Load normal (non-hate) samples from Davidson et al. dataset."""
        try:
            # Davidson dataset has hate_speech(0), offensive(1), neither(2)
            dataset = load_dataset(
                "hate_speech_offensive",
                split="train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                # class 2 = neither (normal/non-hate)
                if item.get("class", -1) == 2:
                    sample = Sample(
                        text=item.get("tweet", ""),
                        label=0,  # Not hate
                        sample_id=f"davidson_normal_{idx}",
                        source_dataset="Davidson",
                        content_type="normal",
                        metadata={},
                    )
                    samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} non-hate samples from Davidson")
            return samples

        except Exception as e:
            logger.warning(f"Could not load Davidson dataset: {e}")
            return self._load_tweeteval_neutral()

    def _load_tweeteval_neutral(self) -> List[Sample]:
        """Load neutral tweets from TweetEval sentiment."""
        try:
            dataset = load_dataset(
                "tweet_eval",
                "sentiment",
                split="train",
                cache_dir=str(self.cache_dir),
            )

            samples = []
            for idx, item in enumerate(dataset):
                # Neutral sentiment (label=1) - less likely to contain hate
                if item.get("label", -1) == 1:
                    sample = Sample(
                        text=item.get("text", ""),
                        label=0,  # Not hate
                        sample_id=f"tweeteval_neutral_{idx}",
                        source_dataset="TweetEval",
                        content_type="normal",
                        metadata={},
                    )
                    samples.append(sample)

            self.samples = samples
            logger.info(f"Loaded {len(samples)} neutral samples from TweetEval")
            return samples

        except Exception as e:
            logger.warning(f"Could not load TweetEval dataset: {e}")
            return []


def load_hate_speech_data(
    cache_dir: str = "./data/cache",
    n_explicit: int = 100,
    n_implicit: int = 100,
    random_seed: int = 42,
    implicit_data_path: Optional[str] = None,
) -> Dict[str, List[Sample]]:
    """
    Load and prepare hate speech data for the study.

    Uses HateXplain for explicit hate and ImplicitHate for implicit hate.
    Balances with non-hate samples.

    Args:
        cache_dir: Cache directory for datasets
        n_explicit: Number of explicit samples (50% hate, 50% non-hate)
        n_implicit: Number of implicit samples (50% hate, 50% non-hate)
        random_seed: Random seed for sampling
        implicit_data_path: Optional path to local implicit hate corpus

    Returns:
        Dictionary with 'explicit' and 'implicit' sample lists.
    """
    np.random.seed(random_seed)

    # Load HateXplain for explicit hate speech
    logger.info("=== Loading Explicit Hate Speech (HateXplain) ===")
    hatexplain_loader = HateXplainLoader(cache_dir=cache_dir)
    hatexplain_samples = hatexplain_loader.load()

    # Load ImplicitHate for implicit hate speech
    logger.info("=== Loading Implicit Hate Speech (ImplicitHate) ===")
    implicit_loader = ImplicitHateLoader(
        cache_dir=cache_dir,
        local_path=implicit_data_path,
    )
    implicit_samples = implicit_loader.load()

    # Load non-hate samples for balancing
    logger.info("=== Loading Non-Hate Samples ===")
    nonhate_loader = NonHateLoader(cache_dir=cache_dir)
    nonhate_samples = nonhate_loader.load()

    # === Build Explicit Dataset (balanced) ===
    logger.info("=== Building Balanced Explicit Dataset ===")
    explicit_hate = [s for s in hatexplain_samples if s.label == 1]
    explicit_nonhate = [s for s in hatexplain_samples if s.label == 0]

    # If not enough non-hate from HateXplain, use Davidson
    if len(explicit_nonhate) < n_explicit // 2:
        logger.info(f"Supplementing with {len(nonhate_samples)} Davidson non-hate samples")
        explicit_nonhate.extend(nonhate_samples[:n_explicit])

    # Sample balanced
    n_per_class = n_explicit // 2
    logger.info(f"Sampling {n_per_class} per class for explicit dataset")

    if len(explicit_hate) >= n_per_class:
        indices = np.random.choice(len(explicit_hate), n_per_class, replace=False)
        explicit_hate = [explicit_hate[i] for i in indices]
    else:
        logger.warning(f"Only {len(explicit_hate)} explicit hate samples available")

    if len(explicit_nonhate) >= n_per_class:
        indices = np.random.choice(len(explicit_nonhate), n_per_class, replace=False)
        explicit_nonhate = [explicit_nonhate[i] for i in indices]
    else:
        logger.warning(f"Only {len(explicit_nonhate)} explicit non-hate samples available")

    explicit = explicit_hate + explicit_nonhate
    for s in explicit:
        s.content_type = "explicit"
    np.random.shuffle(explicit)

    # === Build Implicit Dataset (balanced) ===
    logger.info("=== Building Balanced Implicit Dataset ===")
    implicit_hate = implicit_samples.copy()

    # Use remaining non-hate samples for implicit balancing
    used_ids = {s.sample_id for s in explicit_nonhate}
    implicit_nonhate = [s for s in nonhate_samples if s.sample_id not in used_ids]

    n_per_class = n_implicit // 2
    logger.info(f"Sampling {n_per_class} per class for implicit dataset")

    if len(implicit_hate) >= n_per_class:
        indices = np.random.choice(len(implicit_hate), n_per_class, replace=False)
        implicit_hate = [implicit_hate[i] for i in indices]
    else:
        logger.warning(f"Only {len(implicit_hate)} implicit hate samples available")

    if len(implicit_nonhate) >= n_per_class:
        indices = np.random.choice(len(implicit_nonhate), n_per_class, replace=False)
        implicit_nonhate = [implicit_nonhate[i] for i in indices]
    else:
        logger.warning(f"Only {len(implicit_nonhate)} implicit non-hate samples available")

    implicit = implicit_hate + implicit_nonhate
    for s in implicit:
        s.content_type = "implicit"
    np.random.shuffle(implicit)

    # Log final distribution
    exp_pos = sum(1 for s in explicit if s.label == 1)
    exp_neg = sum(1 for s in explicit if s.label == 0)
    imp_pos = sum(1 for s in implicit if s.label == 1)
    imp_neg = sum(1 for s in implicit if s.label == 0)

    logger.info("=== Final Dataset Distribution ===")
    logger.info(f"Explicit: {exp_pos} hate + {exp_neg} non-hate = {len(explicit)}")
    logger.info(f"Implicit: {imp_pos} hate + {imp_neg} non-hate = {len(implicit)}")

    return {
        "explicit": explicit,
        "implicit": implicit,
    }
