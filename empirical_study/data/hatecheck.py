"""
HateCheck Dataset Loader

HateCheck (Röttger et al., 2021) is a functional test suite for hate speech detection
with 3,728 test cases across 29 functional tests.

Reference: Röttger et al. (2021). HateCheck: Functional Tests for Hate Speech Detection Models.
ACL 2021.

GitHub: https://github.com/paul-rottger/hatecheck-data
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from .dataset_loader import DatasetLoader, Sample


# HateCheck functional test definitions
# Using actual functionality names from the dataset
FUNCTIONAL_TESTS = {
    # Hateful content tests
    "derog_neg_emote_h": {"name": "Expression of strong negative emotions", "hateful": True},
    "derog_neg_attrib_h": {"name": "Description using very negative attributes", "hateful": True},
    "derog_dehum_h": {"name": "Dehumanisation", "hateful": True},
    "derog_impl_h": {"name": "Implicit derogation", "hateful": True},
    "threat_dir_h": {"name": "Direct threat", "hateful": True},
    "threat_norm_h": {"name": "Threat as normative statement", "hateful": True},
    "slur_h": {"name": "Hate expressed using slur", "hateful": True},

    # Non-hateful with slurs
    "slur_homonym_nh": {"name": "Non-hateful homonyms of slurs", "hateful": False},
    "slur_reclaimed_nh": {"name": "Reclaimed slurs", "hateful": False},

    # Profanity
    "profanity_h": {"name": "Hate expressed using profanity", "hateful": True},
    "profanity_nh": {"name": "Non-hateful use of profanity", "hateful": False},

    # Reference and phrasing
    "ref_subs_clause_h": {"name": "Hate expressed through reference in subsequent clauses", "hateful": True},
    "ref_subs_sent_h": {"name": "Hate expressed through reference in subsequent sentences", "hateful": True},
    "negate_neg_nh": {"name": "Negated hate", "hateful": False},
    "negate_pos_h": {"name": "Non-negated hate statement", "hateful": True},
    "phrase_question_h": {"name": "Hate phrased as a question", "hateful": True},
    "phrase_opinion_h": {"name": "Hate phrased as an opinion", "hateful": True},

    # Identity mentions
    "ident_neutral_nh": {"name": "Neutral statements using protected group identifier", "hateful": False},
    "ident_pos_nh": {"name": "Positive statements using protected group identifier", "hateful": False},

    # Counter speech
    "counter_quote_nh": {"name": "Counter speech quoting hate", "hateful": False},
    "counter_ref_nh": {"name": "Counter speech referencing hate", "hateful": False},

    # Target variations
    "target_obj_nh": {"name": "Abuse targeted at objects", "hateful": False},
    "target_indiv_nh": {"name": "Abuse targeted at individuals", "hateful": False},
    "target_group_nh": {"name": "Abuse targeted at non-protected group", "hateful": False},

    # Spelling variations
    "spell_space_add_h": {"name": "Hate with added spaces", "hateful": True},
    "spell_space_del_h": {"name": "Hate with removed spaces", "hateful": True},
    "spell_char_swap_h": {"name": "Hate with swapped characters", "hateful": True},
    "spell_char_del_h": {"name": "Hate with deleted characters", "hateful": True},
    "spell_leet_h": {"name": "Hate with leet speak", "hateful": True},
}

# Data limitation test groups (using actual functionality names)
DATA_LIMITATION_TESTS = {
    "class_imbalance": ["slur_reclaimed_nh", "negate_neg_nh"],  # Non-hate minority
    "identity_bias": ["ident_neutral_nh", "ident_pos_nh"],  # Identity term false positives
    "domain_specificity": ["counter_quote_nh", "counter_ref_nh"],  # Counter speech
    "lexical_coverage": ["spell_space_add_h", "spell_space_del_h", "spell_char_swap_h", "spell_char_del_h", "spell_leet_h"],
}


class HateCheckLoader(DatasetLoader):
    """
    Loader for HateCheck functional test suite.

    HateCheck provides 3,728 test cases across 29 functional tests,
    designed to diagnose specific model capabilities and weaknesses.
    """

    GITHUB_URL = "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv"

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        functional_tests: Optional[List[str]] = None,  # Filter to specific tests
    ):
        super().__init__(cache_dir)
        self.functional_tests = functional_tests
        self.data_path = self.cache_dir / "hatecheck"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_name(self) -> str:
        return "HateCheck"

    def _download_data(self) -> bool:
        """Download HateCheck data if not present."""
        import urllib.request

        csv_path = self.data_path / "test_suite_cases.csv"

        if csv_path.exists():
            return True

        try:
            logger.info("Downloading HateCheck dataset...")
            urllib.request.urlretrieve(self.GITHUB_URL, csv_path)
            logger.info(f"Downloaded to {csv_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download HateCheck: {e}")
            return False

    def load(self) -> List[Sample]:
        """Load HateCheck test cases."""
        logger.info("Loading HateCheck dataset...")

        if not self._download_data():
            self.samples = []
            return self.samples

        csv_path = self.data_path / "test_suite_cases.csv"
        df = pd.read_csv(csv_path)

        logger.info(f"Loaded {len(df)} test cases")
        logger.info(f"Columns: {df.columns.tolist()}")

        samples = []

        for idx, row in df.iterrows():
            func_test = row.get("functionality", "")

            # Filter to specific functional tests if requested
            if self.functional_tests and func_test not in self.functional_tests:
                continue

            # Get label (hateful=1, non-hateful=0)
            label_gold = row.get("label_gold", "")
            label = 1 if label_gold == "hateful" else 0

            sample = Sample(
                text=str(row.get("test_case", "")),
                label=label,
                sample_id=f"hatecheck_{idx}",
                source_dataset="HateCheck",
                content_type="diagnostic",
                metadata={
                    "functionality": func_test,
                    "functionality_name": FUNCTIONAL_TESTS.get(func_test, {}).get("name", ""),
                    "target_ident": row.get("target_ident", ""),
                    "case_templ": row.get("case_templ", ""),
                    "templ_id": row.get("templ_id", ""),
                },
            )
            samples.append(sample)

        self.samples = samples

        # Log distribution by functional test
        func_counts = {}
        for s in samples:
            ft = s.metadata.get("functionality", "unknown")
            func_counts[ft] = func_counts.get(ft, 0) + 1

        logger.info(f"Total samples: {len(samples)}")
        logger.info("Samples per functional test:")
        for ft in sorted(func_counts.keys()):
            hateful = FUNCTIONAL_TESTS.get(ft, {}).get("hateful", None)
            label_str = "hateful" if hateful else "non-hateful" if hateful is not None else "unknown"
            logger.info(f"  {ft}: {func_counts[ft]} ({label_str})")

        return samples

    def get_test_samples(self, test_id: str) -> List[Sample]:
        """Get samples for a specific functional test."""
        if not self.samples:
            self.load()
        return [s for s in self.samples if s.metadata.get("functionality") == test_id]

    def get_data_limitation_samples(self, limitation: str) -> List[Sample]:
        """Get samples for tests targeting a specific data limitation."""
        if limitation not in DATA_LIMITATION_TESTS:
            logger.warning(f"Unknown data limitation: {limitation}")
            return []

        test_ids = DATA_LIMITATION_TESTS[limitation]
        if not self.samples:
            self.load()

        return [s for s in self.samples if s.metadata.get("functionality") in test_ids]

    def get_samples_by_category(self) -> Dict[str, List[Sample]]:
        """Group samples by functional test."""
        if not self.samples:
            self.load()

        by_test = {}
        for s in self.samples:
            ft = s.metadata.get("functionality", "unknown")
            if ft not in by_test:
                by_test[ft] = []
            by_test[ft].append(s)

        return by_test


def get_hatecheck_stats(samples: List[Sample]) -> Dict:
    """Compute statistics for HateCheck samples."""
    stats = {
        "total": len(samples),
        "hateful": sum(1 for s in samples if s.label == 1),
        "non_hateful": sum(1 for s in samples if s.label == 0),
        "by_test": {},
    }

    for s in samples:
        ft = s.metadata.get("functionality", "unknown")
        if ft not in stats["by_test"]:
            stats["by_test"][ft] = {"total": 0, "hateful": 0, "non_hateful": 0}
        stats["by_test"][ft]["total"] += 1
        if s.label == 1:
            stats["by_test"][ft]["hateful"] += 1
        else:
            stats["by_test"][ft]["non_hateful"] += 1

    return stats
