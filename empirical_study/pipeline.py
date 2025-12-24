"""Main evaluation pipeline for the empirical study."""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .data import (
    HatEvalLoader,
    LatentHatredLoader,
    SemEvalSarcasmLoader,
    ISarcasmLoader,
    SapDialectalLoader,
)
from .data.dataset_loader import Sample
from .models import (
    HateBERTDetector,
    RoBERTaHatEvalDetector,
    TwitterRoBERTaIronyDetector,
    GPT4Detector,
)
from .models.base import PredictionResult
from .translation import GoogleTranslator, DeepLTranslator, NLLBTranslator
from .translation.base import TranslationResult
from .evaluation.metrics import (
    evaluate_model,
    compute_preservation_rate,
    mcnemar_test,
    EvaluationResults,
)
from .evaluation.statistical import bonferroni_correction


class EmpiricalStudyPipeline:
    """
    Main pipeline for conducting the empirical study.

    Orchestrates data loading, model inference, translation,
    and evaluation across all conditions.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "./results",
        cache_dir: str = "./data/cache",
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.config = self._load_config(config_path)

        # Store results
        self.results: Dict[str, List[EvaluationResults]] = {}
        self.predictions: Dict[str, Dict[str, List[PredictionResult]]] = {}
        self.translations: Dict[str, List[TranslationResult]] = {}

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        import yaml

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        # Default configuration
        # NOTE: Each count is PER CLASS (positive + negative)
        # So explicit: 100 means 50 hate + 50 non-hate
        return {
            "data": {
                "hate_speech": {"explicit": 100, "implicit": 100},
                "sarcasm": {"clear": 100, "subtle": 100},
                "dialectal": {"aae": 100},
            },
            "translation": {
                "target_languages": ["es", "nl", "it"],
                "systems": ["google", "deepl", "nllb"],
            },
            "evaluation": {
                "bootstrap": {"n_resamples": 1000, "confidence_level": 0.95},
            },
        }

    def _save_samples(
        self,
        samples: List[Sample],
        filename: str,
        predictions: Optional[Dict[str, List[PredictionResult]]] = None,
    ) -> None:
        """
        Save samples to a file for inspection.

        Args:
            samples: List of samples to save
            filename: Name of the output file (without extension)
            predictions: Optional predictions to include
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.samples_dir / f"{filename}_{timestamp}.json"

        # Build prediction lookup
        pred_lookup = {}
        if predictions:
            for model_name, preds in predictions.items():
                for pred in preds:
                    if pred.sample_id not in pred_lookup:
                        pred_lookup[pred.sample_id] = {}
                    pred_lookup[pred.sample_id][model_name] = {
                        "predicted_label": pred.predicted_label,
                        "confidence": pred.confidence,
                    }

        # Build sample records
        records = []
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "true_label": sample.label,
                "source_dataset": sample.source_dataset,
                "content_type": sample.content_type,
                "metadata": sample.metadata,
            }

            # Add predictions if available
            if sample.sample_id in pred_lookup:
                record["predictions"] = pred_lookup[sample.sample_id]

            records.append(record)

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(records)} samples to {filepath}")

        # Also save as CSV for easier viewing
        csv_filepath = self.samples_dir / f"{filename}_{timestamp}.csv"
        df_records = []
        for r in records:
            flat_record = {
                "sample_id": r["sample_id"],
                "text": r["text"][:500],  # Truncate for CSV
                "true_label": r["true_label"],
                "source_dataset": r["source_dataset"],
                "content_type": r["content_type"],
            }
            if "predictions" in r:
                for model, pred in r["predictions"].items():
                    flat_record[f"{model}_pred"] = pred["predicted_label"]
                    flat_record[f"{model}_conf"] = pred["confidence"]
            df_records.append(flat_record)

        pd.DataFrame(df_records).to_csv(csv_filepath, index=False)
        logger.info(f"Saved samples CSV to {csv_filepath}")

    def _balanced_sample(
        self,
        samples: List[Sample],
        n_total: int,
        random_seed: int = 42,
    ) -> List[Sample]:
        """
        Create a balanced sample with equal positive and negative examples.

        Args:
            samples: All available samples
            n_total: Total number of samples to return (will be split 50/50)
            random_seed: Random seed for reproducibility

        Returns:
            Balanced list of samples
        """
        np.random.seed(random_seed)

        # Split by label
        positive = [s for s in samples if s.label == 1]
        negative = [s for s in samples if s.label == 0]

        n_per_class = n_total // 2

        logger.info(f"Available: {len(positive)} positive, {len(negative)} negative")
        logger.info(f"Sampling: {n_per_class} per class")

        # Sample from each class
        if len(positive) >= n_per_class:
            pos_indices = np.random.choice(len(positive), n_per_class, replace=False)
            sampled_pos = [positive[i] for i in pos_indices]
        else:
            logger.warning(f"Only {len(positive)} positive samples available, using all")
            sampled_pos = positive

        if len(negative) >= n_per_class:
            neg_indices = np.random.choice(len(negative), n_per_class, replace=False)
            sampled_neg = [negative[i] for i in neg_indices]
        else:
            logger.warning(f"Only {len(negative)} negative samples available, using all")
            sampled_neg = negative

        # Combine and shuffle
        combined = sampled_pos + sampled_neg
        np.random.shuffle(combined)

        logger.info(f"Final balanced sample: {len(sampled_pos)} pos + {len(sampled_neg)} neg = {len(combined)}")

        return combined

    # ============================================================
    # Data Loading (FIXED: Now includes both classes)
    # ============================================================

    def load_hate_speech_data(self) -> Dict[str, List[Sample]]:
        """
        Load hate speech datasets (HatEval and LatentHatred).

        Returns BALANCED samples with both hate and non-hate examples.
        """
        logger.info("Loading hate speech data...")

        config = self.config["data"]["hate_speech"]

        # Load explicit hate speech from HatEval
        hateval_loader = HatEvalLoader(cache_dir=str(self.cache_dir))
        hateval_samples = hateval_loader.load()

        # FIXED: Get balanced sample (both positive and negative)
        explicit = self._balanced_sample(hateval_samples, config["explicit"])
        for s in explicit:
            s.content_type = "explicit"

        # Load implicit hate speech from LatentHatred
        latent_loader = LatentHatredLoader(cache_dir=str(self.cache_dir))
        latent_samples = latent_loader.load()

        # FIXED: Get balanced sample (both positive and negative)
        implicit = self._balanced_sample(latent_samples, config["implicit"])
        for s in implicit:
            s.content_type = "implicit"

        # Log class distribution
        exp_pos = sum(1 for s in explicit if s.label == 1)
        exp_neg = sum(1 for s in explicit if s.label == 0)
        imp_pos = sum(1 for s in implicit if s.label == 1)
        imp_neg = sum(1 for s in implicit if s.label == 0)

        logger.info(f"Explicit: {exp_pos} hate + {exp_neg} non-hate = {len(explicit)}")
        logger.info(f"Implicit: {imp_pos} hate + {imp_neg} non-hate = {len(implicit)}")

        return {"explicit": explicit, "implicit": implicit}

    def load_sarcasm_data(self) -> Dict[str, List[Sample]]:
        """
        Load sarcasm datasets (SemEval and iSarcasm).

        Returns BALANCED samples with both sarcastic and non-sarcastic examples.
        """
        logger.info("Loading sarcasm data...")

        config = self.config["data"]["sarcasm"]

        # Load clear sarcasm from SemEval
        semeval_loader = SemEvalSarcasmLoader(cache_dir=str(self.cache_dir))
        semeval_samples = semeval_loader.load()

        # FIXED: Get balanced sample
        clear = self._balanced_sample(semeval_samples, config["clear"])
        for s in clear:
            s.content_type = "clear"

        # Load subtle sarcasm from iSarcasm
        isarcasm_loader = ISarcasmLoader(cache_dir=str(self.cache_dir))
        isarcasm_samples = isarcasm_loader.load()

        # FIXED: Get balanced sample
        subtle = self._balanced_sample(isarcasm_samples, config["subtle"])
        for s in subtle:
            s.content_type = "subtle"

        # Log class distribution
        clear_pos = sum(1 for s in clear if s.label == 1)
        clear_neg = sum(1 for s in clear if s.label == 0)
        subtle_pos = sum(1 for s in subtle if s.label == 1)
        subtle_neg = sum(1 for s in subtle if s.label == 0)

        logger.info(f"Clear: {clear_pos} sarcastic + {clear_neg} non-sarcastic = {len(clear)}")
        logger.info(f"Subtle: {subtle_pos} sarcastic + {subtle_neg} non-sarcastic = {len(subtle)}")

        return {"clear": clear, "subtle": subtle}

    def load_dialectal_data(self) -> Dict[str, List[Sample]]:
        """
        Load dialectal data (AAE and SAE).

        Returns BALANCED samples with both hate and non-hate for each dialect.
        """
        logger.info("Loading dialectal data...")

        config = self.config["data"]["dialectal"]

        loader = SapDialectalLoader(cache_dir=str(self.cache_dir))
        all_samples = loader.load()

        aae_all = [s for s in all_samples if s.content_type == "aae"]
        sae_all = [s for s in all_samples if s.content_type == "sae"]

        # FIXED: Get balanced samples for each dialect
        aae = self._balanced_sample(aae_all, config["aae"])
        sae = self._balanced_sample(sae_all, config["aae"])

        # Log class distribution
        aae_pos = sum(1 for s in aae if s.label == 1)
        aae_neg = sum(1 for s in aae if s.label == 0)
        sae_pos = sum(1 for s in sae if s.label == 1)
        sae_neg = sum(1 for s in sae if s.label == 0)

        logger.info(f"AAE: {aae_pos} hate + {aae_neg} non-hate = {len(aae)}")
        logger.info(f"SAE: {sae_pos} hate + {sae_neg} non-hate = {len(sae)}")

        return {"aae": aae, "sae": sae}

    # ============================================================
    # Detection
    # ============================================================

    def run_hate_speech_detection(
        self, samples: List[Sample]
    ) -> Dict[str, List[PredictionResult]]:
        """Run all hate speech detectors on samples."""
        logger.info("Running hate speech detection...")

        texts = [s.text for s in samples]
        sample_ids = [s.sample_id for s in samples]

        results = {}

        # HateBERT
        logger.info("Running HateBERT...")
        hatebert = HateBERTDetector(device=self.device)
        results["hatebert"] = hatebert.predict_batch(texts, sample_ids)

        # RoBERTa-HatEval
        logger.info("Running RoBERTa-HatEval...")
        roberta = RoBERTaHatEvalDetector(device=self.device)
        results["roberta_hateval"] = roberta.predict_batch(texts, sample_ids)

        # GPT-4 (optional, requires API key)
        if os.environ.get("OPENAI_API_KEY"):
            logger.info("Running GPT-4...")
            gpt4 = GPT4Detector(task="hate_speech")
            results["gpt4"] = gpt4.predict_batch(texts, sample_ids)
        else:
            logger.warning("Skipping GPT-4 (OPENAI_API_KEY not set)")

        return results

    def run_sarcasm_detection(
        self, samples: List[Sample]
    ) -> Dict[str, List[PredictionResult]]:
        """Run sarcasm detector on samples."""
        logger.info("Running sarcasm detection...")

        texts = [s.text for s in samples]
        sample_ids = [s.sample_id for s in samples]

        results = {}

        # Twitter-RoBERTa-Irony
        logger.info("Running Twitter-RoBERTa-Irony...")
        twitter_roberta = TwitterRoBERTaIronyDetector(device=self.device)
        results["twitter_roberta_irony"] = twitter_roberta.predict_batch(
            texts, sample_ids
        )

        # GPT-4 (optional)
        if os.environ.get("OPENAI_API_KEY"):
            logger.info("Running GPT-4...")
            gpt4 = GPT4Detector(task="sarcasm")
            results["gpt4"] = gpt4.predict_batch(texts, sample_ids)

        return results

    # ============================================================
    # Translation
    # ============================================================

    def run_translation_experiment(
        self,
        samples: List[Sample],
        target_lang: str = "es",
    ) -> Dict[str, List[TranslationResult]]:
        """Run translation round-trip on samples."""
        logger.info(f"Running translation (EN -> {target_lang} -> EN)...")

        texts = [s.text for s in samples]
        results = {}

        # Google Translate
        try:
            logger.info("Using Google Translate...")
            google = GoogleTranslator()
            results["google"] = google.round_trip_batch(texts, "en", target_lang)
        except Exception as e:
            logger.warning(f"Google Translate failed: {e}")

        # DeepL
        if os.environ.get("DEEPL_API_KEY"):
            try:
                logger.info("Using DeepL...")
                deepl = DeepLTranslator()
                results["deepl"] = deepl.round_trip_batch(texts, "en", target_lang)
            except Exception as e:
                logger.warning(f"DeepL failed: {e}")
        else:
            logger.warning("Skipping DeepL (DEEPL_API_KEY not set)")

        # NLLB-200
        try:
            logger.info("Using NLLB-200...")
            nllb = NLLBTranslator(device=self.device)
            results["nllb"] = nllb.round_trip_batch(texts, "en", target_lang)
        except Exception as e:
            logger.warning(f"NLLB-200 failed: {e}")

        return results

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate_hate_speech_by_content_type(
        self,
        samples: Dict[str, List[Sample]],
        predictions: Dict[str, List[PredictionResult]],
    ) -> pd.DataFrame:
        """
        Evaluate hate speech detection by content type (explicit vs implicit).

        Reproduces Table 1 from the study.
        """
        logger.info("Evaluating hate speech detection by content type...")

        results = []

        for model_name, preds in predictions.items():
            # Map predictions by sample ID
            pred_map = {p.sample_id: p for p in preds}

            for content_type, type_samples in samples.items():
                y_true = []
                y_pred = []

                for sample in type_samples:
                    if sample.sample_id in pred_map:
                        y_true.append(sample.label)
                        y_pred.append(pred_map[sample.sample_id].predicted_label)

                if len(y_true) == 0:
                    continue

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)

                # Log class distribution for debugging
                logger.info(f"{model_name}/{content_type}: "
                           f"true: {sum(y_true==1)} pos, {sum(y_true==0)} neg | "
                           f"pred: {sum(y_pred==1)} pos, {sum(y_pred==0)} neg")

                eval_result = evaluate_model(
                    model_name=model_name,
                    dataset_name="hate_speech",
                    content_type=content_type,
                    y_true=y_true,
                    y_pred=y_pred,
                )

                results.append(eval_result)

        # Create DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])

        return df

    def evaluate_sarcasm_with_translation(
        self,
        samples: List[Sample],
        original_predictions: Dict[str, List[PredictionResult]],
        translations: Dict[str, List[TranslationResult]],
    ) -> pd.DataFrame:
        """
        Evaluate sarcasm detection with translation effects.

        Reproduces Table 2 from the study.
        """
        logger.info("Evaluating sarcasm with translation effects...")

        results = []

        # Get original performance
        for model_name, preds in original_predictions.items():
            y_true = np.array([s.label for s in samples])
            y_pred = np.array([p.predicted_label for p in preds])

            eval_result = evaluate_model(
                model_name=model_name,
                dataset_name="sarcasm",
                content_type="original",
                y_true=y_true,
                y_pred=y_pred,
            )
            results.append(eval_result.to_dict())

        # Evaluate on back-translated text
        for mt_system, trans_results in translations.items():
            # Get back-translated texts
            bt_texts = [t.back_translated_text for t in trans_results]

            # Re-run detection on back-translated text
            detector = TwitterRoBERTaIronyDetector(device=self.device)
            bt_predictions = detector.predict_batch(bt_texts)

            y_true = np.array([s.label for s in samples])
            y_pred_bt = np.array([p.predicted_label for p in bt_predictions])

            # Evaluate
            eval_result = evaluate_model(
                model_name=f"twitter_roberta_{mt_system}",
                dataset_name="sarcasm",
                content_type=f"bt_{mt_system}",
                y_true=y_true,
                y_pred=y_pred_bt,
            )

            # Compute preservation rate
            y_pred_orig = np.array(
                [original_predictions["twitter_roberta_irony"][i].predicted_label
                 for i in range(len(samples))]
            )
            preservation = compute_preservation_rate(y_pred_orig, y_pred_bt)
            eval_result.preservation_rate = preservation

            # McNemar test vs original
            _, p_value = mcnemar_test(y_true, y_pred_orig, y_pred_bt)
            eval_result.p_value = p_value

            results.append(eval_result.to_dict())

        return pd.DataFrame(results)

    def evaluate_dialectal_bias(
        self,
        samples: Dict[str, List[Sample]],
        predictions: Dict[str, List[PredictionResult]],
    ) -> pd.DataFrame:
        """
        Evaluate dialectal bias (AAE vs SAE false positive rates).

        Tests bias as described in Section 7.2.3.
        """
        logger.info("Evaluating dialectal bias...")

        results = []

        for model_name, preds in predictions.items():
            pred_map = {p.sample_id: p for p in preds}

            for dialect, dialect_samples in samples.items():
                y_true = []
                y_pred = []

                for sample in dialect_samples:
                    if sample.sample_id in pred_map:
                        y_true.append(sample.label)
                        y_pred.append(pred_map[sample.sample_id].predicted_label)

                if len(y_true) == 0:
                    continue

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)

                eval_result = evaluate_model(
                    model_name=model_name,
                    dataset_name="dialectal",
                    content_type=dialect,
                    y_true=y_true,
                    y_pred=y_pred,
                )

                results.append({
                    "model": model_name,
                    "dialect": dialect,
                    "fpr": eval_result.false_positive_rate,
                    "fnr": eval_result.false_negative_rate,
                    "f1": eval_result.f1,
                    "macro_f1": eval_result.macro_f1,
                    "n_samples": len(y_true),
                    "n_positive": int(sum(y_true == 1)),
                    "n_negative": int(sum(y_true == 0)),
                })

        df = pd.DataFrame(results)

        # Compute FPR ratio
        if not df.empty:
            pivot = df.pivot(index="model", columns="dialect", values="fpr")
            if "aae" in pivot.columns and "sae" in pivot.columns:
                pivot["fpr_ratio"] = pivot["aae"] / pivot["sae"].replace(0, np.nan)

        return df

    # ============================================================
    # Individual Study Methods (NEW)
    # ============================================================

    def run_study_1_hate_speech_content_type(self) -> pd.DataFrame:
        """
        Run Study 1: Hate Speech Detection by Content Type (Explicit vs Implicit).

        This study compares detection performance on explicit hate speech
        (clear slurs, threats) vs implicit hate speech (no obvious markers).

        Returns:
            DataFrame with results for Table 1
        """
        logger.info("=" * 60)
        logger.info("Study 1: Hate Speech Detection by Content Type")
        logger.info("=" * 60)

        # Load data
        hate_data = self.load_hate_speech_data()
        all_samples = hate_data["explicit"] + hate_data["implicit"]

        # Save samples for inspection
        self._save_samples(all_samples, "study1_hate_speech_samples")

        # Run detection
        predictions = self.run_hate_speech_detection(all_samples)

        # Save samples with predictions
        self._save_samples(all_samples, "study1_hate_speech_with_predictions", predictions)

        # Evaluate
        results = self.evaluate_hate_speech_by_content_type(hate_data, predictions)

        # Save results
        self._save_results({"study1_hate_speech": results})

        return results

    def run_study_2_sarcasm_translation(
        self,
        target_lang: str = "es",
    ) -> pd.DataFrame:
        """
        Run Study 2: Sarcasm Detection with Machine Translation.

        This study measures how sarcasm detection degrades when text
        is translated to another language and back.

        Args:
            target_lang: Target language for translation (es, nl, it)

        Returns:
            DataFrame with results for Table 2
        """
        logger.info("=" * 60)
        logger.info(f"Study 2: Sarcasm Detection with Translation ({target_lang})")
        logger.info("=" * 60)

        # Load data
        sarcasm_data = self.load_sarcasm_data()
        all_samples = sarcasm_data["clear"] + sarcasm_data["subtle"]

        # Save samples
        self._save_samples(all_samples, f"study2_sarcasm_samples_{target_lang}")

        # Run original detection
        original_predictions = self.run_sarcasm_detection(all_samples)

        # Run translation
        translations = self.run_translation_experiment(all_samples, target_lang)

        # Save samples with predictions
        self._save_samples(
            all_samples,
            f"study2_sarcasm_with_predictions_{target_lang}",
            original_predictions
        )

        # Evaluate
        results = self.evaluate_sarcasm_with_translation(
            all_samples, original_predictions, translations
        )

        # Save results
        self._save_results({f"study2_sarcasm_mt_{target_lang}": results})

        return results

    def run_study_3_dialectal_bias(self) -> pd.DataFrame:
        """
        Run Study 3: Dialectal Bias (AAE vs SAE).

        This study measures whether hate speech detectors show
        higher false positive rates on African American English
        compared to Standard American English.

        Returns:
            DataFrame with results for Table 3
        """
        logger.info("=" * 60)
        logger.info("Study 3: Dialectal Bias (AAE vs SAE)")
        logger.info("=" * 60)

        # Load data
        dialectal_data = self.load_dialectal_data()
        all_samples = dialectal_data["aae"] + dialectal_data["sae"]

        # Save samples
        self._save_samples(all_samples, "study3_dialectal_samples")

        # Run detection
        predictions = self.run_hate_speech_detection(all_samples)

        # Save samples with predictions
        self._save_samples(all_samples, "study3_dialectal_with_predictions", predictions)

        # Evaluate
        results = self.evaluate_dialectal_bias(dialectal_data, predictions)

        # Save results
        self._save_results({"study3_dialectal_bias": results})

        return results

    # ============================================================
    # Full Pipeline
    # ============================================================

    def run_full_study(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete empirical study (all 3 studies).

        Returns dictionary of result DataFrames.
        """
        logger.info("=" * 60)
        logger.info("Starting Complete Empirical Study Pipeline")
        logger.info("=" * 60)

        all_results = {}

        # Study 1: Hate Speech by Content Type
        all_results["study1_hate_speech"] = self.run_study_1_hate_speech_content_type()

        # Study 2: Sarcasm with Translation (for each target language)
        for target_lang in self.config["translation"]["target_languages"]:
            key = f"study2_sarcasm_mt_{target_lang}"
            all_results[key] = self.run_study_2_sarcasm_translation(target_lang)

        # Study 3: Dialectal Bias
        all_results["study3_dialectal_bias"] = self.run_study_3_dialectal_bias()

        logger.info("\n" + "=" * 60)
        logger.info("Complete Empirical Study Finished")
        logger.info("=" * 60)

        return all_results

    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, df in results.items():
            if df is not None and not df.empty:
                # Save CSV
                csv_path = self.output_dir / f"{name}_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {csv_path}")

                # Save JSON
                json_path = self.output_dir / f"{name}_{timestamp}.json"
                df.to_json(json_path, orient="records", indent=2)

        # Save summary
        summary = {
            "timestamp": timestamp,
            "config": self.config,
            "tables": list(results.keys()),
        }
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def main():
    """Main entry point for the empirical study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Empirical Study on Hate Speech and Sarcasm Detection"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./data/cache", help="Cache directory"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--study", type=str, default="all",
        choices=["all", "1", "2", "3", "hate_speech", "sarcasm", "dialectal"],
        help="Which study to run (1=hate_speech, 2=sarcasm, 3=dialectal, all=everything)"
    )
    parser.add_argument(
        "--target-lang", type=str, default="es",
        choices=["es", "nl", "it"],
        help="Target language for Study 2 (sarcasm translation)"
    )
    args = parser.parse_args()

    # Configure logging
    logger.add(
        "empirical_study_{time}.log",
        rotation="100 MB",
        level="INFO",
    )

    # Run pipeline
    pipeline = EmpiricalStudyPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    # Run selected study
    if args.study in ["all"]:
        results = pipeline.run_full_study()
    elif args.study in ["1", "hate_speech"]:
        results = {"study1": pipeline.run_study_1_hate_speech_content_type()}
    elif args.study in ["2", "sarcasm"]:
        results = {"study2": pipeline.run_study_2_sarcasm_translation(args.target_lang)}
    elif args.study in ["3", "dialectal"]:
        results = {"study3": pipeline.run_study_3_dialectal_bias()}
    else:
        results = pipeline.run_full_study()

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for name, df in results.items():
        if df is not None and not df.empty:
            print(f"\n{name}:")
            print(df.to_string())


if __name__ == "__main__":
    main()
