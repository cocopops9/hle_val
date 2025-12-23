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

    # ============================================================
    # Data Loading
    # ============================================================

    def load_hate_speech_data(self) -> Dict[str, List[Sample]]:
        """Load hate speech datasets (HatEval and LatentHatred)."""
        logger.info("Loading hate speech data...")

        config = self.config["data"]["hate_speech"]

        # Load explicit hate speech from HatEval
        hateval_loader = HatEvalLoader(cache_dir=str(self.cache_dir))
        hateval_samples = hateval_loader.load()
        explicit = hateval_loader.stratified_sample(
            [s for s in hateval_samples if s.label == 1],
            config["explicit"],
        )

        # Load implicit hate speech from LatentHatred
        latent_loader = LatentHatredLoader(cache_dir=str(self.cache_dir))
        latent_samples = latent_loader.load()
        implicit = latent_loader.stratified_sample(
            [s for s in latent_samples if s.label == 1],
            config["implicit"],
        )

        logger.info(
            f"Loaded {len(explicit)} explicit, {len(implicit)} implicit samples"
        )

        return {"explicit": explicit, "implicit": implicit}

    def load_sarcasm_data(self) -> Dict[str, List[Sample]]:
        """Load sarcasm datasets (SemEval and iSarcasm)."""
        logger.info("Loading sarcasm data...")

        config = self.config["data"]["sarcasm"]

        # Load clear sarcasm from SemEval
        semeval_loader = SemEvalSarcasmLoader(cache_dir=str(self.cache_dir))
        semeval_samples = semeval_loader.load()
        clear = semeval_loader.stratified_sample(semeval_samples, config["clear"])

        # Load subtle sarcasm from iSarcasm
        isarcasm_loader = ISarcasmLoader(cache_dir=str(self.cache_dir))
        isarcasm_samples = isarcasm_loader.load()
        subtle = isarcasm_loader.stratified_sample(isarcasm_samples, config["subtle"])

        logger.info(f"Loaded {len(clear)} clear, {len(subtle)} subtle samples")

        return {"clear": clear, "subtle": subtle}

    def load_dialectal_data(self) -> Dict[str, List[Sample]]:
        """Load dialectal data (AAE and SAE)."""
        logger.info("Loading dialectal data...")

        config = self.config["data"]["dialectal"]

        loader = SapDialectalLoader(cache_dir=str(self.cache_dir))
        all_samples = loader.load()

        aae = [s for s in all_samples if s.content_type == "aae"]
        sae = [s for s in all_samples if s.content_type == "sae"]

        aae = loader.stratified_sample(aae, config["aae"])
        sae = loader.stratified_sample(sae, config["aae"])

        logger.info(f"Loaded {len(aae)} AAE, {len(sae)} SAE samples")

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

                eval_result = evaluate_model(
                    model_name=model_name,
                    dataset_name="hate_speech",
                    content_type=content_type,
                    y_true=np.array(y_true),
                    y_pred=np.array(y_pred),
                )

                results.append(eval_result)

        # Create DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])

        # Pivot for Table 1 format
        if not df.empty:
            pivot = df.pivot(
                index="model_name", columns="content_type", values="macro_f1"
            )
            pivot["delta"] = pivot.get("explicit", 0) - pivot.get("implicit", 0)
            pivot["overall"] = (pivot.get("explicit", 0) + pivot.get("implicit", 0)) / 2

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
                    "n_samples": len(y_true),
                })

        df = pd.DataFrame(results)

        # Compute FPR ratio
        if not df.empty:
            pivot = df.pivot(index="model", columns="dialect", values="fpr")
            if "aae" in pivot.columns and "sae" in pivot.columns:
                pivot["fpr_ratio"] = pivot["aae"] / pivot["sae"].replace(0, np.nan)

        return df

    # ============================================================
    # Full Pipeline
    # ============================================================

    def run_full_study(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete empirical study.

        Returns dictionary of result DataFrames.
        """
        logger.info("=" * 60)
        logger.info("Starting Empirical Study Pipeline")
        logger.info("=" * 60)

        all_results = {}

        # 1. Hate Speech Detection by Content Type
        logger.info("\n--- Hate Speech Detection ---")
        hate_data = self.load_hate_speech_data()
        all_hate_samples = hate_data["explicit"] + hate_data["implicit"]
        hate_predictions = self.run_hate_speech_detection(all_hate_samples)
        all_results["hate_speech"] = self.evaluate_hate_speech_by_content_type(
            hate_data, hate_predictions
        )

        # 2. Sarcasm Detection with Translation
        logger.info("\n--- Sarcasm Detection with Translation ---")
        sarcasm_data = self.load_sarcasm_data()
        all_sarcasm_samples = sarcasm_data["clear"] + sarcasm_data["subtle"]
        sarcasm_predictions = self.run_sarcasm_detection(all_sarcasm_samples)

        # Run translation for each target language
        for target_lang in self.config["translation"]["target_languages"]:
            translations = self.run_translation_experiment(
                all_sarcasm_samples, target_lang
            )
            key = f"sarcasm_mt_{target_lang}"
            all_results[key] = self.evaluate_sarcasm_with_translation(
                all_sarcasm_samples, sarcasm_predictions, translations
            )

        # 3. Dialectal Bias
        logger.info("\n--- Dialectal Bias ---")
        dialectal_data = self.load_dialectal_data()
        all_dialectal_samples = dialectal_data["aae"] + dialectal_data["sae"]
        dialectal_predictions = self.run_hate_speech_detection(all_dialectal_samples)
        all_results["dialectal_bias"] = self.evaluate_dialectal_bias(
            dialectal_data, dialectal_predictions
        )

        # Save results
        self._save_results(all_results)

        logger.info("\n" + "=" * 60)
        logger.info("Empirical Study Complete")
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
