#!/usr/bin/env python
"""
Main entry point for running the empirical study.

Usage:
    # Run all studies
    python scripts/run_study.py --output-dir results/

    # Run individual studies
    python scripts/run_study.py --study 1  # Hate speech: explicit vs implicit
    python scripts/run_study.py --study 2  # Sarcasm with translation
    python scripts/run_study.py --study 3  # Dialectal bias (AAE vs SAE)

    # Run Study 2 with different target language
    python scripts/run_study.py --study 2 --target-lang nl

    # Quick test with reduced samples
    python scripts/run_study.py --quick
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def setup_logging(log_dir: str = "./logs"):
    """Configure logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.add(
        f"{log_dir}/empirical_study_{{time}}.log",
        rotation="100 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def check_environment():
    """Check required environment variables and dependencies."""
    warnings = []

    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set - GPT-4 evaluation will be skipped")

    if not os.environ.get("DEEPL_API_KEY"):
        warnings.append("DEEPL_API_KEY not set - DeepL translation will be skipped")

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available - using CPU (will be slower)")
    except ImportError:
        warnings.append("PyTorch not installed")

    for warning in warnings:
        logger.warning(warning)

    return len(warnings) == 0


def print_study_info():
    """Print information about available studies."""
    info = """
╔══════════════════════════════════════════════════════════════════╗
║                    EMPIRICAL STUDY OPTIONS                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Study 1: Hate Speech Detection by Content Type                 ║
║  --study 1 or --study hate_speech                                ║
║  Compares explicit (clear slurs) vs implicit (subtle) hate       ║
║                                                                  ║
║  Study 2: Sarcasm Detection with Translation                    ║
║  --study 2 or --study sarcasm                                    ║
║  Measures sarcasm preservation through machine translation       ║
║  Use --target-lang to specify: es (Spanish), nl (Dutch), it     ║
║                                                                  ║
║  Study 3: Dialectal Bias                                        ║
║  --study 3 or --study dialectal                                  ║
║  Measures FPR differences between AAE and SAE                    ║
║                                                                  ║
║  Run All: --study all (default)                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(info)


def main():
    parser = argparse.ArgumentParser(
        description="Run Empirical Study on Hate Speech and Sarcasm Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_study.py --study 1              # Run hate speech study
  python scripts/run_study.py --study 2 --target-lang es  # Sarcasm with Spanish
  python scripts/run_study.py --study 3              # Run dialectal bias study
  python scripts/run_study.py --study all            # Run all studies
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="empirical_study/configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/cache",
        help="Cache directory for datasets and models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "hate_speech", "sarcasm", "dialectal"],
        help="Which study to run",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="es",
        choices=["es", "nl", "it"],
        help="Target language for Study 2 (sarcasm translation)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced samples for quick testing",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about available studies",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files",
    )

    args = parser.parse_args()

    # Print info and exit if requested
    if args.info:
        print_study_info()
        return 0

    # Setup
    setup_logging(args.log_dir)

    print_study_info()

    logger.info("=" * 60)
    logger.info("Empirical Study on Hate Speech and Sarcasm Detection")
    logger.info("=" * 60)

    # Check environment
    check_environment()

    # Import here to avoid slow startup when just checking --help
    from empirical_study.pipeline import EmpiricalStudyPipeline

    # Handle quick mode - modify config
    config_override = None
    if args.quick:
        logger.info("Running in QUICK MODE with reduced samples")
        config_override = {
            "data": {
                "hate_speech": {"explicit": 20, "implicit": 20},
                "sarcasm": {"clear": 20, "subtle": 20},
                "dialectal": {"aae": 20},
            },
            "translation": {
                "target_languages": [args.target_lang],
                "systems": ["nllb"],  # Only use local model for quick test
            },
            "evaluation": {
                "bootstrap": {"n_resamples": 100, "confidence_level": 0.95},
            },
        }

    # Create pipeline
    pipeline = EmpiricalStudyPipeline(
        config_path=args.config if not args.quick else None,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    # Override config if quick mode
    if config_override:
        pipeline.config = config_override

    # Run selected study
    logger.info(f"Running study: {args.study}")

    if args.study == "all":
        results = pipeline.run_full_study()
    elif args.study in ["1", "hate_speech"]:
        results = {"study1_hate_speech": pipeline.run_study_1_hate_speech_content_type()}
    elif args.study in ["2", "sarcasm"]:
        results = {f"study2_sarcasm_{args.target_lang}": pipeline.run_study_2_sarcasm_translation(args.target_lang)}
    elif args.study in ["3", "dialectal"]:
        results = {"study3_dialectal": pipeline.run_study_3_dialectal_bias()}
    else:
        results = pipeline.run_full_study()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for name, df in results.items():
        if df is not None and not df.empty:
            print(f"\n{name}:")
            print("-" * 40)
            print(df.to_string())

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Sample files saved to: {args.output_dir}/samples/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
