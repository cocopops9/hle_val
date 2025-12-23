#!/usr/bin/env python
"""
Main entry point for running the empirical study.

Usage:
    python scripts/run_study.py --config configs/config.yaml --output-dir results/
    python scripts/run_study.py --quick  # Run with reduced samples for testing
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from empirical_study.pipeline import EmpiricalStudyPipeline
from empirical_study.analysis.visualizations import create_all_figures, create_latex_tables


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


def main():
    parser = argparse.ArgumentParser(
        description="Run Empirical Study on Hate Speech and Sarcasm Detection"
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
        "--quick",
        action="store_true",
        help="Run with reduced samples for quick testing",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation experiments",
    )
    parser.add_argument(
        "--skip-gpt4",
        action="store_true",
        help="Skip GPT-4 evaluation even if API key is available",
    )
    parser.add_argument(
        "--create-figures",
        action="store_true",
        help="Generate visualization figures",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Generate LaTeX tables",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_dir)
    logger.info("=" * 60)
    logger.info("Empirical Study on Hate Speech and Sarcasm Detection")
    logger.info("=" * 60)

    # Check environment
    check_environment()

    # Handle quick mode
    if args.quick:
        logger.info("Running in quick mode with reduced samples")
        # Override config for quick testing
        os.environ["EMPIRICAL_STUDY_QUICK_MODE"] = "1"

    # Handle skip flags
    if args.skip_gpt4:
        # Temporarily unset API key to skip GPT-4
        os.environ.pop("OPENAI_API_KEY", None)

    # Create and run pipeline
    pipeline = EmpiricalStudyPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    results = pipeline.run_full_study()

    # Create figures if requested
    if args.create_figures:
        figures_dir = Path(args.output_dir) / "figures"
        create_all_figures(results, str(figures_dir))

    # Create tables if requested
    if args.create_tables:
        tables_dir = Path(args.output_dir) / "tables"
        create_latex_tables(results, str(tables_dir))

    # Print summary
    print("\n" + "=" * 60)
    print("Study Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}")

    for name, df in results.items():
        if df is not None and not df.empty:
            print(f"\n{name}:")
            print(df.head().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
