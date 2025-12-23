# Empirical Study: Hate Speech and Sarcasm Detection

A comprehensive empirical study examining detection performance across content types and translation conditions, focusing on:

1. **Hate Speech Detection** - Comparing explicit vs. implicit hate speech detection
2. **Sarcasm Detection** - Evaluating performance degradation through machine translation
3. **Dialectal Bias** - Measuring false positive rate disparities across dialects
4. **Cross-Platform Transfer** - Assessing generalization across social media platforms

## Overview

This study implements the methodology described in Section 7 of the research paper, providing:

- Data loading from established benchmarks (HatEval, LatentHatred, SemEval-2018 Task 3, iSarcasm, Sap et al.)
- Multiple detection systems (HateBERT, RoBERTa-HatEval, Twitter-RoBERTa-Irony, GPT-4)
- Translation through Google Translate, DeepL, and NLLB-200
- Comprehensive evaluation metrics with bootstrap confidence intervals
- Statistical significance testing (McNemar's test with Bonferroni correction)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hle_val

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Configuration

### Environment Variables

Set the following environment variables for full functionality:

```bash
# For GPT-4 evaluation
export OPENAI_API_KEY="your-openai-api-key"

# For DeepL translation
export DEEPL_API_KEY="your-deepl-api-key"

# Optional: Specify GPU device
export CUDA_VISIBLE_DEVICES=0
```

### Configuration File

Edit `empirical_study/configs/config.yaml` to customize:

```yaml
data:
  hate_speech:
    explicit: 100  # Number of explicit samples
    implicit: 100  # Number of implicit samples
  sarcasm:
    clear: 100     # Clear polarity contrast cases
    subtle: 100    # Subtle/contextual cases
  dialectal:
    aae: 100       # African American English samples

translation:
  target_languages:
    - es  # Spanish
    - nl  # Dutch
    - it  # Italian
  systems:
    - google
    - deepl
    - nllb

evaluation:
  bootstrap:
    n_resamples: 1000
    confidence_level: 0.95
```

## Usage

### Running the Full Study

```bash
# Run the complete empirical study
python scripts/run_study.py --output-dir results/

# Quick test with reduced samples
python scripts/run_study.py --quick

# Skip translation experiments (faster)
python scripts/run_study.py --skip-translation

# Generate figures and tables
python scripts/run_study.py --create-figures --create-tables
```

### Programmatic Usage

```python
from empirical_study.pipeline import EmpiricalStudyPipeline

# Initialize pipeline
pipeline = EmpiricalStudyPipeline(
    config_path="empirical_study/configs/config.yaml",
    output_dir="./results",
    device="cuda",  # or "cpu"
)

# Run full study
results = pipeline.run_full_study()

# Access specific results
hate_speech_results = results["hate_speech"]
sarcasm_mt_results = results["sarcasm_mt_es"]
dialectal_results = results["dialectal_bias"]
```

### Individual Components

```python
# Load specific datasets
from empirical_study.data import HatEvalLoader, LatentHatredLoader

loader = HatEvalLoader(cache_dir="./data/cache")
samples = loader.load()

# Run specific detector
from empirical_study.models import HateBERTDetector

detector = HateBERTDetector(device="cuda")
predictions = detector.predict_batch([s.text for s in samples])

# Translate text
from empirical_study.translation import DeepLTranslator

translator = DeepLTranslator()
results = translator.round_trip_batch(texts, source="en", target="es")

# Evaluate
from empirical_study.evaluation import evaluate_model, mcnemar_test

eval_results = evaluate_model(
    model_name="hatebert",
    dataset_name="hateval",
    content_type="explicit",
    y_true=labels,
    y_pred=predictions,
)
```

## Project Structure

```
hle_val/
├── empirical_study/
│   ├── __init__.py
│   ├── pipeline.py              # Main orchestration
│   ├── configs/
│   │   └── config.yaml          # Configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py    # Base loader
│   │   ├── hate_speech.py       # HatEval, LatentHatred
│   │   ├── sarcasm.py           # SemEval, iSarcasm
│   │   └── dialectal.py         # Sap et al.
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Base detector
│   │   ├── hate_speech.py       # HateBERT, RoBERTa
│   │   ├── sarcasm.py           # Twitter-RoBERTa-Irony
│   │   └── gpt4.py              # GPT-4 zero-shot
│   ├── translation/
│   │   ├── __init__.py
│   │   ├── base.py              # Base translator
│   │   ├── google_translate.py
│   │   ├── deepl_translate.py
│   │   └── nllb_translate.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # F1, bootstrap CI
│   │   └── statistical.py       # McNemar, corrections
│   └── analysis/
│       ├── __init__.py
│       ├── visualizations.py    # Charts, tables
│       └── error_analysis.py    # Error patterns
├── scripts/
│   └── run_study.py             # CLI entry point
├── tests/
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Expected Results

The study reproduces the following tables from Section 7:

### Table 1: Hate Speech Detection F1 by Content Type

| Model | Explicit | Implicit | Δ | Overall |
|-------|----------|----------|---|---------|
| HateBERT | 0.89 | 0.61 | 0.28 | 0.75 |
| RoBERTa-HatEval | 0.87 | 0.58 | 0.29 | 0.73 |
| GPT-4 (zero-shot) | 0.86 | 0.62 | 0.24 | 0.74 |

### Table 2: Sarcasm Detection with MT (EN→ES→EN)

| Condition | F1 | Δ | Preservation | p-value |
|-----------|-----|---|--------------|---------|
| Original | 0.74 | — | — | — |
| Google | 0.58 | -0.16 | 56% | <0.001 |
| DeepL | 0.61 | -0.13 | 63% | <0.001 |
| NLLB-200 | 0.51 | -0.23 | 48% | <0.001 |

### Table 3: Dialectal Bias (False Positive Rates)

| Model | AAE FPR | SAE FPR | Ratio |
|-------|---------|---------|-------|
| HateBERT | 34% | 16% | 2.1× |
| RoBERTa | 31% | 17% | 1.8× |
| GPT-4 | 24% | 18% | 1.3× |

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=empirical_study --cov-report=html
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@article{empirical_study_2024,
  title={Empirical Study on Hate Speech and Sarcasm Detection
         Across Content Types and Translation Conditions},
  author={...},
  journal={...},
  year={2024}
}
```
