"""Visualization functions for empirical study results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
    })


def plot_content_type_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Hate Speech Detection F1 by Content Type",
) -> plt.Figure:
    """
    Plot comparison of detection performance by content type.

    Creates a grouped bar chart comparing explicit vs implicit hate speech
    detection across different models.

    Args:
        results_df: DataFrame with columns [model_name, content_type, macro_f1]
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    setup_plotting_style()

    # Pivot for plotting
    pivot = results_df.pivot(
        index="model_name", columns="content_type", values="macro_f1"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar positions
    x = np.arange(len(pivot.index))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        pivot.get("explicit", [0] * len(x)),
        width,
        label="Explicit",
        color="#2ecc71",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        pivot.get("implicit", [0] * len(x)),
        width,
        label="Implicit",
        color="#e74c3c",
        edgecolor="black",
    )

    # Customize
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Add delta annotations
    if "explicit" in pivot.columns and "implicit" in pivot.columns:
        for i, model in enumerate(pivot.index):
            delta = pivot.loc[model, "explicit"] - pivot.loc[model, "implicit"]
            ax.annotate(
                f"Δ={delta:.2f}",
                xy=(i, max(pivot.loc[model]) + 0.05),
                ha="center",
                fontsize=9,
                color="gray",
            )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_mt_preservation(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Sarcasm Detection F1 and Preservation Rate by MT System",
) -> plt.Figure:
    """
    Plot sarcasm detection performance and preservation rates across MT systems.

    Creates a dual-axis bar chart showing F1 and preservation rate.

    Args:
        results_df: DataFrame with columns [condition, f1, preservation_rate]
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    setup_plotting_style()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Filter to MT conditions
    mt_df = results_df[results_df["content_type"].str.startswith("bt_")].copy()
    if mt_df.empty:
        logger.warning("No MT results to plot")
        return fig

    mt_df["mt_system"] = mt_df["content_type"].str.replace("bt_", "")

    x = np.arange(len(mt_df))
    width = 0.4

    # F1 bars
    bars1 = ax1.bar(
        x - width / 2,
        mt_df["macro_f1"],
        width,
        label="F1 Score",
        color="#3498db",
        edgecolor="black",
    )
    ax1.set_xlabel("MT System")
    ax1.set_ylabel("Macro F1 Score", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")
    ax1.set_ylim(0, 1)

    # Preservation rate on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        mt_df["preservation_rate"] if "preservation_rate" in mt_df.columns else [0] * len(x),
        width,
        label="Preservation Rate",
        color="#e67e22",
        edgecolor="black",
    )
    ax2.set_ylabel("Preservation Rate", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax2.set_ylim(0, 1)

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(mt_df["mt_system"].str.upper())

    # Title and legend
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_dialectal_bias(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "False Positive Rates by Dialect",
) -> plt.Figure:
    """
    Plot dialectal bias comparing AAE vs SAE false positive rates.

    Args:
        results_df: DataFrame with columns [model, dialect, fpr]
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Pivot for plotting
    pivot = results_df.pivot(index="model", columns="dialect", values="fpr")

    x = np.arange(len(pivot.index))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        pivot.get("aae", [0] * len(x)) * 100,  # Convert to percentage
        width,
        label="African American English",
        color="#9b59b6",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        pivot.get("sae", [0] * len(x)) * 100,
        width,
        label="Standard American English",
        color="#1abc9c",
        edgecolor="black",
    )

    # Customize
    ax.set_xlabel("Model")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend()

    # Add ratio annotations
    if "aae" in pivot.columns and "sae" in pivot.columns:
        for i, model in enumerate(pivot.index):
            if pivot.loc[model, "sae"] > 0:
                ratio = pivot.loc[model, "aae"] / pivot.loc[model, "sae"]
                max_fpr = max(pivot.loc[model, "aae"], pivot.loc[model, "sae"]) * 100
                ax.annotate(
                    f"{ratio:.1f}×",
                    xy=(i, max_fpr + 2),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#e74c3c",
                )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_cross_platform_transfer(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Cross-Platform Transfer Performance",
) -> plt.Figure:
    """
    Plot cross-platform transfer degradation.

    Args:
        results_df: DataFrame with platform transfer results
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Example structure - would need actual cross-platform data
    platforms = ["Twitter→Twitter", "Twitter→Reddit", "Reddit→Reddit", "Reddit→Twitter"]
    f1_scores = [0.82, 0.66, 0.79, 0.63]

    colors = ["#2ecc71" if "→" in p and p.split("→")[0] == p.split("→")[1] else "#e74c3c" for p in platforms]

    bars = ax.bar(platforms, f1_scores, color=colors, edgecolor="black")

    ax.set_xlabel("Train → Test Platform")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_preservation_by_language(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Sarcasm Preservation by Target Language",
) -> plt.Figure:
    """
    Plot sarcasm preservation rates across different target languages.

    Args:
        results_df: DataFrame with language-specific preservation rates
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Example data structure
    languages = ["Spanish (ES)", "Dutch (NL)", "Italian (IT)"]
    preservation = [0.63, 0.59, 0.61]
    f1_scores = [0.61, 0.57, 0.59]

    x = np.arange(len(languages))
    width = 0.35

    bars1 = ax.bar(x - width / 2, preservation, width, label="Preservation Rate", color="#3498db")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score", color="#e67e22")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def create_latex_tables(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "./tables",
) -> Dict[str, str]:
    """
    Create LaTeX tables from results DataFrames.

    Args:
        results: Dictionary of result DataFrames
        output_dir: Directory to save LaTeX files

    Returns:
        Dictionary mapping table names to LaTeX strings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    latex_tables = {}

    # Table 1: Hate Speech Detection by Content Type
    if "hate_speech" in results:
        df = results["hate_speech"]
        pivot = df.pivot(index="model_name", columns="content_type", values="macro_f1")
        if "explicit" in pivot.columns and "implicit" in pivot.columns:
            pivot["delta"] = pivot["explicit"] - pivot["implicit"]
            pivot["overall"] = (pivot["explicit"] + pivot["implicit"]) / 2

        latex = pivot.to_latex(
            float_format="%.2f",
            caption="Hate Speech Detection F1 by Content Type",
            label="tab:hate_speech",
        )
        latex_tables["table1_hate_speech"] = latex

        with open(output_path / "table1_hate_speech.tex", "w") as f:
            f.write(latex)

    # Table 2: Sarcasm MT Preservation
    for key in results:
        if key.startswith("sarcasm_mt"):
            df = results[key]
            latex = df.to_latex(
                float_format="%.2f",
                caption=f"Sarcasm Detection with Translation ({key})",
                label=f"tab:{key}",
            )
            latex_tables[key] = latex

            with open(output_path / f"{key}.tex", "w") as f:
                f.write(latex)

    # Table 3: Dialectal Bias
    if "dialectal_bias" in results:
        df = results["dialectal_bias"]
        pivot = df.pivot(index="model", columns="dialect", values="fpr")
        if "aae" in pivot.columns and "sae" in pivot.columns:
            pivot["ratio"] = pivot["aae"] / pivot["sae"].replace(0, np.nan)

        latex = pivot.to_latex(
            float_format="%.2f",
            caption="False Positive Rates by Dialect",
            label="tab:dialectal",
        )
        latex_tables["table3_dialectal"] = latex

        with open(output_path / "table3_dialectal.tex", "w") as f:
            f.write(latex)

    logger.info(f"Created {len(latex_tables)} LaTeX tables in {output_dir}")

    return latex_tables


def create_all_figures(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "./figures",
) -> None:
    """
    Create all figures for the empirical study.

    Args:
        results: Dictionary of result DataFrames
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Figure 1: Hate Speech by Content Type
    if "hate_speech" in results:
        plot_content_type_comparison(
            results["hate_speech"],
            output_path=str(output_path / "fig1_hate_speech_content_type.png"),
        )

    # Figure 2: MT Preservation
    for key in results:
        if key.startswith("sarcasm_mt"):
            plot_mt_preservation(
                results[key],
                output_path=str(output_path / f"fig2_{key}_preservation.png"),
            )

    # Figure 3: Dialectal Bias
    if "dialectal_bias" in results:
        plot_dialectal_bias(
            results["dialectal_bias"],
            output_path=str(output_path / "fig3_dialectal_bias.png"),
        )

    logger.info(f"Created figures in {output_dir}")
