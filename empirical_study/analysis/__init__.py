"""Analysis and visualization modules for empirical study."""

from .visualizations import (
    plot_content_type_comparison,
    plot_mt_preservation,
    plot_dialectal_bias,
    plot_cross_platform_transfer,
    create_latex_tables,
)
from .error_analysis import (
    analyze_implicit_hate_errors,
    analyze_sarcasm_errors,
    analyze_mt_induced_errors,
)

__all__ = [
    "plot_content_type_comparison",
    "plot_mt_preservation",
    "plot_dialectal_bias",
    "plot_cross_platform_transfer",
    "create_latex_tables",
    "analyze_implicit_hate_errors",
    "analyze_sarcasm_errors",
    "analyze_mt_induced_errors",
]
