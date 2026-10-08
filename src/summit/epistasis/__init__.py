"""Quantitative epistasis: study-matched summaries and explicit small-data oracles."""

__all__ = [
    "EpistasisSummary",
    "fit_epistasis",
    "load_summary",
    "write_summary",
    "LinearScoreSummary",
    "prepare_linear_scores",
    "linear_score_tests",
    "load_linear_scores",
    "write_linear_scores",
    "prespecified_followup",
    "RobustScoreSummary",
    "prepare_robust_scores",
    "robust_score_tests",
    "load_robust_scores",
    "write_robust_scores",
    "robust_followup",
]


def __getattr__(name):
    if name in __all__[10:]:
        from . import robust

        return getattr(robust, name)
    if name in __all__[4:10]:
        from . import score

        return getattr(score, name)
    if name in __all__[:4]:
        from . import summary

        return getattr(summary, name)
    raise AttributeError(name)
