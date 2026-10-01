"""Plot ld limits for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ld_axis_limits(sub: pd.DataFrame, case_label: str) -> tuple[float, float]:
    """Keep the small negative EUR B=1,024 differences visible."""
    values = pd.to_numeric(sub["deficit_pct"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-2.0, 10.0)
    lower_default = -3.0 if case_label == "No covariates" else -2.0
    lower = min(lower_default, float(np.floor(values.min() - 0.25)))
    upper_value = float(values.max())
    if case_label == "No covariates":
        upper = max(105.0, float(np.ceil((upper_value + 3.0) / 10.0) * 10.0))
    else:
        upper = max(16.0, float(np.ceil((upper_value + 1.2) / 2.0) * 2.0))
    return lower, upper
