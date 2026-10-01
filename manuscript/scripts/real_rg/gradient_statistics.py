"""Plot gradient statistics for the SUMMIT manuscript."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


BIN_RE = re.compile(r"^m(?P<lo>\d+)_(?P<hi>\d+)_ld(?P<ld>\d+)$")


@dataclass(frozen=True)
class BinInfo:
    bin_index: int
    bin_name: str
    maf_lo: float
    maf_hi: float
    ld_index: int
    ld_label: str
    maf_label: str


def canonical_pair(a: object, b: object) -> Tuple[str, str]:
    x = str(a)
    y = str(b)
    return (x, y) if x <= y else (y, x)


def infer_K_from_columns(df: pd.DataFrame) -> int:
    if "n_bins" in df.columns and df["n_bins"].notnull().any():
        try:
            return int(np.nanmax(pd.to_numeric(df["n_bins"], errors="coerce").values))
        except Exception:
            pass
    k = 0
    while f"bin_name_{k}" in df.columns:
        k += 1
    if k > 0:
        return k
    k = 0
    while f"rg_bin_{k}" in df.columns:
        k += 1
    return k


def infer_bin_names(df: pd.DataFrame, K: int) -> List[str]:
    for _, row in df.iterrows():
        vals: List[str] = []
        ok = False
        for k in range(K):
            raw = row.get(f"bin_name_{k}", "")
            val = "" if pd.isna(raw) else str(raw).strip()
            if val:
                ok = True
            vals.append(val or f"bin_{k}")
        if ok:
            return vals
    return [f"bin_{k}" for k in range(K)]


def parse_bin_name(bin_index: int, bin_name: str) -> BinInfo:
    m = BIN_RE.match(str(bin_name).strip())
    if m is None:
        raise ValueError(
            f"Could not parse MAFLD bin name {bin_name!r}. "
            "Expected format like 'm010_012_ld3'."
        )
    lo = float(m.group("lo")) / 1000.0
    hi = float(m.group("hi")) / 1000.0
    ld_raw = int(m.group("ld"))
    return BinInfo(
        bin_index=int(bin_index),
        bin_name=str(bin_name),
        maf_lo=lo,
        maf_hi=hi,
        ld_index=ld_raw - 1,
        ld_label=f"ld{ld_raw}",
        maf_label=f"m{m.group('lo')}_{m.group('hi')}",
    )
