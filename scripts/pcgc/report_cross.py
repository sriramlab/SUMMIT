#!/usr/bin/env python3
"""Bivariate qualification with fixed covariance/rg margins and failure counts."""
import argparse
import json

import numpy as np

from report_pcgc import collect, metrics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = collect(args.runs)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("phase", "scenario", "risk", "method"))
        groups.setdefault(key, []).append(row)
    results = []
    for key, group in sorted(groups.items()):
        for quantity, truth in (("conditional_covariance", "truth_covariance"), ("genetic_correlation", "truth_rg")):
            def values(field):
                return [0. if field == "truth_covariance" and r["truth_rg"] == 0 else
                        r[field] if r.get(field) is not None else np.nan for r in group]
            result = metrics(values(quantity), values(truth), values(quantity+"_standard_error"), len(group),
                             component=True, absolute_margin=.05 if quantity == "genetic_correlation" else None)
            result.update(dict(zip(("phase", "scenario", "risk", "method"), key)), quantity=quantity,
                          all_ratio_deletions_finite=sum(r.get("finite_ratio_deletions", 0) == r["jackknife_blocks"] for r in group))
            results.append(result)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"report.json").write_text(json.dumps(dict(sources=sources, screens=results), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
