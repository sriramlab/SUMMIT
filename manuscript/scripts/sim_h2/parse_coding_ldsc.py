"""Plot parse coding ldsc for the SUMMIT manuscript."""
from __future__ import annotations

import os, re, glob, math, pandas as pd


LDSC_TOTAL_RE = re.compile(
    r"^Total\s+(?:Observed\s+scale\s+)?h2:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
    re.ASCII,
)


def parse_ldsc_total_from_log(log_path: str):
    h2, se = math.nan, math.nan
    with open(log_path) as f:
        for line in f:
            m = LDSC_TOTAL_RE.search(line)
            if m:
                h2 = float(m.group(1))
                se = float(m.group(2))
                break
    return h2, se


def _stdize_cols(cols):
    return [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in cols]


def parse_ldsc_results_table(results_path: str, num_bins: int, total_h2: float):
    if not os.path.exists(results_path):
        return (
            [math.nan] * num_bins,
            [math.nan] * num_bins,
            [math.nan] * num_bins,
            [math.nan] * num_bins,
        )

    df = pd.read_csv(results_path, sep=r"\s+", engine="python")
    df.columns = _stdize_cols(df.columns.tolist())

    c_prop = next((c for c in df.columns if c in ("prop_h2", "prop_h2_")), None)
    c_prop_se = next(
        (
            c
            for c in df.columns
            if c in ("prop_h2_std_error", "prop_h2_se", "prop_h2_stderr")
        ),
        None,
    )
    c_enr = next((c for c in df.columns if c == "enrichment"), None)
    c_enr_se = next(
        (
            c
            for c in df.columns
            if c in ("enrichment_std_error", "enrichment_se", "enrichment_stderr")
        ),
        None,
    )

    if c_prop is None or c_prop_se is None or c_enr is None or c_enr_se is None:
        return (
            [math.nan] * num_bins,
            [math.nan] * num_bins,
            [math.nan] * num_bins,
            [math.nan] * num_bins,
        )

    dsub = df.iloc[:num_bins].copy()

    h2bins = (dsub[c_prop].astype(float) * total_h2).tolist()
    h2bins_se = (dsub[c_prop_se].astype(float) * total_h2).tolist()
    enr = dsub[c_enr].astype(float).tolist()
    enr_se = dsub[c_enr_se].astype(float).tolist()

    for arr in (h2bins, h2bins_se, enr, enr_se):
        if len(arr) < num_bins:
            arr += [math.nan] * (num_bins - len(arr))
        elif len(arr) > num_bins:
            del arr[num_bins:]

    return h2bins, h2bins_se, enr, enr_se


def main():
    """Add the 900 complete 20-Mb LDSC fits to each calibration table."""
    import argparse
    from pathlib import Path
    import numpy as np

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    for pop in ["EUR", "SAS", "AFR"]:
        name = "estimates.csv"
        original = pd.read_csv(
            args.existing_root / pop / name, float_precision="round_trip"
        )
        # Preserve the original row order for the original bootstrap sampling.
        records = []
        for true_h2 in [0.1, 0.25, 0.4]:
            for p_causal in [1.0, 0.1, 0.01]:
                folder = (
                    args.results_root
                    / pop
                    / "ldsc/window_20000kb"
                    / f"sims_{true_h2}_{p_causal}"
                )
                for run in sorted(range(100), key=lambda i: f"sim_{i}.log"):
                    log = folder / f"sim_{run}.log"
                    text = log.read_text()
                    assert (
                        "Traceback" not in text
                        and "Results printed to" in text
                        and "Analysis finished" in text
                    ), log
                    assert "--n-blocks 100" in text and "w_20000kb" in text, log
                    h2, se = parse_ldsc_total_from_log(log)
                    hb, hbse, enr, ense = parse_ldsc_results_table(
                        folder / f"sim_{run}.results", 2, h2
                    )
                    row = dict(
                        method="ldsc",
                        run=run,
                        window=20000,
                        true_h2=true_h2,
                        p_causal=p_causal,
                        setting="calib",
                        num_bins=2,
                        h2=h2,
                        h2_se=se,
                    )
                    for j in range(2):
                        row.update(
                            {
                                f"h2bin_{j}": hb[j],
                                f"h2bin_se_{j}": hbse[j],
                                f"enr_{j}": enr[j],
                                f"enr_se_{j}": ense[j],
                            }
                        )
                    records.append(row)
        new = pd.DataFrame(records)
        assert len(new) == 900
        keep = original.loc[
            ~(original.method.eq("ldsc") & original.window.eq(20000))
        ].copy()
        result = pd.concat([keep, new], ignore_index=True)
        assert not result.duplicated(
            ["method", "window", "true_h2", "p_causal", "run"]
        ).any()
        destination = args.out_root / pop / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        finite = int(np.isfinite((new.enr_0 - 1) / new.enr_se_0).sum())
        print(
            f"{pop}: added 900 complete 20-Mb LDSC fits; {finite} finite coding Z statistics; "
            f"{len(result)} total rows."
        )


if __name__ == "__main__":
    main()
