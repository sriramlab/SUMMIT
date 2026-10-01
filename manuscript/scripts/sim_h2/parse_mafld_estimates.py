#!/usr/bin/env python3
"""
Parse SUM-RHE, RHE/GENIE, LDSC, covLDSC, and covSUM-RHE partitioned outputs into one wide CSV
that includes TOTAL h2 and per-bin h2/enrichment (with SEs when available).

Output columns per row (replicate):
  method, run, window, true_h2, p_causal, setting, num_bins,
  h2, h2_se,
  h2bin_0..h2bin_{B-1}, h2bin_se_0..,
  enr_0..enr_{B-1},     enr_se_0..   (NaN if not printed)

Paths expected:
  SUM-RHE:
    <outs-root>/{pop}/sumrhe/sims_{true_h2}_{p}_{setting}_{B}bins/sim_{i}.log

  covSUM-RHE (same log format as SUM-RHE):
    <outs-root>/{pop}/covsumrhe/sims_{true_h2}_{p}_{setting}_{B}bins/sim_{i}.log

  RHE/GENIE:
    <outs-root>/{pop}/rhe/sims_{true_h2}_{p}_{setting}_{B}bins/sim_{i}.log

  LDSC:
    <outs-root>/{pop}/ldsc/window_{W}kb/sims_{true_h2}_{p}_{setting}_{B}bins/sim_{i}.log
    and paired sim_{i}.results

  covLDSC (same log format as LDSC; no window subdir):
    <outs-root>/{pop}/covldsc/sims_{true_h2}_{p}_{setting}_{B}bins/sim_{i}.log
    and paired sim_{i}.results
"""
import os
import re
import glob
import math
import pandas as pd
from typing import List, Optional
import sys

# ---------- SUM-RHE / covSUM-RHE ----------
# allow floats OR 'nan' (case-insensitive) for the numeric fields
_NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUM_OR_NAN = rf"(?:{_NUM}|nan|NaN|NAN)"

SUM_BIN_RE = re.compile(
    rf"^\^\^\^\s+Phenotype\s+\d+\s+Bin\s+\[(?P<label>[^\]]+)\].*?"
    rf"h\^2_cat:\s+(?P<h2cat>{_NUM_OR_NAN})\s+\(SE:\s+(?P<h2cat_se>{_NUM_OR_NAN})\).*?"
    rf"Enrichment:\s+(?P<enr>{_NUM_OR_NAN})(?:\s+\(SE:\s+(?P<enr_se>{_NUM_OR_NAN})\))?",
    re.ASCII | re.IGNORECASE,
)

SUM_TOTAL_RE = re.compile(
    r"^\^\^\^\s+Phenotype\s+\d+\s+Total SNP heritability\s+\(h\^2\):\s+"
    r"(?P<h2>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+SE:\s+(?P<se>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.ASCII,
)


def parse_sumrhe_log(path: str, num_bins: int):
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return math.nan

    h2, h2_se = math.nan, math.nan
    h2bins, h2bins_se, enrs, enrs_se = [], [], [], []
    with open(path) as f:
        for line in f:
            m = SUM_BIN_RE.search(line)
            if m:
                h2bins.append(_to_float(m.group("h2cat")))
                h2bins_se.append(_to_float(m.group("h2cat_se")))
                enrs.append(_to_float(m.group("enr")))
                enrs_se.append(_to_float(m.group("enr_se")))
                continue
            t = SUM_TOTAL_RE.search(line)
            if t:
                h2 = _to_float(t.group("h2"))
                h2_se = _to_float(t.group("se"))

    # pad/trim to num_bins (preserve parsed h2bin/h2bin_se even if enrichment was NaN)
    for lst in (h2bins, h2bins_se, enrs, enrs_se):
        if len(lst) < num_bins:
            lst += [math.nan] * (num_bins - len(lst))
        elif len(lst) > num_bins:
            del lst[num_bins:]
    return h2, h2_se, h2bins, h2bins_se, enrs, enrs_se


def _parse_sumrhe_like_partitioned(
    base: str,
    method_label: str,
    settings: List[str],
    bins_list: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    for setting in settings:
        for B in bins_list:
            sim_dir = f"{base}/sims_{true_h2}_{p_causal}_{setting}_{B}bins"
            if not os.path.isdir(sim_dir):
                print(f"[warn] {method_label} missing: {sim_dir}")
                continue
            logs = sorted(glob.glob(os.path.join(sim_dir, "sim_*.log")))
            if max_runs:
                logs = logs[:max_runs]
            for lp in logs:
                run_m = re.search(r"sim_(\d+)\.log$", os.path.basename(lp))
                if not run_m:
                    continue
                run = int(run_m.group(1))
                h2, h2_se, hb, hb_se, er, er_se = parse_sumrhe_log(lp, B)
                row = {
                    "method": method_label,
                    "run": run,
                    "window": -1,
                    "true_h2": true_h2,
                    "p_causal": p_causal,
                    "setting": setting,
                    "num_bins": B,
                    "h2": h2,
                    "h2_se": h2_se,
                }
                for j in range(B):
                    row[f"h2bin_{j}"] = hb[j]
                    row[f"h2bin_se_{j}"] = hb_se[j]
                    row[f"enr_{j}"] = er[j]
                    row[f"enr_se_{j}"] = er_se[j]
                rows.append(row)
    return pd.DataFrame(rows)


def parse_sumrhe_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    base = f"{OUTS_ROOT}/{pop}/sumrhe"
    return _parse_sumrhe_like_partitioned(
        base, "sumrhe", settings, bins_list, true_h2, p_causal, max_runs
    )


def parse_covsumrhe_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    base = f"{OUTS_ROOT}/{pop}/covsumrhe"
    return _parse_sumrhe_like_partitioned(
        base, "covsumrhe", settings, bins_list, true_h2, p_causal, max_runs
    )


# ---------- RHE / GENIE ----------
RHE_TOTAL_RE = re.compile(
    r"^Total h2\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*SE:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.ASCII,
)
RHE_H2BIN_RE = re.compile(
    r"^h2_g\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*SE\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.ASCII,
)
RHE_ENR_RE = re.compile(
    r"^Enrichment\s+g\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*SE\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.ASCII,
)


def parse_rhe_log(path: str, num_bins: int):
    h2, h2_se = math.nan, math.nan
    hb = [math.nan] * num_bins
    hbse = [math.nan] * num_bins
    enr = [math.nan] * num_bins
    ense = [math.nan] * num_bins
    with open(path) as f:
        for line in f:
            if m := RHE_TOTAL_RE.search(line):
                h2 = float(m.group(1))
                h2_se = float(m.group(2))
            elif m := RHE_H2BIN_RE.search(line):
                i = int(m.group(1))
                if 0 <= i < num_bins:
                    hb[i] = float(m.group(2))
                    hbse[i] = float(m.group(3))
            elif m := RHE_ENR_RE.search(line):
                i = int(m.group(1))
                if 0 <= i < num_bins:
                    enr[i] = float(m.group(2))
                    ense[i] = float(m.group(3))
    return h2, h2_se, hb, hbse, enr, ense


def parse_rhe_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    base = f"{OUTS_ROOT}/{pop}/rhe"
    for setting in settings:
        for B in bins_list:
            d1 = f"{base}/sims_{true_h2}_{p_causal}_{setting}_{B}bins"
            d2 = f"{base}/sims_{true_h2}_{p_causal}_{setting}"
            sim_dir = d1 if os.path.isdir(d1) else (d2 if os.path.isdir(d2) else None)
            if not sim_dir:
                print(f"[warn] RHE missing: {d1} (and {d2})")
                continue
            pats = ["sim_*.log", "*.log", "*.out", "*.txt"]
            logs = sorted(
                {p for pat in pats for p in glob.glob(os.path.join(sim_dir, pat))}
            )
            if max_runs:
                logs = logs[:max_runs]
            for idx, lp in enumerate(logs):
                m = re.search(r"sim_(\d+)\.", os.path.basename(lp))
                run = int(m.group(1)) if m else idx
                h2, h2_se, hb, hbse, enr, ense = parse_rhe_log(lp, B)
                row = {
                    "method": "rhe",
                    "run": run,
                    "window": -1,
                    "true_h2": true_h2,
                    "p_causal": p_causal,
                    "setting": setting,
                    "num_bins": B,
                    "h2": h2,
                    "h2_se": h2_se,
                }
                for j in range(B):
                    row[f"h2bin_{j}"] = hb[j]
                    row[f"h2bin_se_{j}"] = hbse[j]
                    row[f"enr_{j}"] = enr[j]
                    row[f"enr_se_{j}"] = ense[j]
                rows.append(row)
    return pd.DataFrame(rows)


# ---------- LDSC / covLDSC ----------
LDSC_TOTAL_RE = re.compile(
    r"^Total\s+(?:Observed\s+scale\s+)?h2:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
    re.ASCII,
)


def parse_ldsc_total_from_log(log_path: str):
    """Return (total_h2, total_se) from sim_#.log; NaN if missing."""
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
    # normalize result headers like 'Prop._h2' -> 'prop_h2'
    return [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in cols]


def parse_ldsc_results_table(results_path: str, num_bins: int, total_h2: float):
    """
    Read sim_#.results and return per-bin (h2, h2_se, enrichment, enrichment_se) arrays.
    - uses Prop._h2 and Prop._h2_std_error and scales them by total_h2 to get absolute h2, h2_se
    - preserves row order; pads/trims to num_bins
    """
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


def parse_ldsc_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    windows_kb: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    base = f"{OUTS_ROOT}/{pop}/ldsc"
    for W in windows_kb:
        wdir = f"{base}/window_{W}kb"
        for setting in settings:
            for B in bins_list:
                sim_dir = f"{wdir}/sims_{true_h2}_{p_causal}_{setting}_{B}bins"
                if not os.path.isdir(sim_dir):
                    print(f"[warn] LDSC missing: {sim_dir}")
                    continue
                logs = sorted(glob.glob(os.path.join(sim_dir, "sim_*.log")))
                if max_runs:
                    logs = logs[:max_runs]
                for lp in logs:
                    run = int(
                        re.search(r"sim_(\d+)\.log$", os.path.basename(lp)).group(1)
                    )
                    tot_h2, tot_se = parse_ldsc_total_from_log(lp)
                    rp = lp.replace(".log", ".results")
                    h2bins, h2bins_se, enr, enr_se = parse_ldsc_results_table(
                        rp, B, tot_h2
                    )

                    row = {
                        "method": "ldsc",
                        "run": run,
                        "window": W,
                        "true_h2": true_h2,
                        "p_causal": p_causal,
                        "setting": setting,
                        "num_bins": B,
                        "h2": tot_h2,
                        "h2_se": tot_se,
                    }
                    for j in range(B):
                        row[f"h2bin_{j}"] = h2bins[j]
                        row[f"h2bin_se_{j}"] = h2bins_se[j]
                        row[f"enr_{j}"] = enr[j]
                        row[f"enr_se_{j}"] = enr_se[j]
                    rows.append(row)
    return pd.DataFrame(rows)


def parse_covldsc_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    true_h2: float,
    p_causal: float,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    """
    covLDSC: same parsing as LDSC, but lives at:
      {OUTS_ROOT}/{pop}/covldsc/sims_{h2}_{p}_{setting}_{B}bins/sim_{i}.{log,results}
    No window directory -> window=-1 in output.
    """
    rows = []
    base = f"{OUTS_ROOT}/{pop}/covldsc"
    for setting in settings:
        for B in bins_list:
            sim_dir = f"{base}/sims_{true_h2}_{p_causal}_{setting}_{B}bins"
            if not os.path.isdir(sim_dir):
                print(f"[warn] covLDSC missing: {sim_dir}")
                continue
            logs = sorted(glob.glob(os.path.join(sim_dir, "sim_*.log")))
            if max_runs:
                logs = logs[:max_runs]
            for lp in logs:
                run = int(re.search(r"sim_(\d+)\.log$", os.path.basename(lp)).group(1))
                tot_h2, tot_se = parse_ldsc_total_from_log(lp)
                rp = lp.replace(".log", ".results")
                h2bins, h2bins_se, enr, enr_se = parse_ldsc_results_table(rp, B, tot_h2)

                row = {
                    "method": "covldsc",
                    "run": run,
                    "window": -1,
                    "true_h2": true_h2,
                    "p_causal": p_causal,
                    "setting": setting,
                    "num_bins": B,
                    "h2": tot_h2,
                    "h2_se": tot_se,
                }
                for j in range(B):
                    row[f"h2bin_{j}"] = h2bins[j]
                    row[f"h2bin_se_{j}"] = h2bins_se[j]
                    row[f"enr_{j}"] = enr[j]
                    row[f"enr_se_{j}"] = enr_se[j]
                rows.append(row)
    return pd.DataFrame(rows)


# ---------- SumHer (partitioned; simplest) ----------
def _read_hers_simple(hers_path: str, num_bins: int):
    """
    Read sim_#.hers with columns:
      Component  Heritability  SE  Influence  SE
    Returns: total_h2, total_se, h2bins[num_bins], h2bins_se[num_bins]
    """
    tot_h2 = math.nan
    tot_se = math.nan
    hb = [math.nan] * num_bins
    hbse = [math.nan] * num_bins

    if not os.path.exists(hers_path):
        return tot_h2, tot_se, hb, hbse

    df = pd.read_csv(hers_path, sep=r"\s+", engine="python")
    cols = list(df.columns)

    comp_col = "Component"
    h2_col = "Heritability"
    h2_se_col = cols[cols.index(h2_col) + 1]  # SE right after Heritability

    for _, r in df.iterrows():
        comp = str(r[comp_col])
        if comp == "Her_All":
            tot_h2 = float(r[h2_col]) if pd.notna(r[h2_col]) else math.nan
            tot_se = float(r[h2_se_col]) if pd.notna(r[h2_se_col]) else math.nan
        elif comp.startswith("Her_P"):
            idx = int(comp.split("Her_P", 1)[1]) - 1
            if 0 <= idx < num_bins:
                hb[idx] = float(r[h2_col]) if pd.notna(r[h2_col]) else math.nan
                hbse[idx] = float(r[h2_se_col]) if pd.notna(r[h2_se_col]) else math.nan

    return tot_h2, tot_se, hb, hbse


def _read_enrich_simple(enrich_path: str, num_bins: int):
    """
    Read sim_#.enrich with columns:
      Component  Share  SE  Expected  Enrichment  SE  Z-Stat1  Z-Stat2
    Returns: enr[num_bins], enr_se[num_bins]
    """
    enr = [math.nan] * num_bins
    ense = [math.nan] * num_bins

    if not os.path.exists(enrich_path):
        return enr, ense

    df = pd.read_csv(enrich_path, sep=r"\s+", engine="python")
    cols = list(df.columns)

    comp_col = "Component"
    enr_col = "Enrichment"
    enr_se_col = cols[cols.index(enr_col) + 1]  # SE right after Enrichment

    for _, r in df.iterrows():
        comp = str(r[comp_col])
        if comp.startswith("Enrich_P"):
            idx = int(comp.split("Enrich_P", 1)[1]) - 1
            if 0 <= idx < num_bins:
                v = r[enr_col]
                se = r[enr_se_col]
                enr[idx] = float(v) if pd.notna(v) else math.nan
                ense[idx] = float(se) if pd.notna(se) else math.nan

    return enr, ense


def parse_sumher_partitioned(
    pop: str,
    settings: List[str],
    bins_list: List[int],
    windows_kb: List[int],
    true_h2: float,
    p_causal: float,
    method_name: str,
    max_runs: Optional[int] = None,
) -> pd.DataFrame:
    """
    method_name: 'sumher' (GCTA) or 'sumher_ldak' (LDAK)
    Looks under:
      {OUTS_ROOT}/{pop}/{method_name}/window_{W}kb/sims_{h2}_{p}_{setting}_{B}bins/
    Only parses files named exactly:
      sim_<idx>.hers  and  sim_<idx>.enrich
    """
    rows = []
    base = f"{OUTS_ROOT}/{pop}/{method_name}"
    for W in windows_kb:
        wdir = f"{base}/window_{W}kb"
        for setting in settings:
            for B in bins_list:
                sim_dir = f"{wdir}/sims_{true_h2}_{p_causal}_{setting}_{B}bins"
                if not os.path.isdir(sim_dir):
                    print(f"[warn] {method_name} missing: {sim_dir}")
                    continue

                hers_files = sorted(glob.glob(os.path.join(sim_dir, "sim_*.hers")))
                hers_files = [
                    p
                    for p in hers_files
                    if re.match(r"^sim_\d+\.hers$", os.path.basename(p))
                ]
                if max_runs:
                    hers_files = hers_files[:max_runs]

                for hp in hers_files:
                    run = int(
                        re.match(r"^sim_(\d+)\.hers$", os.path.basename(hp)).group(1)
                    )
                    ep = os.path.join(sim_dir, f"sim_{run}.enrich")
                    if not os.path.exists(ep):
                        print(f"[warn] missing enrich for run {run}: {ep}")
                    tot_h2, tot_se, hb, hbse = _read_hers_simple(hp, B)
                    enr, ense = _read_enrich_simple(ep, B)

                    row = {
                        "method": method_name,
                        "run": run,
                        "window": W,
                        "true_h2": true_h2,
                        "p_causal": p_causal,
                        "setting": setting,
                        "num_bins": B,
                        "h2": tot_h2,
                        "h2_se": tot_se,
                    }
                    for j in range(B):
                        row[f"h2bin_{j}"] = hb[j]
                        row[f"h2bin_se_{j}"] = hbse[j]
                        row[f"enr_{j}"] = enr[j]
                        row[f"enr_se_{j}"] = ense[j]
                    rows.append(row)

    return pd.DataFrame(rows)


# ---------- optional log-to-table preparation ----------
def main():
    import argparse
    from pathlib import Path

    global OUTS_ROOT
    ap = argparse.ArgumentParser(
        description="Parse simulation estimate logs into the S12/S13 plotting tables. Original method logs are not included in the public package."
    )
    ap.add_argument("--outs-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    OUTS_ROOT = args.outs_root
    if not OUTS_ROOT.is_dir():
        ap.error("Input method-output directory does not exist")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        ap.error("Output directory must be new or empty")
    columns = [
        "method",
        "run",
        "window",
        "true_h2",
        "p_causal",
        "setting",
        "num_bins",
        "h2",
        "h2_se",
    ]
    for pop in ["EUR", "SAS", "AFR"]:
        frames = []
        for pc in [1.0, 0.01]:
            common = (pop, ["maf01", "maf05", "maf01_ldak", "maf05_ldak"], [24])
            frames.extend(
                [
                    parse_covsumrhe_partitioned(*common, 0.5, pc),
                    parse_rhe_partitioned(*common, 0.5, pc),
                    parse_covldsc_partitioned(*common, 0.5, pc),
                    parse_ldsc_partitioned(*common, [20000], 0.5, pc),
                    parse_sumher_partitioned(
                        *common, [20000], 0.5, pc, method_name="sumher"
                    ),
                    parse_sumher_partitioned(
                        *common, [20000], 0.5, pc, method_name="sumher_ldak"
                    ),
                ]
            )
        frame = pd.concat(frames, ignore_index=True)[columns]
        if len(frame) != 4800 or frame.duplicated(columns[:7]).any():
            raise ValueError(
                f"{pop}: expected 4800 unique method/setting/replicate records"
            )
        destination = args.out_dir / pop / "estimates.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(
            f"{pop}: {len(frame)} records; {frame.h2.notna().sum()} nonmissing h2 entries"
        )


if __name__ == "__main__":
    main()
