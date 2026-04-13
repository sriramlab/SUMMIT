from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import utils

from moments import resolve_cov_rank, effective_n_scale, derived_wald_z


_CHI2_MEDIAN_1DF = 0.454936423119572

def _chisq_summary(chisq: np.ndarray) -> dict:
    chisq = np.asarray(chisq, dtype=np.float64)
    finite = np.isfinite(chisq)
    x = chisq[finite]
    out = {
        "M": int(chisq.size),
        "M_finite": int(x.size),
        "M_nonfinite": int(chisq.size - x.size),
    }
    if x.size == 0:
        return out
    out["mean"] = float(np.mean(x))
    out["median"] = float(np.median(x))
    out["max"] = float(np.max(x))
    out["lambda_gc"] = float(out["median"] / _CHI2_MEDIAN_1DF)
    for q in [90, 95, 99, 99.9, 99.99, 99.999]:
        out[f"p{q}"] = float(np.percentile(x, q))
    return out


def _top_chisq_rows(snps, a1, a2, chisq, topk: int = 10):
    if topk <= 0:
        return []
    chisq = np.asarray(chisq, dtype=np.float64)
    finite = np.isfinite(chisq)
    if not finite.any():
        return []
    idx = np.where(finite)[0]
    x = chisq[finite]
    k = min(topk, x.size)
    part = np.argpartition(-x, kth=k - 1)[:k]
    best = part[np.argsort(-x[part])]
    rows = []
    for t in best:
        j = int(idx[t])
        rows.append((str(snps[j]), str(a1[j]), str(a2[j]), float(chisq[j])))
    return rows

def _maybe_find_column(columns, candidates):
    cols = list(columns)
    lower_to_name = {str(c).lower(): c for c in cols}
    for cand in candidates:
        hit = lower_to_name.get(str(cand).lower())
        if hit is not None:
            return hit
    return None


@dataclass(frozen=True)
class MatchedSumstats:
    snps: np.ndarray
    z: np.ndarray
    chi2: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    nsamp: float
    n_scale: float
    cov_rank: int
    cov_rank_source: str
    name: str
    used_summary: dict | None = None
    used_top: list | None = None
    clip_count: int = 0
    clip_threshold: float | None = None

    @property
    def nsnps(self) -> int:
        return int(self.snps.size)

    def subset(self, keep_mask) -> "MatchedSumstats":
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self.nsnps:
            raise ValueError(
                f"keep_mask must be length {self.nsnps}; got {keep_mask.shape}."
            )
        return MatchedSumstats(
            snps=self.snps[keep_mask],
            z=self.z[keep_mask],
            chi2=self.chi2[keep_mask],
            beta=self.beta[keep_mask],
            se=self.se[keep_mask],
            n=self.n[keep_mask],
            a1=self.a1[keep_mask],
            a2=self.a2[keep_mask],
            nsamp=self.nsamp,
            n_scale=self.n_scale,
            cov_rank=self.cov_rank,
            cov_rank_source=self.cov_rank_source,
            name=self.name,
            used_summary=self.used_summary,
            used_top=self.used_top,
            clip_count=self.clip_count,
            clip_threshold=self.clip_threshold,
        )


@dataclass(frozen=True)
class AlignedSumstats:
    trace: object
    sumstats: "Sumstats"
    pos_on_trace: np.ndarray    # len(trace.snps), -1 if absent

    def matched_mask(self) -> np.ndarray:
        return self.pos_on_trace >= 0

    def keep_mask(self, *, chisq_threshold=None, chisq_action="drop") -> np.ndarray:
        mask = self.matched_mask()
        action = str(chisq_action).strip().lower()
        if action not in ("drop", "clip", "warn", "none"):
            raise ValueError(f"Invalid chisq_action={chisq_action!r}")

        thr, _ = utils._resolve_chisq_threshold(self.sumstats.n_scale, chisq_threshold)
        if action != "drop" or thr is None or (not np.isfinite(float(thr))) or float(thr) <= 0.0:
            return mask

        thr = float(thr)
        out = mask.copy()
        pos = self.pos_on_trace[mask]
        out[mask] = np.isfinite(self.sumstats.chi2[pos]) & (self.sumstats.chi2[pos] <= thr)
        return out

    def materialize(
        self,
        keep_mask,
        *,
        chisq_threshold=None,
        chisq_action="drop",
        allowed_mask=None,
        compute_diagnostics: bool = True,
    ) -> MatchedSumstats:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self.trace.nsnps:
            raise ValueError(
                f"keep_mask must be length {self.trace.nsnps}; got {keep_mask.shape}."
            )

        action = str(chisq_action).strip().lower()
        if action not in ("drop", "clip", "warn", "none"):
            raise ValueError(f"Invalid chisq_action={chisq_action!r}")

        if allowed_mask is None:
            allowed = self.keep_mask(chisq_threshold=chisq_threshold, chisq_action=chisq_action)
        else:
            allowed = np.asarray(allowed_mask, dtype=bool)
            if allowed.ndim != 1 or allowed.size != self.trace.nsnps:
                raise ValueError(
                    f"allowed_mask must be length {self.trace.nsnps}; got {allowed.shape}."
                )

        if np.any(keep_mask & ~allowed):
            bad = int(np.sum(keep_mask & ~allowed))
            raise ValueError(
                f"keep_mask contains {bad} SNPs that are unavailable under the requested "
                f"chi^2 policy ({action})."
            )

        pos = self.pos_on_trace[keep_mask]
        snps = self.trace.snps[keep_mask]
        z = self.sumstats.z[pos].astype(np.float64, copy=False)
        chi2 = self.sumstats.chi2[pos].astype(np.float64, copy=True)
        beta = self.sumstats.beta[pos].astype(np.float64, copy=False)
        se = self.sumstats.se[pos].astype(np.float64, copy=False)
        n = self.sumstats.n[pos].astype(np.float64, copy=False)
        a1 = self.sumstats.a1[pos]
        a2 = self.sumstats.a2[pos]

        thr, _ = utils._resolve_chisq_threshold(self.sumstats.n_scale, chisq_threshold)
        clip_count = 0
        clip_threshold = None
        if action == "clip" and thr is not None and np.isfinite(float(thr)) and float(thr) > 0.0:
            thr = float(thr)
            clip_threshold = thr
            clip_mask = np.isfinite(chi2) & (chi2 > thr)
            clip_count = int(np.sum(clip_mask))
            if clip_count > 0:
                np.minimum(chi2, thr, out=chi2)

        if compute_diagnostics:
            used_summary = _chisq_summary(chi2)
            used_top = _top_chisq_rows(snps, a1, a2, chi2, topk=10)
        else:
            used_summary = None
            used_top = None

        return MatchedSumstats(
            snps=snps,
            z=z,
            chi2=chi2,
            beta=beta,
            se=se,
            n=n,
            a1=a1,
            a2=a2,
            nsamp=float(self.sumstats.nsamp),
            n_scale=float(self.sumstats.n_scale),
            cov_rank=int(self.sumstats.cov_rank),
            cov_rank_source=str(self.sumstats.cov_rank_source),
            name=self.sumstats.name,
            used_summary=used_summary,
            used_top=used_top,
            clip_count=clip_count,
            clip_threshold=clip_threshold,
        )


class Sumstats:
    """
    Immutable phenotype-level summary statistics after intrinsic QC.

    Read-time QC only:
      - parse SNP / Z / N / A1 / A2
      - drop non-finite N or Z
      - rescale Z by sqrt(N / Nmax)
      - deduplicate SNP IDs

    No trace-specific matching or chi^2 thresholding happens here.
    """

    def __init__(
        self,
        *,
        snps,
        z,
        beta,
        se,
        n,
        nsamp,
        n_scale,
        cov_rank,
        cov_rank_source,
        a1,
        a2,
        name,
        log=None,
        removed_snps=None,
        read_summary=None,
        read_top=None,
    ):
        self.snps = np.asarray(snps, dtype=str)
        self.z = np.asarray(z, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.se = np.asarray(se, dtype=np.float64)
        self.n = np.asarray(n, dtype=np.float64)
        self.chi2 = self.z * self.z
        self.a1 = np.asarray(a1, dtype=str)
        self.a2 = np.asarray(a2, dtype=str)
        self.nsamp = float(nsamp)
        self.n_scale = float(n_scale)
        self.cov_rank = int(cov_rank)
        self.cov_rank_source = str(cov_rank_source)
        self.name = str(name)
        self.log = log
        self.removed_snps = [] if removed_snps is None else list(removed_snps)
        self.read_summary = read_summary
        self.read_top = read_top

        if not (
            self.snps.shape == self.z.shape == self.beta.shape == self.se.shape
            == self.n.shape == self.a1.shape == self.a2.shape
        ):
            raise RuntimeError("Sumstats arrays do not all have the same length.")

        self.index = pd.Index(self.snps)
        if self.index.has_duplicates:
            raise RuntimeError("Sumstats unexpectedly contains duplicate SNP IDs after deduplication.")

    @property
    def nsnps(self) -> int:
        return int(self.snps.size)

    @classmethod
    def from_file(
        cls,
        path,
        *,
        name=None,
        log=None,
        cov_rank=None,
        cov_rank_source=None,
        compute_diagnostics: bool = True,
    ) -> "Sumstats":
        hdr = pd.read_csv(path, sep=r"\s+", compression="infer", nrows=0)
        cols = list(hdr.columns)

        idcol = utils._parse_column_name(hdr, ["ID", "id", "snp", "SNP"], default_pos=0)
        a1col = utils._parse_column_name(hdr, ["A1", "ALT"], default_pos=1)
        a2col = utils._parse_column_name(hdr, ["A2", "REF"], default_pos=2)

        ncol = _maybe_find_column(cols, ["OBS_CT", "obs_ct", "N", "n"])
        if ncol is None:
            ncol = utils._parse_column_name(hdr, ["N", "n"], default_pos=3)

        betacol = _maybe_find_column(cols, ["BETA", "beta"])
        secol = _maybe_find_column(cols, ["SE", "se", "STDERR", "stderr"])
        if betacol is None or secol is None:
            raise RuntimeError(
                f"Phenotype [{name or path}] must contain BETA and SE columns."
            )

        covrankcol = _maybe_find_column(cols, ["COV_RANK", "cov_rank", "P_EFF", "p_eff"])

        usecols = [idcol, a1col, a2col, ncol, betacol, secol]
        if covrankcol is not None:
            usecols.append(covrankcol)
        usecols = list(dict.fromkeys(usecols))

        df = pd.read_csv(
            path,
            sep=r"\s+",
            compression="infer",
            usecols=usecols,
            dtype={idcol: str, a1col: str, a2col: str},
        )

        rename_map = {
            idcol: "SNP",
            a1col: "A1",
            a2col: "A2",
            ncol: "N",
            betacol: "BETA",
            secol: "SE",
        }
        if covrankcol is not None:
            rename_map[covrankcol] = "COV_RANK"

        df = df.rename(columns=rename_map)
        df["SNP"] = df["SNP"].astype(str)
        df["A1"] = df["A1"].astype(str).str.upper()
        df["A2"] = df["A2"].astype(str).str.upper()
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
        df["SE"] = pd.to_numeric(df["SE"], errors="coerce")

        file_cov_rank = None
        if "COV_RANK" in df.columns:
            cr = pd.to_numeric(df["COV_RANK"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            cr = cr[np.isfinite(cr)]
            if cr.size > 0:
                rcr = np.rint(cr)
                if np.any(np.abs(cr - rcr) > 1e-8):
                    raise RuntimeError(
                        f"Phenotype [{name or path}] has non-integer COV_RANK / P_EFF values."
                    )
                uniq = np.unique(rcr.astype(np.int64))
                if uniq.size != 1:
                    raise RuntimeError(
                        f"Phenotype [{name or path}] has non-constant COV_RANK / P_EFF values."
                    )
                file_cov_rank = int(uniq[0])
                if file_cov_rank < 0:
                    raise RuntimeError(
                        f"Phenotype [{name or path}] has negative COV_RANK / P_EFF={file_cov_rank}."
                    )

        if (cov_rank is not None) and (file_cov_rank is not None) and (int(cov_rank) != int(file_cov_rank)):
            if log is not None:
                log._log(
                    f"[sumstats] [{name or path}] overriding file cov_rank={int(file_cov_rank)} "
                    f"with explicit cov_rank={int(cov_rank)} "
                    f"(source={cov_rank_source or 'explicit'})."
                )

        n_arr = df["N"].to_numpy(dtype=np.float64, copy=False)
        beta_arr = df["BETA"].to_numpy(dtype=np.float64, copy=False)
        se_arr = df["SE"].to_numpy(dtype=np.float64, copy=False)

        bad = (
            (~np.isfinite(n_arr)) | (n_arr <= 0.0) |
            (~np.isfinite(beta_arr)) |
            (~np.isfinite(se_arr)) | (se_arr <= 0.0)
        )

        removed = []
        if bad.any():
            removed = df.loc[bad, "SNP"].dropna().astype(str).tolist()
            if log is not None:
                log._log(
                    f"Dropping {len(removed)} SNPs with NA/non-finite N/BETA/SE values [{name or path}]."
                )
            df = df.loc[~bad].copy()
        else:
            if log is not None:
                log._log(f"Dropping 0 SNPs with NA/non-finite N/BETA/SE values [{name or path}].")

        if df.shape[0] == 0:
            raise RuntimeError(f"No valid SNPs remain after basic filtering for phenotype [{name or path}].")

        n_arr = df["N"].to_numpy(dtype=np.float64, copy=False)
        beta_arr = df["BETA"].to_numpy(dtype=np.float64, copy=False)
        se_arr = df["SE"].to_numpy(dtype=np.float64, copy=False)

        nmax = float(np.max(n_arr))
        resolved_cov_rank, resolved_source = resolve_cov_rank(
            explicit=cov_rank,
            explicit_source=cov_rank_source,
            sumstats_value=file_cov_rank,
            sumstats_source="file",
        )
        n_scale = float(effective_n_scale(nmax, resolved_cov_rank))

        if log is not None:
            log._log(
                f"[sumstats] [{name or path}] using cov_rank={resolved_cov_rank} "
                f"(source={resolved_source}), n_scale={n_scale:.6g}."
            )

        z_scaled = derived_wald_z(beta_arr, se_arr, n_arr, n_scale)
        badz = ~np.isfinite(z_scaled)
        if badz.any():
            removed2 = df.loc[badz, "SNP"].astype(str).tolist()
            removed.extend(removed2)
            df = df.loc[~badz].copy()
            z_scaled = z_scaled[~badz]
            if log is not None:
                log._log(
                    f"Dropping {len(removed2)} SNPs with non-finite derived Z values [{name or path}]."
                )

        df["Z"] = z_scaled

        if df["SNP"].duplicated().any():
            df["_row"] = np.arange(df.shape[0], dtype=np.int64)
            df = (
                df.sort_values(["SNP", "N", "_row"], ascending=[True, False, True])
                .drop_duplicates(subset="SNP", keep="first")
                .sort_values("_row")
                .drop(columns=["_row"])
                .reset_index(drop=True)
            )
            if log is not None:
                log._log(
                    f"Detected duplicate SNP IDs; kept 1 row per SNP (remaining={df.shape[0]})."
                )
        else:
            df = df.reset_index(drop=True)

        if compute_diagnostics:
            read_summary = _chisq_summary(df["Z"].to_numpy(dtype=np.float64, copy=False) ** 2)
            read_top = _top_chisq_rows(
                df["SNP"].astype(str).to_numpy(),
                df["A1"].astype(str).to_numpy(),
                df["A2"].astype(str).to_numpy(),
                df["Z"].to_numpy(dtype=np.float64, copy=False) ** 2,
                topk=10,
            )
        else:
            read_summary = None
            read_top = None

        return cls(
            snps=df["SNP"].astype(str).to_numpy(),
            z=df["Z"].to_numpy(dtype=np.float64, copy=False),
            beta=df["BETA"].to_numpy(dtype=np.float64, copy=False),
            se=df["SE"].to_numpy(dtype=np.float64, copy=False),
            n=df["N"].to_numpy(dtype=np.float64, copy=False),
            nsamp=nmax,
            n_scale=n_scale,
            cov_rank=resolved_cov_rank,
            cov_rank_source=resolved_source,
            a1=df["A1"].astype(str).to_numpy(),
            a2=df["A2"].astype(str).to_numpy(),
            name=(name or path),
            log=log,
            removed_snps=removed,
            read_summary=read_summary,
            read_top=read_top,
        )

    def align_to_trace(self, trace) -> AlignedSumstats:
        pos = self.index.get_indexer(trace.snps)
        return AlignedSumstats(trace=trace, sumstats=self, pos_on_trace=pos)

    def log_chisq_diagnostics(
        self,
        matched: MatchedSumstats | None = None,
        *,
        chisq_threshold=None,
        topk: int = 10,
        warn_min_count: int = 10,
        warn_min_frac: float = 1e-5,
        verbose=False,
    ):
        if self.log is None:
            return

        def _should_expand(summ: dict) -> bool:
            if not summ or ("M_finite" not in summ):
                return False
            nscale = float(self.n_scale)
            thr = max(80.0, 0.001 * nscale)
            n_hi = int(summ.get("n_gt_suggested", 0)) if "n_gt_suggested" in summ else None
            if n_hi is None:
                return float(summ.get("max", 0.0)) > thr
            frac = float(summ.get("frac_gt_suggested", 0.0))
            return (n_hi >= warn_min_count) or (frac >= warn_min_frac)

        def _log_stage(label: str, summ: dict, top_rows: list, expand: bool):
            if not summ:
                return
            if "mean" in summ:
                self.log._log(
                    f"[chisq] [{self.name}] {label}  M={summ['M']} (finite={summ['M_finite']})  "
                    f"lambda_gc={summ['lambda_gc']:.4f}  mean={summ['mean']:.4f}  "
                    f"p99.9={summ.get('p99.9', float('nan')):.2f}  max={summ['max']:.2f}"
                )
            else:
                self.log._log(f"[chisq] [{self.name}] {label}  M={summ.get('M', 0)}")
            if expand and top_rows:
                self.log._log(f"[chisq] [{self.name}] top outliers (SNP A1 A2 chi2):")
                for i, (snp, aa1, aa2, chi2) in enumerate(top_rows[:topk], start=1):
                    self.log._log(f"[chisq] [{self.name}]  {i:2d}. {snp}\t{aa1}\t{aa2}\t{chi2:.3f}")

        thr, mode = utils._resolve_chisq_threshold(self.n_scale, chisq_threshold)
        filter_active = thr is not None and np.isfinite(float(thr)) and float(thr) > 0.0

        if verbose or filter_active:
            if filter_active:
                tag = " (auto)" if mode == "auto" else ""
                self.log._log(
                    f"[chisq] [{self.name}] active chi^2 filter: threshold={float(thr):.3f}{tag}; "
                    f"cov_rank={self.cov_rank} ({self.cov_rank_source}), n_scale={self.n_scale:.6g}"
                )
            _log_stage("read", self.read_summary, self.read_top, expand=True)

        if matched is not None:
            _log_stage(
                "used",
                matched.used_summary,
                matched.used_top,
                expand=(verbose or _should_expand(matched.used_summary)),
            )