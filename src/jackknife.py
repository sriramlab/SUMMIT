from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Literal

import numpy as np

import utils


JackknifeMode = Literal["block", "chr"]


@dataclass(frozen=True)
class JackknifeSpec:
    mode: JackknifeMode
    nblocks: int | None = None
    delete: int = 1
    nrep: int | None = None
    seed: int | None = None
    block_balance: str = "snp"      # snp | ldscore | ldtrace
    min_block_snps: int = 1

    @classmethod
    def parse(
        cls,
        njack,
        *,
        block_balance: str = "snp",
        min_block_snps: int = 1,
    ) -> "JackknifeSpec":
        block_balance = str(block_balance).strip().lower()
        if block_balance not in {"snp", "ldscore", "ldtrace"}:
            raise ValueError(
                f"block_balance must be one of {{'snp','ldscore','ldtrace'}}, got {block_balance!r}"
            )

        min_block_snps = int(min_block_snps)
        if min_block_snps <= 0:
            raise ValueError("min_block_snps must be >= 1")

        if njack is None:
            return cls(
                mode="block",
                nblocks=100,
                block_balance=block_balance,
                min_block_snps=min_block_snps,
            )

        if isinstance(njack, str):
            s = njack.strip().lower()
            if s.startswith("chr"):
                rest = s[3:]
                delete = 1
                nrep = None
                seed = None

                if rest == "":
                    pass
                elif rest.startswith(":"):
                    parts = s.split(":")
                    if len(parts) >= 2 and parts[1] != "":
                        delete = int(parts[1])
                    if len(parts) >= 3 and parts[2] != "":
                        nrep = int(parts[2])
                    if len(parts) >= 4 and parts[3] != "":
                        seed = int(parts[3])
                    if len(parts) > 4:
                        raise ValueError(
                            f"Invalid njack spec {njack!r}. Use chr[:d[:nrep[:seed]]]."
                        )
                else:
                    if rest.isdigit():
                        delete = int(rest)
                    else:
                        raise ValueError(
                            f"Invalid njack spec {njack!r}. Use chr[:d[:nrep[:seed]]]."
                        )

                if delete < 1:
                    raise ValueError(f"delete-d must be >= 1; got {delete}")
                if nrep is not None and nrep <= 0:
                    raise ValueError(f"nrep must be positive; got {nrep}")
                return cls(mode="chr", delete=delete, nrep=nrep, seed=seed)

            nblocks = int(float(s))
            if nblocks <= 0:
                raise ValueError(f"njack must be positive; got {njack!r}")
            return cls(
                mode="block",
                nblocks=nblocks,
                block_balance=block_balance,
                min_block_snps=min_block_snps,
            )

        nblocks = int(njack)
        if nblocks <= 0:
            raise ValueError(f"njack must be positive; got {njack!r}")
        return cls(
            mode="block",
            nblocks=nblocks,
            block_balance=block_balance,
            min_block_snps=min_block_snps,
        )


def _equal_count_block_boundaries(M: int, B: int):
    if not (1 <= B <= M):
        raise ValueError(f"Need 1 <= nblocks <= nsnps, got B={B}, M={M}")
    edges = np.floor(np.linspace(0, M, B + 1)).astype(np.int64)
    edges[-1] = M
    starts = edges[:-1].copy()
    ends = edges[1:].copy()
    if np.any(ends <= starts):
        raise RuntimeError("Equal-count partition produced an empty block.")
    return starts, ends


def _per_snp_block_mass(trace_view, metric: str) -> np.ndarray:
    metric = str(metric).strip().lower()

    if metric == "ldscore":
        L = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")
        if L.ndim != 2:
            raise ValueError("trace_view.ldscores must be 2D")
        q = np.sum(L, axis=1, dtype=np.float64)

    elif metric == "ldtrace":
        A = np.asarray(trace_view.annot, dtype=np.float64, order="C")
        L = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")

        if A.ndim != 2 or L.ndim != 2 or A.shape != L.shape:
            raise ValueError(
                f"ldtrace balancing requires annot and ldscores with identical 2D shape; "
                f"got A.shape={A.shape}, L.shape={L.shape}"
            )

        Ak = np.sum(A, axis=0, dtype=np.float64)
        invAk = np.zeros_like(Ak, dtype=np.float64)
        good = np.isfinite(Ak) & (Ak > 0.0)
        invAk[good] = 1.0 / Ak[good]

        # q_j = (a_j / Ak).sum() * (l_j / Ak).sum()
        q = (A @ invAk) * (L @ invAk)

    else:
        raise ValueError(f"Unsupported block balance metric {metric!r}")

    q = np.asarray(q, dtype=np.float64).ravel()
    q = np.where(np.isfinite(q) & (q > 0.0), q, 0.0)
    return q


def _weighted_contiguous_block_boundaries(
    mass: np.ndarray,
    B: int,
    *,
    min_snps: int = 1,
):
    mass = np.asarray(mass, dtype=np.float64).ravel()
    M = int(mass.size)

    if not (1 <= B <= M):
        raise ValueError(f"Need 1 <= nblocks <= nsnps, got B={B}, M={M}")

    min_snps = int(min_snps)
    if min_snps <= 0:
        raise ValueError("min_snps must be >= 1")
    if min_snps * B > M:
        raise ValueError(
            f"min_snps * nblocks exceeds nsnps: {min_snps} * {B} > {M}"
        )

    mass = np.where(np.isfinite(mass) & (mass > 0.0), mass, 0.0)
    total = float(np.sum(mass))
    if not (np.isfinite(total) and total > 0.0):
        return _equal_count_block_boundaries(M, B)

    cs = np.cumsum(mass, dtype=np.float64)
    starts = np.empty(B, dtype=np.int64)
    ends = np.empty(B, dtype=np.int64)

    s = 0
    for b in range(B):
        starts[b] = s
        rem = B - b - 1

        if rem == 0:
            e = M
        else:
            low = s + min_snps
            high = M - rem * min_snps

            # global weighted-quantile target
            target = (b + 1) * total / B
            e = int(np.searchsorted(cs, target, side="left") + 1)

            if e < low:
                e = low
            if e > high:
                e = high

        ends[b] = e
        s = e

    if starts[0] != 0 or ends[-1] != M or np.any(ends <= starts):
        raise RuntimeError("Weighted contiguous partition produced invalid block boundaries.")
    return starts, ends


@dataclass(frozen=True)
class JackknifeDesign:
    spec: JackknifeSpec
    nrep: int
    nunit: int
    unit_id: np.ndarray         # (M,)
    unit_labels: np.ndarray     # (U,)
    delete_sets: np.ndarray     # (R, d)
    D: np.ndarray               # (R, U)
    starts: np.ndarray          # (U,)
    ends: np.ndarray            # (U,)
    nsnps: int

    @property
    def mode(self) -> JackknifeMode:
        return self.spec.mode

    @property
    def delete(self) -> int:
        return int(self.spec.delete)

    @classmethod
    def from_trace_view(cls, trace_view, spec: JackknifeSpec, log=None) -> "JackknifeDesign":
        M = int(trace_view.nsnps)
        if M <= 0:
            raise ValueError("Cannot construct JackknifeDesign on an empty TraceView.")

        if spec.mode == "block":
            B = int(spec.nblocks)
            if B <= 0:
                raise ValueError("block jackknife requires nblocks > 0")
            if B > M:
                raise ValueError(f"nblocks cannot exceed nsnps; got B={B}, M={M}")

            balance = str(spec.block_balance).strip().lower()

            if balance == "snp":
                starts, ends = _equal_count_block_boundaries(M, B)
                mass = None
            else:
                mass = _per_snp_block_mass(trace_view, balance)
                starts, ends = _weighted_contiguous_block_boundaries(
                    mass,
                    B,
                    min_snps=int(spec.min_block_snps),
                )

            unit_id = np.empty(M, dtype=np.int64)
            for u, (s, e) in enumerate(zip(starts, ends)):
                unit_id[s:e] = u

            unit_labels = np.arange(B, dtype=np.int64)
            delete_sets = np.arange(B, dtype=np.int16)[:, None]
            D = np.eye(B, dtype=np.float64)

            if log is not None:
                n_per = (ends - starts).astype(np.float64)
                msg = (
                    f"[jackknife] block mode with {B} contiguous blocks; "
                    f"balance={balance}, "
                    f"SNP-count mean={n_per.mean():.2f}, "
                    f"min={int(n_per.min())}, max={int(n_per.max())}"
                )
                if mass is not None:
                    q_per = np.array(
                        [mass[s:e].sum(dtype=np.float64) for s, e in zip(starts, ends)],
                        dtype=np.float64,
                    )
                    cv_q = float(np.std(q_per) / np.mean(q_per)) if np.mean(q_per) > 0 else np.nan
                    cv_n = float(np.std(n_per) / np.mean(n_per)) if np.mean(n_per) > 0 else np.nan
                    msg += f", block-mass CV={cv_q:.4f}, SNP-count CV={cv_n:.4f}"
                log._log(msg)

            return cls(
                spec=spec,
                nrep=B,
                nunit=B,
                unit_id=unit_id,
                unit_labels=unit_labels,
                delete_sets=delete_sets,
                D=D,
                starts=starts,
                ends=ends,
                nsnps=M,
            )

        # chr delete-d
        if trace_view.chr is None:
            raise ValueError("chr jackknife requires chromosome labels in TraceView.")

        chr_arr = np.asarray(trace_view.chr, dtype=np.int32).ravel()
        if chr_arr.size != M:
            raise ValueError("TraceView chromosome vector length mismatch.")
        if chr_arr.size > 1 and np.any(chr_arr[1:] < chr_arr[:-1]):
            raise ValueError("chr jackknife requires SNPs sorted by nondecreasing CHR.")

        unit_labels = np.unique(chr_arr)
        unit_labels = unit_labels[np.isfinite(unit_labels)]
        unit_labels = np.asarray(unit_labels, dtype=np.int32)
        unit_labels.sort()
        U = int(unit_labels.size)
        d = int(spec.delete)

        if not (1 <= d < U):
            raise ValueError(
                f"delete-d must satisfy 1 <= d < #chromosomes; got d={d}, U={U}."
            )

        starts = np.searchsorted(chr_arr, unit_labels, side="left").astype(np.int64)
        ends = np.searchsorted(chr_arr, unit_labels, side="right").astype(np.int64)

        unit_id = np.searchsorted(unit_labels, chr_arr).astype(np.int64)
        valid = (unit_id >= 0) & (unit_id < U) & (unit_labels[unit_id] == chr_arr)
        if not np.all(valid):
            bad = np.flatnonzero(~valid)[:10].tolist()
            raise RuntimeError(
                f"Found chromosomes not represented in the jackknife units. First bad indices: {bad}"
            )

        total = math.comb(U, d)
        if spec.nrep is None or spec.nrep >= total:
            delete_sets = np.fromiter(
                (x for combi in combinations(range(U), d) for x in combi),
                dtype=np.int16,
                count=total * d,
            ).reshape(total, d)
            R = int(total)
            if log is not None:
                log._log(
                    f"[jackknife] chr delete-{d}: using ALL combinations C({U},{d}) = {R}."
                )
        else:
            rng = np.random.default_rng(spec.seed)
            all_combos = list(combinations(range(U), d))
            pick = rng.choice(len(all_combos), size=int(spec.nrep), replace=False)
            delete_sets = np.asarray([all_combos[i] for i in pick], dtype=np.int16)
            R = int(delete_sets.shape[0])
            if log is not None:
                log._log(
                    f"[jackknife] chr delete-{d}: using RANDOM {R} replicates out of "
                    f"C({U},{d})={total} (seed={spec.seed})."
                )

        D = np.zeros((R, U), dtype=np.float64)
        rr = np.arange(R, dtype=np.int64)[:, None]
        D[rr, delete_sets.astype(np.int64)] = 1.0

        return cls(
            spec=spec,
            nrep=R,
            nunit=U,
            unit_id=unit_id,
            unit_labels=unit_labels,
            delete_sets=delete_sets,
            D=D,
            starts=starts,
            ends=ends,
            nsnps=M,
        )

    def unit_sizes(self, active_mask=None, dtype=np.float64) -> np.ndarray:
        if active_mask is None:
            out = (self.ends - self.starts).astype(dtype, copy=False)
            return out

        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.ndim != 1 or active_mask.size != self.nsnps:
            raise ValueError(
                f"active_mask must be 1D with length {self.nsnps}; got {active_mask.shape}."
            )
        out = np.bincount(self.unit_id[active_mask], minlength=self.nunit).astype(dtype, copy=False)
        return out

    def summarize(
        self,
        reps,
        *,
        unit_sizes,
        axis=0,
        center="mean",
        nan_policy="omit",
    ):
        return utils._calc_jackknife_se_from_delete_sets(
            reps,
            D=self.D,
            unit_sizes=unit_sizes,
            axis=axis,
            center=center,
            nan_policy=nan_policy,
        )
