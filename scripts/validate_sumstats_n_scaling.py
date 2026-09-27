#!/usr/bin/env python3
"""Compare corrected and uncorrected HE moments on existing GWAS summaries.

Read-only inputs; writes a compact before/after table into a new directory.
The reference structural sums are shared when all SNPs are retained. Filtered
traits use their own post-filter jackknife blocks, as in the ordinary h2 CLI.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from summit.inference.h2core import (
    compute_h2_structural_unit_stats, prepare_h2_reference_axis, fit_h2,
)
from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.trace import Trace
from summit.sumstats.sumstats import Sumstats
from summit.sumstats.moments import build_h2_summary_moment, exact_score_z_from_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ldscores', required=True)
    parser.add_argument('--annot')
    parser.add_argument('--sumstats', nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--njack', default='200')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    print('Loading reference', flush=True)
    trace = Trace(ldscores=args.ldscores, annot=args.annot)
    full_tv = trace.materialize_view()
    spec = JackknifeSpec.parse(args.njack)
    full_jk = JackknifeDesign.from_trace_view(full_tv, spec)
    full_struct = compute_h2_structural_unit_stats(full_tv, full_jk)
    print(f'Reference ready: {trace.nsnps} SNPs, {trace.nbins} annotations, '
          f'{time.monotonic()-start:.1f}s', flush=True)
    rows = []
    for path in args.sumstats:
        ss = Sumstats.from_file(path, compute_diagnostics=False)
        aligned = ss.align_to_trace(trace)
        keep = aligned.keep_mask(chisq_threshold='auto', chisq_action='drop')
        tv = full_tv if np.all(keep) else trace.materialize_view(keep)
        jk = full_jk if np.all(keep) else JackknifeDesign.from_trace_view(tv, spec)
        struct = full_struct if np.all(keep) else compute_h2_structural_unit_stats(tv, jk)
        matched = aligned.materialize(keep, compute_diagnostics=False)
        corrected, info = build_h2_summary_moment(matched)
        old = exact_score_z_from_arrays(matched.beta, matched.se, matched.n, matched.nsamp, 0)**2
        row = dict(input=str(Path(path).resolve()), n_snps=tv.nsnps,
                   n_max=matched.nsamp, n_min=float(matched.n.min()))
        for label, y in [('before', old), ('after', corrected)]:
            p = prepare_h2_reference_axis(
                tv, matched, jk, np.ones(tv.nsnps, dtype=bool),
                summary_y=y, summary_y_info=info, full_struct=struct,
            )
            fit = fit_h2(p)
            row[label+'_h2'] = float(fit.h2[-1, 0])
            row[label+'_se'] = float(fit.h2[-1, 1])
            row[label+'_max_abs_component'] = float(np.max(np.abs(fit.sigma_reps[-1, :tv.nbins])))
        rows.append(row)
        print(row, flush=True)
        del ss, aligned, matched, corrected, old, p, fit
    pd.DataFrame(rows).to_csv(args.out_dir/'comparison.tsv', sep='\t', index=False)
    print(f'Completed in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
