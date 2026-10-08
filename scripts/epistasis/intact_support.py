"""Genotype-only diagnosis of nuisance leverage; no selection on outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from scripts.epistasis.intact_validation import intact_panel, coordinates


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--covariates", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw, cov, axis, meta = intact_panel(
        a.genotypes, a.covariates, "12:66358347", "5", 73451
    )
    x, d, _, _ = coordinates(raw)
    fixed = np.random.default_rng(73452).choice(len(x), 6144, replace=False)[2048:]
    f = x[:, 0] * x[:, 16:].sum(1) / 8
    records = []
    for n in (4096, 8192, 32768, 65536):
        ids = (
            fixed
            if n == 4096
            else np.random.default_rng(138971).integers(len(x), size=n)
        )
        cs = [np.ones(len(x)), cov, x[:, :16], d[:, :16]]
        for name, extra in (
            ("local", None),
            ("all_additive", x[:, 16:]),
            ("all_dominance", d[:, 16:]),
            ("PC1_local", cov[:, 5, None] * x[:, :16]),
            ("PC1_burden", cov[:, 5] * x[:, 16:].sum(1) / 8),
        ):
            if extra is not None:
                cs.append(extra)
            design = np.column_stack(cs + [f])[ids]
            u = thin_rank_revealing_fixed_effect_basis(design)
            h = (u * u).sum(1)
            records.append(
                dict(
                    n=n,
                    stage=name,
                    rank=u.shape[1],
                    max_leverage=float(h.max()),
                    PC1_at_max_leverage=float(cov[ids[np.argmax(h)], 5]),
                )
            )
    # A bounded actual-genotype counterexample to unrestricted fixed-panel
    # identification: if additive columns span R^N, they also span a product.
    ids = np.random.default_rng(392873).choice(len(x), 48, replace=False)
    additive = np.column_stack([np.ones(len(ids)), x[ids]])
    fit = np.linalg.lstsq(additive, f[ids], rcond=1e-11)
    obstruction = dict(
        n=len(ids),
        additive_columns=additive.shape[1],
        rank=int(fit[2]),
        relative_product_residual=float(
            np.linalg.norm(f[ids] - additive @ fit[0]) / np.linalg.norm(f[ids])
        ),
        implication="on this fixed panel, unrestricted additive coefficients reproduce the interaction mean; a finite mean restriction, population assumptions or additional input is necessary",
    )
    with a.out.open("x") as handle:
        json.dump(
            dict(panel=meta, records=records, fixed_panel_identification=obstruction),
            handle,
            indent=2,
        )
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
