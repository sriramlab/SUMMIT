# Frozen oracle fixtures and analytic microcases

The Q=1–4 fixtures were generated from the bundled correctness-first Python source at snapshot commit `251f197950775ca891f244dedc109476b2ad43b4`. They are synthetic, license-safe, and contain no participant data.

Each NPZ contains explicit inputs and expected fixed-probe reference, same-person, grouped-deletion, trait, transfer, and raw-fit outputs. The agent must independently recompute them with the active Python oracle before treating them as authoritative. Numerical changes require review; do not silently replace expected files.

Coverage:

- `Q=1,2,3,4`;
- `K=2`, with `Q=4,K=3`;
- strict-disjoint annotations across four deletion groups;
- explicit sample and variant Rademacher probes;
- residual rank different from sample count;
- exact dense and fixed-probe Gram moments;
- signed variant-probe same-person U-statistic;
- grouped Gram/RHS/trace/genetic-residual numerators;
- transfer for `study_N != reference_N`;
- full raw fit and every frozen group deletion;
- negative fitted off-diagonals for `Q>=2` in the deterministic construction.

Analytic microcases separately freeze projection order, directional factors, `Omega` packing, sample-count transfer, rank-deficient overlap behavior, and exact equivalence of direct grouped TN versus group-restricted actions under nontrivial overlapping annotations.

Regenerate the oracle fixtures with:

```bash
python FIXTURES/generate_fixtures.py \
  --source-root <path-to-generalized_python/src> \
  --output-dir FIXTURES
```

Then run:

```bash
python TOOLS/validate_package.py
```

The validator independently reconstructs features, dense kernels, fixed-probe actions/Gram, both grouped-attribution algorithms, the signed same-person U-statistic, and study-side trait moments from the stored inputs rather than only checking stored arrays against one another.
