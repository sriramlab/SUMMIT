# Stage 0 fixture validation log

Validation date: 2026-08-20. Frozen package source commit:
`251f197950775ca891f244dedc109476b2ad43b4`. Active SUMMIT commit at
validation: the same commit, on the new Stage 0 branch.

## Import integrity

The nine NPZ files, `README.md`, and `fixtures_manifest.json` were copied
without regeneration from
`/home/bronsonj/SUMMIT_generalized_gxe_package/FIXTURES`. Every NPZ SHA-256
matches `fixtures_manifest.json`:

| File | SHA-256 |
|---|---|
| `fixture_q1_k2.npz` | `e146a8d9c1f63dc51a34dd409baed669217f0d38c5f271e08dc8ddb025cde96a` |
| `fixture_q2_k2.npz` | `735f8123458f02bc7e1e66dc51f4a2de64e346ba948e7d7e1ac899ecdca655f1` |
| `fixture_q3_k2.npz` | `339a8f78a341cc6124c8b457a5bd5e49fdf25dc3937c71179440648b6539d1c5` |
| `fixture_q4_k3.npz` | `49153e514bb00003c15a713ab35b812b10a89203588261701534a10d85ffa997` |
| `micro_projection_order.npz` | `6e52f9c4a0c7f65835823c5a70b23eb3a9216402a60915b2fdce0b37d87b66f5` |
| `micro_directional_factors.npz` | `415f6170b495862ec3fc2ac07174aea334e845e476f8562d9373f1bfe2ef284c` |
| `micro_transfer_n_vs_r.npz` | `4700032104b95bf5c5b9a579e93fc27adf82c1eca8aafc45cded9f103fedabc8` |
| `micro_rank_deficient_overlap.npz` | `d5400c55f5ed10f0aba94df46f86ad4102a45cfc6a2a925be0271d700b0f8daf` |
| `micro_overlap_grouped_algorithms.npz` | `b745a3c6c3fdf3a2375f31cf7a2ea5fa0f1ca0c0d298bd71f6ef663360a6f394` |

Verification command:

```bash
sha256sum tests/fixtures/context_native_stage0/*.npz
```

## Package validator

```bash
cd /home/bronsonj/SUMMIT_generalized_gxe_package
/home/bronsonj/anaconda3/envs/summit/bin/python TOOLS/validate_package.py
```

Result: `PASS`; 61 package-manifest files, 35 independent checks for each of
Q1--Q4, and all five analytic microcases passed. The validator's largest
reconstruction discrepancy was `1.1641532182693481e-09` (Q3).

## Independent active-oracle regeneration

Fixtures were independently regenerated into a new temporary directory, not
over the imported corpus:

```bash
cd /home/bronsonj/SUMMIT_generalized_gxe_package
/home/bronsonj/anaconda3/envs/summit/bin/python FIXTURES/generate_fixtures.py \
  --source-root /home/bronsonj/SUMMIT/src \
  --output-dir /tmp/summit-stage0-fixtures.zS3IB7
```

Every key, shape, and non-floating array matched. Floating-array maxima against
the imported files were:

| Fixture | Maximum absolute difference | Array attaining maximum |
|---|---:|---|
| Q1 K2 | `7.275957614183426e-12` | grouped Gram numerator |
| Q2 K2 | `1.4551915228366852e-11` | grouped Gram numerator |
| Q3 K2 | `4.656612873077393e-10` | grouped Gram numerator |
| Q4 K3 | `1.7462298274040222e-10` | grouped Gram numerator |
| Projection/directional/transfer/rank microcases | `0` | exact |
| Overlap grouped-algorithm microcase | `4.547473508864641e-13` | group-restricted numerator |

The reviewed overall difference is below the frozen conformance tolerance
`atol=1e-9, rtol=1e-11`. It is confined to floating reduction order; no
scientific convention, map, rank, sign, normalization, or fixture was changed.
The committed tests independently call the active oracle rather than trusting
only the package generator or internal fixture equalities.
