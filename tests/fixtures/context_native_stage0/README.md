# Synthetic numerical fixtures

These small arrays test contextual covariance equations. They contain generated
inputs and expected results, with no participant IDs or external cohort inputs.

The Q=1–4 cases cover component ordering, fixed-effect projection, kernel Gram
matrices, same-person terms, grouped deletion, reference transfer, and fitting.
The analytic microcases isolate off-diagonal factors, projection order,
overlapping annotations, rank deficiency, and unequal study/reference sizes.

Run the independent calculations with:

```bash
python -m pytest -q tests/test_context_stage0_fixtures.py tests/test_context_stage1_dense_native.py
```

Check expected results against the equations before changing a fixture.
