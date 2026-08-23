# Superseding J-check and inference-boundary decision

Date: 2026-08-22

## Decision

- A changed-`J` invariance experiment is no longer an implementation or
  qualification gate.
- Reference construction still performs exactly two complete genotype passes
  and accumulates `block_directed_numerator` plus
  `block_annotation_mass` during pass 2.
- New reference construction does not form or publish a
  `deleted_genetic_gram` cube. The inference adapter constructs each requested
  delete-block Gram on demand, without genotype access, and reuses the full
  same-person matrix.
- Readers retain compatibility with older artifacts that contain a cached
  `deleted_genetic_gram`; new writers do not emit one.

The explicit changed-`J` comparison was removed from the live tests and from
the package stage prompts that required it. Historical stage reports remain
unchanged as records of what was run at the time.

## Package bookkeeping check

The authoritative package initially failed only because
`CODEX_LAUNCH_PROMPT.md` had a stale entry in `MANIFEST.sha256`. Its observed
content hash was recorded in the manifest. No estimator source, mathematical
oracle, source snapshot, or repository patch was changed to resolve that
bookkeeping mismatch.

After updating the package decisions/prompts and their manifest entries:

```text
python scripts/check_package.py
PACKAGE_CHECK_OK

python scripts/run_math_checks.py
10 passed
```

This manifest-only mismatch is not an estimator qualification issue.

## Focused pre-native verification

The pure-Python generalized suites completed with 69 passes and one expected
skip. Two tests requiring an installed native extension could not collect that
extension from the source tree; they are deferred to the fresh private-BLIS
build.
