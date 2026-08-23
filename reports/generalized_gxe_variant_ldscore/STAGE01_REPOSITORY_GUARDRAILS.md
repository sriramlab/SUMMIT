# Stage 01: repository guardrails

Date: 2026-08-21

Base commit: `8b369d9` (`docs: record generalized GxE baseline audit`)

Scope: documentation and source comments only

## Result

**PASS.** The repository now durably distinguishes the production generalized
per-variant G×E LD-score estimator from the existing sample-probe contextual
covariance/action estimator. No executable statement, numerical path, schema,
or output was changed.

The isolated `CODEX_LAUNCH_PROMPT.md` package-manifest mismatch remains recorded
in the Stage 00 report and was explicitly waived by the user. All Stage 01
technical and scientific checks passed; the waiver was not extended to any
other condition.

## Patch review and merge

The authoritative patch
`repository_patch/0001-record-generalized-gxe-variant-ldscore-contract.patch`
was read in full and checked with `git apply --check` before application. It
applied cleanly to the live tree. No conflict or manual semantic reconstruction
was required.

The implementation snapshot referenced by the package and live HEAD differed
only by previously audited Stage 7 reports/evidence and the Stage 00 report
commit. None overlapped the patch. Existing live documentation was therefore
preserved.

## Files changed and durable wording

| File | Guardrail visible to future developers/agents |
|---|---|
| `AGENTS.md` | Requires reading the new contract, ADR, and contextual contract; names the two estimator identities; freezes the two-pass, variant-probe, fixed-full-sum deletion and common-scale rules; defines the systems-only reuse boundary. |
| `docs/generalized_gxe_variant_ldscore_contract.md` | Records contract ID `generalized_gxe_variant_ldscore_v1`, `F_q=P diag(phi_q)G`, pair order, directional LD-score estimand, two-pass randomized construction, fixed-full-sum jackknife, and scope distinction. |
| `docs/architecture/ADR-generalized-gxe-variant-ldscore.md` | Records the accepted architectural decision, exactly-two-traversal invariant, hard source/target dependency, consequences, and rejected alternatives. |
| `docs/context_native/scientific_contract_v1.md` | Adds a leading scope guardrail that its sample-axis `Z` actions and grouped numerators are not per-variant G×E LD scores or their jackknife. |
| `docs/contextual_covariance_status.md` | Adds an estimator-identity warning against using the contextual grouped-action path for the new estimator. |
| `docs/gxe_genie.md` | Identifies the mature X/W implementation as a systems template while excluding its hard-coded four-panel layout and separate X/W scales. |
| `src/summit/context/reference_v1.py` | Adds a module-docstring identity warning: sample-axis aggregate contextual covariance/action, not generalized per-variant G×E LD score. |
| `src/native/contextual_streamed_reference_v1.inc` | Adds the matching native source comment and excludes grouped actions from the fixed-full-genome row-deletion jackknife. |
| `src/summit/ldscore/gwe_ldscore.py` | Adds the mature-template warning and excludes its fixed X/W scientific layout and separate post-projection scales. |

This report is the tenth Stage 01 file.

## Cross-links and index audit

The new top-level agent instructions link the contract, ADR, and contextual
contract. The contract links back to the contextual scientific contract; the
ADR links the two estimator identities; the contextual scientific/status docs
and mature G×E documentation link to the new contract.

No repository-wide documentation index exists. The root `README.md` is a user
guide rather than a documentation index, and there was no pre-existing
generalized per-variant G×E entry-point document to amend. The mature G×E and
contextual entry documents now provide the relevant cross-links.

## Ambiguity search

A repository search over `README.md`, `docs`, `src/summit`, and `src/native`
found no use of either of these prohibited phrases:

```text
one-pass point estimate
exact grouped jackknife
```

The repository also had no existing phrase that mislabeled a sample-probe
contextual action as a “generalized G×E LD score.” The focused patch therefore
added identity warnings without rewriting unrelated documentation.

## Validation

The following checks passed:

```text
git apply --check authoritative-patch                 PASS
required contract/ADR/contextual paths exist          PASS
variant-axis and sample-axis identity grep assertions PASS
Python and native source warning assertions           PASS
prohibited ambiguity search                           PASS (no matches)
git diff --check                                      PASS
```

No repository Markdown linter or documentation-link checker was found. Direct
path checks verified all files named by the new guardrails.

## Stop gate

Stage 01 is complete. Its changes are documentation, docstrings, and comments
only, the required identities are explicit, links resolve, and validation
passes. Stage 02 may begin after this focused stage is committed.
