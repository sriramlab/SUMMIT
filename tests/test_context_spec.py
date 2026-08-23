from __future__ import annotations

import copy

import numpy as np
import pytest

from summit.context.spec import (
    CONTEXT_REFERENCE_KIND,
    BasisColumnSpec,
    ContextBasisSpec,
    ContextComponentIndex,
    ContextPairIndex,
    canonical_json,
    canonical_sha256,
    validate_context_manifest,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_pair_index_is_diagonal_first_round_trippable_and_hash_stable(q: int) -> None:
    index = ContextPairIndex(q)
    expected = [(i, i) for i in range(q)] + [
        (i, j) for i in range(q - 1) for j in range(i + 1, q)
    ]
    assert [(entry.q, entry.r) for entry in index.entries] == expected
    assert [entry.kernel_factor for entry in index.entries] == [
        1 if i == j else 2 for i, j in expected
    ]
    assert len(index) == q * (q + 1) // 2
    for entry in index.entries:
        assert index.index(entry.q, entry.r) == entry.index
        assert index.index(entry.r, entry.q) == entry.index
        assert index.pair(entry.index) == entry
    assert index.digest == ContextPairIndex(q).digest
    with pytest.raises(ValueError, match="out of range"):
        index.pair(len(index))
    with pytest.raises(ValueError, match="Invalid basis pair"):
        index.index(-1, 0)


def test_component_index_is_annotation_major_and_stores_factor_once() -> None:
    components = ContextComponentIndex(("left", "right"), ContextPairIndex(3))
    assert components.names[:6] == (
        "context:left:0,0",
        "context:left:1,1",
        "context:left:2,2",
        "context:left:0,1",
        "context:left:0,2",
        "context:left:1,2",
    )
    assert components.names[6] == "context:right:0,0"
    assert components.entries[3].kernel_factor == 2
    assert components.to_dict()["entries"][3]["kernel_factor"] == 2
    assert (
        components.digest
        == ContextComponentIndex(("left", "right"), ContextPairIndex(3)).digest
    )
    with pytest.raises(ValueError, match="unique"):
        ContextComponentIndex(("left", "left"), ContextPairIndex(2))


def test_basis_spec_evaluates_supported_small_schema_and_round_trips() -> None:
    spec = ContextBasisSpec(
        basis_id="mixed_basis_v1",
        columns=(
            BasisColumnSpec("constant", "constant", include_fixed_effect=False),
            BasisColumnSpec("age", "linear", "age"),
            BasisColumnSpec(
                "age_z",
                "standardized",
                "age",
                parameters=(("center", 50.0), ("scale", 10.0)),
            ),
            BasisColumnSpec("age_sq", "polynomial", "age", parameters=(("power", 2),)),
            BasisColumnSpec(
                "group_b", "one_hot", "group", parameters=(("category", "b"),)
            ),
            BasisColumnSpec("external", "precomputed", "external"),
        ),
    )
    data = {
        "age": np.array([40.0, 50.0, 60.0]),
        "group": np.array(["a", "b", "b"], dtype=object),
        "external": np.array([-1.0, 0.0, 2.0]),
    }
    observed = spec.evaluate(data)
    expected = np.column_stack(
        [
            np.ones(3),
            data["age"],
            [-1.0, 0.0, 1.0],
            data["age"] ** 2,
            [0.0, 1.0, 1.0],
            data["external"],
        ]
    )
    np.testing.assert_array_equal(observed, expected)
    reparsed = ContextBasisSpec.from_dict(spec.to_dict())
    assert reparsed == spec
    assert reparsed.digest == spec.digest
    assert canonical_json(spec.to_dict()) == canonical_json(reparsed.to_dict())


def test_basis_spec_rejects_ambiguous_or_unfixed_definitions() -> None:
    with pytest.raises(ValueError, match="fixed center and scale"):
        BasisColumnSpec("age_z", "standardized", "age")
    with pytest.raises(ValueError, match="positive integer"):
        BasisColumnSpec("bad_power", "polynomial", "age", parameters=(("power", 0),))
    with pytest.raises(ValueError, match="unique"):
        ContextBasisSpec(
            "duplicates",
            (
                BasisColumnSpec("x", "linear", "x"),
                BasisColumnSpec("x", "linear", "x"),
            ),
        )
    with pytest.raises(ValueError, match="raw_projected"):
        ContextBasisSpec(
            "wrong_scale",
            (BasisColumnSpec("constant", "constant"),),
            feature_mode="standardized",
        )
    spec = ContextBasisSpec("finite", (BasisColumnSpec("x", "precomputed", "x"),))
    with pytest.raises(ValueError, match="non-finite"):
        spec.evaluate({"x": [1.0, np.nan]})


def test_private_manifest_validation_fails_closed_on_hash_or_identity_mismatch() -> (
    None
):
    payload = {
        "kind": CONTEXT_REFERENCE_KIND,
        "schema_version": 1,
        "feature_mode": "raw_projected",
        "basis_hash": _sha("basis"),
        "fixed_effect_hash": _sha("fixed"),
        "variant_hash": _sha("variant"),
        "annotation_hash": _sha("annotation"),
        "component_index_hash": _sha("component"),
        "loo_grouping_hash": None,
        "dimensions": {
            "n_samples": 20,
            "residual_rank": 17,
            "n_variants": 15,
            "q": 2,
            "k": 1,
        },
    }
    assert (
        validate_context_manifest(
            payload,
            expected_kind=CONTEXT_REFERENCE_KIND,
            expected={"basis_hash": payload["basis_hash"]},
        )
        == payload
    )
    bad = copy.deepcopy(payload)
    bad["basis_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_context_manifest(bad)
    with pytest.raises(ValueError, match="mismatch"):
        validate_context_manifest(payload, expected={"variant_hash": _sha("other")})
    noncanonical = copy.deepcopy(payload)
    noncanonical["diagnostic"] = np.nan
    with pytest.raises(ValueError, match="canonical JSON"):
        validate_context_manifest(noncanonical)
