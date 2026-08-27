from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from summit.context import ContextComponentIndex, ContextPairIndex


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generalized_gxe"
    / "workflow.py"
)
sys.path.insert(0, str(WORKFLOW_PATH.parent))
SPEC = importlib.util.spec_from_file_location("generalized_gxe_workflow", WORKFLOW_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKFLOW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKFLOW
SPEC.loader.exec_module(WORKFLOW)


def _axes(n: int = 8):
    return WORKFLOW.PlinkAxes(
        prefix=Path("unused"),
        sample_ids=tuple(f"sample_{index}" for index in range(n)),
        variant_ids=("v1",),
        counted_alleles=("A",),
        other_alleles=("G",),
    )


def test_row_selection_supports_disjoint_reference_and_study_rows() -> None:
    axes = _axes()
    reference = WORKFLOW._normalized_row_selection(axes, 3, [0, 2, 4])
    study = WORKFLOW._normalized_row_selection(axes, 3, [1, 3, 5])
    np.testing.assert_array_equal(reference, [0, 2, 4])
    np.testing.assert_array_equal(study, [1, 3, 5])
    assert not np.intersect1d(reference, study).size


@pytest.mark.parametrize(
    ("sample_count", "rows", "message"),
    (
        (3, None, "explicit row selection"),
        (2, [0, 1, 2], "match the input sample axis"),
        (2, [0, 8], "out-of-range"),
        (2, [1, 1], "duplicate"),
    ),
)
def test_row_selection_rejects_invalid_maps(sample_count, rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        WORKFLOW._normalized_row_selection(_axes(), sample_count, rows)


def test_rank_reduced_residual_basis_prunes_binary_square() -> None:
    age = np.linspace(-1.5, 1.5, 12)
    sex = np.tile([-1.0, 1.0], 6)
    basis = np.column_stack([np.ones(age.size), age, sex])
    residual, names, pairs = (
        WORKFLOW.rank_reduced_symmetric_context_residual_basis(
            basis, ("intercept", "age", "sex")
        )
    )
    assert residual.shape == (12, 5)
    assert np.linalg.matrix_rank(residual) == 5
    assert pairs == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))
    assert "residual_sex_x_sex" not in names


def test_restricted_genetic_fit_reuses_one_normal_equation_batch(monkeypatch) -> None:
    component_index = ContextComponentIndex(("all",), ContextPairIndex(2))
    reference = SimpleNamespace(
        component_index=component_index,
        block_labels=("block_0", "block_1"),
    )
    trait = SimpleNamespace(residual_names=("residual_0", "residual_1"))
    matrices = (
        np.diag([2.0, 3.0, 4.0, 5.0, 6.0]),
        np.diag([2.1, 3.1, 4.1, 5.1, 6.1]),
        np.diag([1.9, 2.9, 3.9, 4.9, 5.9]),
    )
    rhs = np.asarray([2.0, 6.0, 12.0, 20.0, 30.0])
    systems = tuple(SimpleNamespace(matrix=matrix, rhs=rhs) for matrix in matrices)

    def forbidden_assembly(*args, **kwargs):
        raise AssertionError("preassembled systems should be reused")

    monkeypatch.setattr(
        WORKFLOW,
        "assemble_generalized_gxe_normal_equation_batch_v1",
        forbidden_assembly,
    )
    fit = WORKFLOW.restricted_genetic_fit(
        reference,
        trait,
        0,
        genetic_pair_indices=(0, 2),
        normal_equations=systems,
    )
    assert fit["component_names"] == [
        "omega_0_0",
        "omega_0_1",
        "residual_0",
        "residual_1",
    ]
    assert fit["selected_indices"] == [0, 2, 3, 4]
    np.testing.assert_allclose(fit["coefficients"], [1.0, 3.0, 4.0, 5.0])
    assert np.asarray(fit["loo_coefficients"]).shape == (2, 4)
