from __future__ import annotations

import numpy as np

from scripts.gxe import benchmark_gemm_orientations as benchmark


class _FakeProtectedModule:
    @staticmethod
    def protected_matmul_nn(left, right, threads):
        assert threads == 2
        return np.asfortranarray(np.asarray(left) @ np.asarray(right)), 0

    @staticmethod
    def protected_matmul_tn(left, right, threads):
        assert threads == 2
        return np.asfortranarray(np.asarray(left).T @ np.asarray(right)), 0


def test_production_profile_contains_all_governing_block_widths():
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--profile", "production",
            "--operations", "source",
            "--orientations", "current",
            "--layouts", "column_major",
            "--max-cases", "5",
            "--dry-run",
        ]
    )
    benchmark._validate_args(parser, args)
    cases = benchmark._build_case_specs(args)
    assert tuple(case.block_width for case in cases) == (
        1024,
        2000,
        3072,
        4096,
        6144,
    )
    assert {case.n_samples for case in cases} == {benchmark.PRODUCTION_N}
    assert {case.probe_tile for case in cases} == {32}
    assert {case.environment_tile for case in cases} == {1}
    assert {case.threads for case in cases} == {32}


def test_exact_orientation_lowerings_match_sampled_dense_oracle_without_copies():
    module = _FakeProtectedModule()
    supported = 0
    for operation in ("source", "target"):
        for orientation in ("current", "transposed"):
            for layout in ("column_major", "row_major"):
                case = benchmark.CaseSpec(
                    operation=operation,
                    orientation=orientation,
                    layout=layout,
                    n_samples=31,
                    block_width=7,
                    probe_tile=2,
                    environment_tile=1,
                    threads=2,
                )
                prepared = benchmark._prepare_call(case, module, seed=9182)
                if operation == "target" and layout == "row_major":
                    assert isinstance(prepared, str)
                    assert "without an explicit O(NM) copy" in prepared
                    continue
                assert not isinstance(prepared, str)
                output, repaired = prepared.invoke()
                assert repaired == 0
                correctness = benchmark._validate_samples(prepared, output)
                assert correctness["maximum_absolute_error"] <= 1.0e-10
                assert all(array.flags.f_contiguous for _, array in prepared.operands)
                supported += 1
    assert supported == 6


def test_shape_metadata_and_timeout_guards_are_explicit():
    source = benchmark.CaseSpec(
        "source", "current", "column_major", 289_111, 2000, 32, 3, 32
    )
    target = benchmark.CaseSpec(
        "target", "current", "column_major", 289_111, 2000, 32, 3, 32
    )
    assert source.panel_columns == 192
    assert target.panel_columns == 384
    assert benchmark._requested_cblas(source) == {
        "interface": "column_major",
        "m": 289_111,
        "n": 192,
        "k": 2000,
        "transpose_a": "N",
        "transpose_b": "N",
        "lda": 289_111,
        "ldb": 2000,
        "ldc": 289_111,
        "output_layout": "column_major",
    }
    assert benchmark._requested_cblas(target) == {
        "interface": "column_major",
        "m": 2000,
        "n": 384,
        "k": 289_111,
        "transpose_a": "T",
        "transpose_b": "N",
        "lda": 289_111,
        "ldb": 289_111,
        "ldc": 2000,
        "output_layout": "column_major",
    }
    assert benchmark.MAX_CONFIGURATION_SECONDS == 90.0
    assert benchmark.MAX_SWEEP_SECONDS == 1200.0
    assert benchmark.DEFAULT_WARMUPS == 3
    assert benchmark.DEFAULT_REPEATS == 5
    assert benchmark._estimated_bytes(source)["headroom_fraction"] == 0.20
