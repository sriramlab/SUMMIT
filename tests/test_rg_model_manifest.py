from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from summit.manifest.rg_manifest_fast import (
    _StructUnitStats,
    _load_fast_model_specs,
    _row_struct_correction,
    _row_struct_correction_selected,
    _select_struct_columns,
)


class FastModelManifestTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "models.tsv"
        path.write_text(text)
        return path

    def test_loads_json_and_csv_bins_with_shared_aliases(self):
        path = self._write(
            "model\tbins\taliases\n"
            'ct1\t["base","ct1"]\t["base","focal"]\n'
            "ct2\tbase,ct2\tbase,focal\n"
        )

        specs = _load_fast_model_specs(path, np.array(["base", "ct1", "ct2"]))

        self.assertEqual([spec.name for spec in specs], ["ct1", "ct2"])
        np.testing.assert_array_equal(specs[0].indices, [0, 1])
        np.testing.assert_array_equal(specs[1].indices, [0, 2])
        self.assertEqual(specs[0].aliases, ("base", "focal"))
        self.assertEqual(specs[1].aliases, ("base", "focal"))

    def test_rejects_unknown_bins(self):
        path = self._write("model\tbins\nct1\tbase,missing\n")
        with self.assertRaisesRegex(ValueError, "absent from the union trace"):
            _load_fast_model_specs(path, np.array(["base", "ct1"]))

    def test_rejects_inconsistent_output_schema(self):
        path = self._write(
            "model\tbins\taliases\n"
            "ct1\tbase,ct1\tbase,focal\n"
            "ct2\tbase,ct2\tbase,celltype\n"
        )
        with self.assertRaisesRegex(ValueError, "same ordered aliases"):
            _load_fast_model_specs(path, np.array(["base", "ct1", "ct2"]))

    def test_selects_vector_and_square_structural_terms(self):
        stats = _StructUnitStats(
            m=np.array([10.0, 20.0]),
            Ak=np.arange(8.0).reshape(2, 4),
            Ak2=np.arange(8.0, 16.0).reshape(2, 4),
            AA=np.arange(32.0).reshape(2, 4, 4),
            AL=np.arange(32.0, 64.0).reshape(2, 4, 4),
        )
        indices = np.array([0, 3])

        selected = _select_struct_columns(stats, indices)

        np.testing.assert_array_equal(selected.m, stats.m)
        np.testing.assert_array_equal(selected.Ak, stats.Ak[:, indices])
        np.testing.assert_array_equal(selected.Ak2, stats.Ak2[:, indices])
        np.testing.assert_array_equal(
            selected.AA, stats.AA[:, indices, :][:, :, indices]
        )
        np.testing.assert_array_equal(
            selected.AL, stats.AL[:, indices, :][:, :, indices]
        )

    def test_selected_sparse_correction_matches_contiguous_model(self):
        rng = np.random.default_rng(20260717)
        A = rng.normal(size=(31, 7))
        L = rng.normal(size=(31, 7))
        indices = np.array([5, 0, 3, 2])
        rows = np.array([1, 2, 8, 9, 10, 22, 30])
        unit_id = np.repeat(np.arange(4), [8, 8, 8, 7])

        expected = _row_struct_correction(
            np.ascontiguousarray(A[:, indices]),
            np.ascontiguousarray(L[:, indices]),
            rows,
            unit_id,
            4,
            len(indices),
        )
        observed = _row_struct_correction_selected(
            A, L, rows, indices, unit_id, 4
        )

        for field in ("m", "Ak", "Ak2", "AA", "AL"):
            np.testing.assert_array_equal(
                getattr(observed, field), getattr(expected, field)
            )


if __name__ == "__main__":
    unittest.main()
