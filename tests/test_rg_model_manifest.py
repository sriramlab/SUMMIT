from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from summit.manifest.rg_manifest_fast import (
    _StructUnitStats,
    _load_fast_model_specs,
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


if __name__ == "__main__":
    unittest.main()
