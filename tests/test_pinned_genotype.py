from contextlib import ExitStack
import io
import os
from pathlib import Path

import numpy as np
import pytest
from bed_reader import to_bed

from summit.ldscore._pinned_genotype import (
    DescriptorBEDReader, _PositionalReader, open_pinned_file,
)


def test_metadata_readers_have_independent_positions(tmp_path):
    path = tmp_path / "metadata"
    path.write_bytes(b"first\nsecond\nthird\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        with io.BufferedReader(_PositionalReader(descriptor)) as first:
            with io.BufferedReader(_PositionalReader(descriptor)) as second:
                assert first.readline() == second.readline() == b"first\n"
                assert first.read() == second.read() == b"second\nthird\n"
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        for _ in range(2):
            with open_pinned_file(Path("/dev/fd") / str(descriptor)) as handle:
                assert handle.read() == path.read_bytes()
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("order", ["F", "C"])
def test_descriptor_reader_missing_calls_subsets_and_mutation(tmp_path, order):
    raw = np.array([[0, 1, np.nan, 2], [2, 0, 1, 0], [1, 2, 0, 1]], dtype=float)
    prefix = str(tmp_path / "genotype")
    to_bed(prefix + ".bed", raw)
    with ExitStack() as stack:
        descriptors = {}
        for suffix in (".bed", ".bim", ".fam"):
            descriptor = os.open(prefix + suffix, os.O_RDONLY)
            descriptors[suffix] = descriptor
            stack.callback(os.close, descriptor)
            os.lseek(descriptor, 1, os.SEEK_SET)
        reader = DescriptorBEDReader(descriptors, raw.shape, num_threads=2)
        stack.callback(reader.close)
        rows = np.array([0, 2])
        for _ in range(2):
            observed = reader.read((rows, slice(1, 4)), order=order)
            np.testing.assert_array_equal(observed, raw[rows, 1:4])
            assert observed.flags.f_contiguous if order == "F" else observed.flags.c_contiguous
        assert all(os.lseek(fd, 0, os.SEEK_CUR) == 1 for fd in descriptors.values())
        replacement = tmp_path / "replacement.bed"
        replacement.write_bytes(Path(prefix + ".bed").read_bytes())
        os.replace(replacement, prefix + ".bed")
        with pytest.raises(RuntimeError, match="changed"):
            reader.read((rows, slice(1, 4)))


def test_portable_reference_and_scoring_match_native(tmp_path, monkeypatch):
    # Exercise the macOS descriptor route on Linux too. The existing oracle
    # compares all four LD-score directions and scores a phenotype afterward.
    from summit.ldscore import gwe_ldscore, gxe_score
    from test_gxe_native_core import test_opt_in_native_reference_matches_python_artifacts

    monkeypatch.setattr(gwe_ldscore, "descriptor_directory", lambda: Path("/dev/fd"))
    monkeypatch.setattr(gxe_score, "descriptor_directory", lambda: Path("/dev/fd"))
    test_opt_in_native_reference_matches_python_artifacts(tmp_path)
