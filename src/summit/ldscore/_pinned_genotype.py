"""POSIX descriptor I/O without shared file-position or pathname races."""

from contextlib import contextmanager
import io
import os
from pathlib import Path
import weakref

import numpy as np
import pandas as pd


def descriptor_directory() -> Path:
    for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
        if directory.is_dir():
            return directory
    raise RuntimeError("Pinned genotype input requires Linux or macOS descriptor access.")


class _PositionalReader(io.RawIOBase):
    """A private logical cursor, even when /dev/fd duplicates a shared offset."""

    def __init__(self, descriptor):
        super().__init__()
        self.descriptor = descriptor
        self.position = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        self._checkClosed()
        data = os.pread(self.descriptor, len(buffer), self.position)
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


@contextmanager
def open_pinned_file(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        with io.BufferedReader(_PositionalReader(descriptor)) as handle:
            yield handle
    finally:
        os.close(descriptor)


def read_metadata_table(path, **kwargs):
    with open_pinned_file(path) as handle:
        return pd.read_csv(handle, **kwargs)


class DescriptorBEDReader:
    """Small bed-reader adapter reusing SUMMIT's authenticated native decoder.

    The source owns the supplied descriptors. The native reader duplicates them
    and checks their identity and mutation state before and after each read.
    Only the selected sample-by-block calls are materialized.
    """

    def __init__(self, descriptors, shape, num_threads=None):
        self.descriptors = descriptors
        self.shape = tuple(shape)
        self.num_threads = num_threads or 1
        self._reader = None
        self._key = None
        self._finalizer = None

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
        self._reader = None
        self._key = None

    def read(self, index, dtype=np.float64, order="F", num_threads=None):
        from .. import gxeldcore

        def selected(selection, size):
            if isinstance(selection, slice):
                return np.arange(*selection.indices(size), dtype=np.int64)
            return np.ascontiguousarray(selection, dtype=np.int64)

        rows = selected(index[0], self.shape[0])
        variants = selected(index[1], self.shape[1])
        threads = int(num_threads or self.num_threads)
        key = (rows.tobytes(), threads)
        if self._reader is None or self._key != key:
            self.close()
            self._reader = gxeldcore.PredictionBEDReader(
                *(self.descriptors[suffix] for suffix in (".bed", ".bim", ".fam")),
                rows, threads, int(len(rows) * self.shape[1]),
            )
            self._finalizer = weakref.finalize(self, self._reader.close)
            self._key = key
        raw = np.empty((len(rows), len(variants)), dtype=np.int8, order="F")
        self._reader.read(variants, raw)
        result = raw.astype(dtype, order=order)
        result[raw == -127] = np.nan
        return result
