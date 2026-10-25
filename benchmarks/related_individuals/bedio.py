"""Minimal PLINK 1 .bed reader for the benchmark harness (outside SUMMIT). Returns A1-allele counts as in PLINK's
bed encoding (00 -> 2, 10 -> 1, 11 -> 0, 01 -> missing), i.e. counts of the first .bim allele (A1)."""
import numpy as np

_LUT = np.array([2.0, np.nan, 1.0, 0.0])


def bed_codes(prefix, n, start=0, end=None):
    """Raw 2-bit codes (uint8 0..3) for SNPs [start, end), shape (end-start, n)."""
    raw = np.memmap(prefix + '.bed', dtype=np.uint8, mode='r')
    if raw[0] != 0x6C or raw[1] != 0x1B or raw[2] != 1:
        raise ValueError('not a SNP-major PLINK .bed')
    bpr = (n + 3) // 4
    m = (raw.size - 3) // bpr
    end = m if end is None else min(end, m)
    blk = np.asarray(raw[3 + start * bpr: 3 + end * bpr]).reshape(end - start, bpr)
    codes = np.empty((end - start, bpr * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (blk >> (2 * k)) & 3
    return codes[:, :n]


def read_bed(prefix, n, dtype=np.float32, start=0, end=None):
    """Sample-major dosage matrix (n x snps) with NaN for missing."""
    codes = bed_codes(prefix, n, start, end)
    return _LUT.astype(dtype)[codes].T.copy()


def n_snps(prefix, n):
    raw = np.memmap(prefix + '.bed', dtype=np.uint8, mode='r')
    return (raw.size - 3) // ((n + 3) // 4)
