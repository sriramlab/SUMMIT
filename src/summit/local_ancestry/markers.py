"""Local-ancestry markers: ancestry dosages on a grid of positions, written as a PLINK fileset.

Local-ancestry calls enter SUMMIT as one more set of markers. Each marker is the number of haplotypes (0, 1 or 2) of
one ancestry at one position, with positions spaced along each chromosome. A GWAS of these markers is an
admixture-mapping scan, and their LD scores with the SNPs separate SNP heritability from heritability tagged only by
local ancestry. The same calls give each individual's global ancestry, which must be among the covariates of the GWAS
and the LD scores.

Supported calls:
  lanc  admix-kit .lanc: a header "n_sites n_samples", then one line per sample of tokens "stop:ab", meaning that the
        two haplotypes have ancestries a and b on sites [previous stop, stop). Sites come from the matching .pvar or
        .bim, samples from --samples or the matching .psam.
  msp   RFMix 2 .msp.tsv: one row per window (chm, spos, epos, sgpos, egpos, n snps, then one column per haplotype).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from bed_reader import to_bed

CHROMOSOMES = range(1, 23)


def _read_sites(path: str) -> np.ndarray:
    """Base-pair positions of the calling sites, from a .pvar or .bim file."""
    if path.endswith(".bim"):
        return pd.read_csv(path, sep=r"\s+", header=None, usecols=[3]).iloc[:, 0].to_numpy(np.int64)
    pv = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[1])
    return pv.iloc[:, 0].to_numpy(np.int64)


def _read_psam_ids(path: str) -> np.ndarray:
    ps = pd.read_csv(path, sep="\t", dtype=str)
    col = "IID" if "IID" in ps.columns else "#IID"
    return ps[col].to_numpy()


def read_lanc(path: str, n_sites: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per sample: segment ends (exclusive site indices) and the two haplotypes' ancestries (segments x 2)."""
    out = []
    with open(path) as fh:
        n, n_samples = map(int, fh.readline().split())
        if n != n_sites:
            raise ValueError(f"{path} has {n} sites but the site file has {n_sites}.")
        for line in fh:
            stops, anc = [], []
            for tok in line.split():
                stop, ab = tok.split(":")
                stops.append(int(stop))
                anc.append((int(ab[0]), int(ab[1])))
            if stops[-1] != n_sites:
                raise ValueError(f"{path}: a sample's last segment ends at site {stops[-1]}, not {n_sites}.")
            out.append((np.array(stops), np.array(anc, dtype=np.int8)))
    if len(out) != n_samples:
        raise ValueError(f"{path} lists {len(out)} samples but its header says {n_samples}.")
    return out


def _lanc_chromosome(calls: str, sites: str, samples: str | None, spacing: int, code: int):
    bp = _read_sites(sites)
    segs = read_lanc(calls, len(bp))
    ids = (pd.read_csv(samples, header=None, dtype=str).iloc[:, 0].to_numpy() if samples
           else _read_psam_ids(os.path.splitext(sites)[0] + ".psam"))
    if len(ids) != len(segs):
        raise ValueError(f"{calls}: {len(segs)} samples in the calls, {len(ids)} sample IDs.")
    # one calling site at or after each grid position
    grid = np.arange(bp[0], bp[-1] + 1, spacing)
    idx = np.unique(np.clip(np.searchsorted(bp, grid), 0, len(bp) - 1))
    dosage = np.empty((len(segs), idx.size), dtype=np.int8)
    count = np.empty(len(segs))
    for i, (stops, anc) in enumerate(segs):
        hit = (anc == code).sum(axis=1)
        dosage[i] = hit[np.searchsorted(stops, idx, side="right")]
        count[i] = (np.diff(np.r_[0, stops]) * hit).sum()
    return ids, bp[idx], dosage, count, 2.0 * len(bp)


def _msp_chromosome(calls: str, spacing: int, code: int):
    with open(calls) as fh:
        header = None
        for line in fh:
            if line.startswith("#chm"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                break
    if header is None:
        raise ValueError(f"{calls} has no '#chm' header line.")
    msp = pd.read_csv(calls, sep="\t", comment="#", header=None, names=header)
    hap_cols = header[6:]
    if len(hap_cols) % 2:
        raise ValueError(f"{calls}: odd number of haplotype columns.")
    ids = np.array([c.rsplit(".", 1)[0] for c in hap_cols[0::2]])
    hap = msp[hap_cols].to_numpy(np.int8)
    hit = (hap[:, 0::2] == code).astype(np.int8) + (hap[:, 1::2] == code)      # windows x samples
    spos, epos = msp["spos"].to_numpy(np.int64), msp["epos"].to_numpy(np.int64)
    grid = np.arange(spos[0], epos[-1] + 1, spacing)
    win = np.clip(np.searchsorted(epos, grid), 0, len(epos) - 1)              # first window ending at or after
    nsnp = msp[header[5]].to_numpy(np.float64)
    count = (nsnp[:, None] * hit).sum(axis=0)
    return ids, grid, hit[win].T.copy(), count, 2.0 * nsnp.sum()


def build_markers(calls: str, fam: str, out: str, *, fmt: str, code: int, sites: str | None = None,
                  samples: str | None = None, spacing: int = 250_000, fill: int | None = None, log=print) -> dict:
    """Write OUT.bed/.bim/.fam (ancestry-`code` dosages, individuals in FAM order) and OUT.theta.tsv.

    `calls` and `sites` contain "{chr}". Individuals of FAM without calls are an error unless `fill` gives them that
    ancestry on both haplotypes (for example unadmixed participants analyzed with admixed ones)."""
    if "{chr}" not in calls or (fmt == "lanc" and (sites is None or "{chr}" not in sites)):
        raise ValueError("--calls (and --sites for lanc) must contain '{chr}'.")
    famdf = pd.read_csv(fam, sep=r"\s+", header=None, dtype=str)
    iid = famdf[1].to_numpy()
    blocks, bim, called, total = [], [], np.zeros(len(iid)), 0.0
    order = None
    for c in CHROMOSOMES:
        if fmt == "lanc":
            ids, pos, dos, cnt, tot = _lanc_chromosome(calls.format(chr=c), sites.format(chr=c), samples, spacing, code)
        elif fmt == "msp":
            ids, pos, dos, cnt, tot = _msp_chromosome(calls.format(chr=c), spacing, code)
        else:
            raise ValueError(f"unknown format {fmt}")
        if order is None:
            first_ids = ids
            order = pd.Index(ids).get_indexer(iid)
            missing = int((order < 0).sum())
            if missing and fill is None:
                raise ValueError(f"{missing} individuals of {fam} have no calls (use --fill to give them one ancestry).")
            log(f"[local-ancestry] {len(iid) - missing} of {len(iid)} individuals have calls")
        elif not np.array_equal(ids, first_ids):
            raise ValueError(f"chr{c}: the calls list different samples, or a different order, than chr1.")
        have = order >= 0
        d = np.full((len(iid), pos.size), 2 if fill == code else 0, dtype=np.int8)
        d[have] = dos[order[have]]
        blocks.append(d)
        called += np.where(have, cnt[np.maximum(order, 0)], tot if fill == code else 0.0)
        total += tot
        bim += [(str(c), f"LA:{c}:{p}", 0.0, int(p), "A", "B") for p in pos]
        log(f"[local-ancestry] chr{c}: {pos.size} markers")
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    b = pd.DataFrame(bim, columns=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    to_bed(out + ".bed", X, count_A1=True, properties={
        "fid": famdf[0].to_numpy(), "iid": iid, "father": famdf[2].to_numpy(), "mother": famdf[3].to_numpy(),
        "sex": famdf[4].astype(int).to_numpy(), "pheno": famdf[5].to_numpy(),
        "chromosome": b.CHR.to_numpy(), "sid": b.SNP.to_numpy(), "cm_position": b.CM.to_numpy(),
        "bp_position": b.BP.to_numpy(), "allele_1": b.A1.to_numpy(), "allele_2": b.A2.to_numpy()})
    theta = called / total
    pd.DataFrame({"FID": famdf[0], "IID": iid, "THETA": theta}).to_csv(out + ".theta.tsv", sep="\t", index=False,
                                                                       float_format="%.6g")
    summary = dict(individuals=len(iid), markers=int(X.shape[1]), mean_theta=float(theta.mean()))
    log(f"[local-ancestry] {summary['markers']} markers; mean global ancestry {summary['mean_theta']:.4f}")
    return summary


def merge_markers(geno: str, markers: str, annot: str, out: str, log=print) -> dict:
    """Interleave SNPs and local-ancestry markers by position (SNPs first at equal positions) into OUT.bed/.bim/.fam,
    and write OUT.annot.tsv.gz: the classes of `annot` for SNPs and a class LA for the markers.

    The two filesets must hold the same individuals in the same order. Their .bed files are SNP-major with the same
    number of bytes per marker, so markers are copied as byte blocks; keeping them in genomic order keeps SUMMIT's
    contiguous jackknife blocks genomic."""
    fs = pd.read_csv(geno + ".fam", sep=r"\s+", header=None, dtype=str)
    fl = pd.read_csv(markers + ".fam", sep=r"\s+", header=None, dtype=str)
    if not np.array_equal(fs[1].to_numpy(), fl[1].to_numpy()):
        raise ValueError("The SNP and local-ancestry filesets must list the same individuals in the same order.")
    nbyte = (len(fs) + 3) // 4
    bs = pd.read_csv(geno + ".bim", sep=r"\s+", header=None, dtype={1: str})
    bl = pd.read_csv(markers + ".bim", sep=r"\s+", header=None, dtype={1: str})
    bs["kind"], bl["kind"] = 0, 1
    bs["src"], bl["src"] = np.arange(len(bs)), np.arange(len(bl))
    allb = pd.concat([bs, bl], ignore_index=True)
    allb["chr"] = pd.to_numeric(allb[0], errors="raise")
    allb = allb.sort_values(["chr", 3, "kind", "src"], kind="mergesort").reset_index(drop=True)
    if allb[1].duplicated().any():
        raise ValueError("Marker IDs are not unique across the two filesets.")
    src = {}
    for name, prefix, n in (("snp", geno, len(bs)), ("la", markers, len(bl))):
        m = np.memmap(prefix + ".bed", dtype=np.uint8, mode="r")
        if tuple(m[:3]) != (0x6C, 0x1B, 0x01) or m.size != 3 + n * nbyte:
            raise ValueError(f"{prefix}.bed is not a SNP-major PLINK 1 file matching its .bim and .fam.")
        src[name] = m[3:].reshape(n, nbyte)
    dest = np.memmap(out + ".bed", dtype=np.uint8, mode="w+", shape=(3 + len(allb) * nbyte,))
    dest[:3] = (0x6C, 0x1B, 0x01)
    rows = dest[3:].reshape(len(allb), nbyte)
    kind, idx = allb["kind"].to_numpy(), allb["src"].to_numpy()
    step = 50_000
    for s in range(0, len(allb), step):
        k, i = kind[s:s + step], idx[s:s + step]
        blk = np.empty((k.size, nbyte), np.uint8)
        blk[k == 0] = src["snp"][i[k == 0]]
        blk[k == 1] = src["la"][i[k == 1]]
        rows[s:s + k.size] = blk
    dest.flush()
    allb[[0, 1, 2, 3, 4, 5]].to_csv(out + ".bim", sep="\t", header=False, index=False)
    fs.to_csv(out + ".fam", sep=" ", header=False, index=False)

    an = pd.read_csv(annot, sep=r"\s+", dtype={"SNP": str})
    classes = [c for c in an.columns if c not in ("CHR", "BP", "SNP", "CM")]
    pos = pd.Index(an["SNP"]).get_indexer(bs[1])
    if (pos < 0).any():
        raise ValueError(f"{(pos < 0).sum()} SNPs of {geno}.bim are missing from {annot}.")
    Z = np.zeros((len(allb), len(classes)), dtype=int)
    Z[kind == 0] = an[classes].to_numpy()[pos[idx[kind == 0]]]
    res = pd.DataFrame({"CHR": allb[0].to_numpy(), "BP": allb[3].to_numpy(), "SNP": allb[1].to_numpy(), "CM": 0})
    for j, c in enumerate(classes):
        res[c] = Z[:, j]
    res["LA"] = kind
    res.to_csv(out + ".annot.tsv.gz", sep="\t", index=False, compression="gzip")
    summary = dict(snps=len(bs), markers=len(bl), individuals=len(fs))
    log(f"[local-ancestry] {summary['snps']} SNPs + {summary['markers']} local-ancestry markers -> {out}.bed; "
        f"annotation {out}.annot.tsv.gz ({len(classes)} classes + LA)")
    return summary
