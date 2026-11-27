"""The cohort's release (`summit admixed-ld release`): one folder with everything that an analyst needs from the cohort.

It runs `curve` on the release's people with the GWAS covariates (the curve's pass also writes the sample statistic;
the background term and the cell matrix are included) and `bins` on the same people, then writes release.json (what
was made, from how many people and covariates, and which files to share) and README.txt (the same in plain language).
Every file it writes is an aggregate; it writes no row about a person and copies none of its inputs.
"""
from __future__ import annotations

import json
import os
import time

NAME = "cohort"
RECORD = "release.json"
README = "README.txt"
LOG = "release.log"

_WHAT = {
    "curve.tsv": "the covariance of local ancestry by genetic distance (the curve)",
    "curve.json": "the curve's parameters, the moments of global ancestry and the counts of people and SNPs",
    "background.tsv": "the background term by genetic distance (LD that pairs of people sharing segments add)",
    "cells.npz": "the cell matrix (covariances between 2-cM segments of the genome, in two halves of the people)",
    "sample_factors.json": "the sample statistic (a small covariance matrix of person factors, for the same-person "
                           "term)",
    "bins8.annot.gz": "the eight MAF/LD bins of the fit (0/1 columns per SNP)",
    "maf2.annot.gz": "the two MAF classes of the fit (MAF below 0.05 and at least 0.05; 0/1 columns per SNP)",
}


def _check_folder(out):
    """The release folder must be new, empty or a release folder (whose own files are replaced)."""
    if not os.path.exists(out):
        os.makedirs(out)
        return
    if not os.path.isdir(out):
        raise ValueError(f"--out {out} exists and is not a folder.")
    present = sorted(f for f in os.listdir(out) if f != LOG)
    if not present:
        return
    if RECORD not in present:
        raise ValueError(f"--out {out} already holds files ({', '.join(present[:5])}) and is not a release folder; "
                         "give a new or empty folder, so that the release holds only its own files.")
    with open(os.path.join(out, RECORD)) as fh:
        listed = {f["file"] for f in json.load(fh).get("files", [])} | {RECORD, README}
    other = [f for f in present if f not in listed]
    if other:
        raise ValueError(f"--out {out} holds files that its earlier release did not write ({', '.join(other[:5])}); "
                         "give a new or empty folder.")
    for f in present:
        os.remove(os.path.join(out, f))


def make_release(geno, ref, out, *, keep=None, covar=None, relative_pairs=None, kinship_col=None, kinship_min=None,
                 sources=None, match="position", cell_cm=0.25, step=4096, threads=1, drop_ambiguous=False,
                 background_memory_gb=32.0, relatedness_memory_gb=4.0, name=NAME, log=print) -> dict:
    """The release folder `out`: the curve with its sample statistic, background term and cell matrix, the MAF/LD
    bins, release.json and README.txt. Returns the record."""
    from .bins import make_bins
    from .curve import _cohort_samples, genotype_curve
    from .sample_factors import MIN_PEOPLE
    t0 = time.perf_counter()
    if covar is None:
        log("[admixed-ld] release: WARNING: no --covar; the release assumes a GWAS without covariates. Give the GWAS's "
            "covariates, principal components included, which every GWAS of the cohort must use")
    _check_folder(out)
    prefix = os.path.join(out, name)
    _, rows, C = _cohort_samples(geno, covar, keep, log)
    if rows.size < MIN_PEOPLE:
        raise ValueError(f"The release would describe {rows.size} people; it needs at least {MIN_PEOPLE}.")
    _, info = genotype_curve(geno, ref, prefix, sources=sources, covar=covar, keep=keep, match=match, cell_cm=cell_cm,
                             relative_pairs=relative_pairs, kinship_col=kinship_col, kinship_min=kinship_min,
                             step=step, threads=threads, background=True, background_memory_gb=background_memory_gb,
                             relatedness_memory_gb=relatedness_memory_gb, drop_ambiguous=drop_ambiguous,
                             sample_factors=True, log=log)
    bins = make_bins(geno, prefix, rows=rows, threads=threads, log=log)
    files = []
    for suffix, what in _WHAT.items():
        f = f"{name}.{suffix}"
        if os.path.exists(os.path.join(out, f)):
            files.append(dict(file=f, what=what))
    missing = [s for s in ("curve.tsv", "curve.json", "bins8.annot.gz", "maf2.annot.gz")
               if not os.path.exists(f"{prefix}.{s}")]
    if missing:
        raise RuntimeError(f"The release is missing {missing}.")
    if os.path.exists(os.path.join(out, LOG)):
        files.append(dict(file=LOG, what="the log of this run (commands, counts and file names; no row about a "
                                         "person)"))
    rel = info.get("relatedness") or {}
    rec = dict(format="summit-admixed-ld-release", version=1, curve=name, people=int(info["n_samples"]),
               covariate_rank=int(info["covariate_rank"]), sources=info["sources"], baseline=info["baseline"],
               n_snps_curve=int(info["n_snps"]), n_snps_bins=int(bins["n_snps"]),
               relative_pairs=int(rel["pairs"]) if "pairs" in rel else None,
               sample_statistic=bool(info.get("sample_factors")),
               reference=dict(name=os.path.basename(os.path.normpath(ref)),
                              sources=[dict(name=s.get("name"), n=s.get("n"))
                                                                for s in info["reference"]["sources"]],
                              window_kb=info["reference"]["window_kb"]),
               bins=bins, files=files,
               flags=list(info.get("flags", [])),
               never_share=["the genotypes", "the phenotypes", "the --keep, --covar and --relative-pairs files",
                            "any per-person table (ancestry estimates, principal components, relatedness)"],
               rules=dict(min_people=MIN_PEOPLE,
                          statement="A sample statistic (this release's or one made later with sample-factors) needs "
                                    f"at least {MIN_PEOPLE} people. Do not release a statistic of a part of these "
                                    f"people when that part, or the people it leaves out, number fewer than "
                                    f"{MIN_PEOPLE}, because two statistics whose samples differ by a few people "
                                    "describe those people."))
    with open(os.path.join(out, RECORD), "w") as fh:
        json.dump(rec, fh, indent=1)
    _write_readme(out, rec)
    log(f"[admixed-ld] release: wrote {out}/ ({rec['people']} people, covariate rank {rec['covariate_rank']}, "
        f"{len(files) + 2} files) in {time.perf_counter() - t0:.0f} s. Share the whole folder; {README} lists what it "
        "holds and what must never leave the cohort's environment")
    return rec


def _write_readme(out, rec):
    import textwrap
    para = lambda t, indent="": textwrap.fill(t, 100, initial_indent=indent,
                                              subsequent_indent=indent + ("  " if indent else ""))
    ref = rec["reference"]
    lines = [
        "SUMMIT-MIX release of an admixed cohort",
        "",
        para(f"Made with `summit admixed-ld release` on {rec['people']} people with the GWAS covariates (covariate "
             f"rank {rec['covariate_rank']}), sources {', '.join(rec['sources'])} (the last is the baseline), and the "
             f"reference {ref['name']} (panels "
             + ", ".join(f"{s['name']} of {s['n']} people" for s in ref["sources"]) + ")."),
        "",
        para("Every file in this folder holds aggregates over the people of the release. No file has a row about a "
             "person, and none copies the genotypes, the phenotypes or the files that list people. Share the whole "
             "folder with the analysts who compose LD scores for the cohort's GWAS, subject to the cohort's own rules "
             "on small counts (release.json records the counts of people, SNPs and related pairs)."),
        "",
        "Files to share (the whole folder):",
    ]
    lines += [para(f"{f['file']}: {f['what']}", "  ") for f in rec["files"]]
    lines += [para(f"{RECORD}: this list in machine-readable form, with the counts and the reference used", "  "),
              para(f"{README}: this file", "  "),
              "",
              "Never share:"]
    lines += [para(x, "  ") for x in rec["never_share"]]
    lines += [
        "",
        "Statistics of part of these people:",
        para(rec["rules"]["statement"], "  "),
        "",
        "The analyst composes the LD scores of every GWAS of the cohort from this folder, the same reference and the",
        "GWAS summary statistics:",
        "  summit admixed-ld compose --ref REF --release THIS_FOLDER --sumstats GWAS_DIR \\",
        f"    --annot THIS_FOLDER/{rec['curve']}.bins8.annot.gz --out ld/cohort",
        f"  summit --h2 GWAS_DIR --ldscores ld/cohort --annot THIS_FOLDER/{rec['curve']}.bins8.annot.gz "
        "--out results/h2",
    ]
    with open(os.path.join(out, README), "w") as fh:
        fh.write("\n".join(lines) + "\n")
