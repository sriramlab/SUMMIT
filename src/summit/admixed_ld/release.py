"""The cohort's release (`summit admixed-ld release`): one folder with everything that an analyst needs from the cohort.

It runs `bins` and `curve` on the release's people with the GWAS covariates (the curve's pass also writes the sample
statistic; the background term and the cell matrix are included), then writes release.json (what was made, from how
many people and covariates, and which files to share) and README.txt (the same in plain language). Every file it
writes is an aggregate; it writes no row about a person and copies none of its inputs.

The inputs are checked before anything is written. The release is built in a temporary folder next to --out and moved
into place only when it is complete, so an earlier release at --out stays as it was until the new one replaces it; a
run that fails leaves its partial folder and log at OUT.failed and nothing at --out.
"""
from __future__ import annotations

import json
import os
import shutil
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


def check_release_target(out):
    """--out must be new, an empty folder or an earlier release (replaced once the new one is complete). Raises
    before anything is created."""
    out = os.path.normpath(out)
    if not os.path.lexists(out):
        return
    if os.path.islink(out) or not os.path.isdir(out):
        raise ValueError(f"--out {out} exists and is not a folder; give a new folder for the release.")
    present = sorted(os.listdir(out))
    if not present:
        return
    if RECORD not in present:
        raise ValueError(f"--out {out} already holds files ({', '.join(present[:5])}) and is not a release folder; "
                         "give a new or empty folder, so that the release holds only its own files.")
    with open(os.path.join(out, RECORD)) as fh:
        listed = {f["file"] for f in json.load(fh).get("files", [])} | {RECORD, README, LOG}
    other = [f for f in present if f not in listed]
    if other:
        raise ValueError(f"--out {out} holds files that its earlier release did not write ({', '.join(other[:5])}); "
                         "give a new or empty folder.")


def check_release_inputs(geno, ref, out, *, keep=None, covar=None, relative_pairs=None, sources=None):
    """Every input of a release, checked before anything is written: --out (check_release_target), the people (at
    least 50 with every covariate), the reference and its sources, and the related pairs' file. Returns the
    sample rows."""
    from .curve import _cohort_samples
    from .reference import Reference
    from .sample_factors import MIN_PEOPLE
    check_release_target(out)
    ref_ = Reference(ref)
    src = list(sources or ref_.names[:2])
    if len(src) < 2 or len(set(src)) != len(src):
        raise ValueError("The release needs two or more distinct sources.")
    for s in src:
        ref_.source_index(s)
    if relative_pairs is not None and not os.path.isfile(relative_pairs):
        raise ValueError(f"--relative-pairs {relative_pairs} does not exist.")
    _, rows, _ = _cohort_samples(geno, covar, keep, lambda *a: None)
    if rows.size < MIN_PEOPLE:
        raise ValueError(f"The release would describe {rows.size} people (in --keep, with every covariate); it needs "
                         f"at least {MIN_PEOPLE}.")
    return rows


def make_release(geno, ref, out, *, keep=None, covar=None, relative_pairs=None, kinship_col=None, kinship_min=None,
                 sources=None, match="position", cell_cm=0.25, step=4096, threads=1, drop_ambiguous=False,
                 background_memory_gb=32.0, relatedness_memory_gb=4.0, name=NAME, log=print, attach_log=None) -> dict:
    """The release folder `out`: the MAF/LD bins, the curve with its sample statistic, background term and cell
    matrix, release.json and README.txt. attach_log(path) starts the run's log file (release.log) inside the folder
    being built, once the inputs have passed their checks. Returns the record."""
    from .bins import make_bins
    from .curve import genotype_curve
    from .sample_factors import MIN_PEOPLE
    t0 = time.perf_counter()
    out = os.path.normpath(out)
    rows = check_release_inputs(geno, ref, out, keep=keep, covar=covar, relative_pairs=relative_pairs,
                                sources=sources)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    work = os.path.join(parent, f".{os.path.basename(out)}.building.{os.getpid()}")
    if os.path.lexists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    if attach_log is not None:
        attach_log(os.path.join(work, LOG))
    if covar is None:
        log("[admixed-ld] release: WARNING: no --covar; the release assumes a GWAS without covariates. Give the GWAS's "
            "covariates, principal components included, which every GWAS of the cohort must use")
    log(f"[admixed-ld] release: inputs checked ({rows.size} people); building the release in {work}, moved to {out} "
        "when complete")
    prefix = os.path.join(work, name)
    try:
        # the bins first: they are quick and stop a release whose SNP set cannot be binned before the curve's pass
        bins = make_bins(geno, prefix, rows=rows, threads=threads, log=log)
        _, info = genotype_curve(geno, ref, prefix, sources=sources, covar=covar, keep=keep, match=match,
                                 cell_cm=cell_cm, relative_pairs=relative_pairs, kinship_col=kinship_col,
                                 kinship_min=kinship_min, step=step, threads=threads, background=True,
                                 background_memory_gb=background_memory_gb,
                                 relatedness_memory_gb=relatedness_memory_gb, drop_ambiguous=drop_ambiguous,
                                 sample_factors=True, log=log)
        files = [dict(file=f"{name}.{suffix}", what=what) for suffix, what in _WHAT.items()
                 if os.path.exists(f"{prefix}.{suffix}")]
        missing = [s for s in ("curve.tsv", "curve.json", "bins8.annot.gz", "maf2.annot.gz")
                   if not os.path.exists(f"{prefix}.{s}")]
        if missing:
            raise RuntimeError(f"The release is missing {missing}.")
        if os.path.exists(os.path.join(work, LOG)):
            files.append(dict(file=LOG, what="the log of this run (commands, counts and file names; no row about a "
                                             "person)"))
        rel = info.get("relatedness") or {}
        no_stat = next((f.get("detail") for f in info.get("flags", []) if f.get("flag") == "sample_factors_not_made"),
                       None)
        rec = dict(format="summit-admixed-ld-release", version=1, curve=name, people=int(info["n_samples"]),
                   covariate_rank=int(info["covariate_rank"]), sources=info["sources"], baseline=info["baseline"],
                   n_snps_curve=int(info["n_snps"]), n_snps_bins=int(bins["n_snps"]),
                   relative_pairs=int(rel["pairs"]) if "pairs" in rel else None,
                   sample_statistic=bool(info.get("sample_factors")), sample_statistic_missing=no_stat,
                   reference=dict(name=os.path.basename(os.path.normpath(ref)),
                                  sources=[dict(name=s.get("name"), n=s.get("n"))
                                           for s in info["reference"]["sources"]],
                                  window_kb=info["reference"]["window_kb"]),
                   bins=bins, files=files, flags=list(info.get("flags", [])),
                   never_share=["the genotypes", "the phenotypes", "the --keep, --covar and --relative-pairs files",
                                "any per-person table (ancestry estimates, principal components, relatedness)"],
                   rules=dict(min_people=MIN_PEOPLE, statement=_rule(MIN_PEOPLE)))
        with open(os.path.join(work, RECORD), "w") as fh:
            json.dump(rec, fh, indent=1)
        _write_readme(work, rec)
    except BaseException:
        failed = f"{out}.failed"
        if os.path.lexists(failed):
            shutil.rmtree(failed)
        os.replace(work, failed)
        log(f"[admixed-ld] release: stopped; {out} is unchanged, and the partial folder and its log are in {failed}")
        raise
    # the new release is complete: it takes the place of an earlier one at out
    old = None
    if os.path.lexists(out):
        old = os.path.join(parent, f".{os.path.basename(out)}.replaced.{os.getpid()}")
        os.replace(out, old)
    os.replace(work, out)
    if old is not None:
        shutil.rmtree(old)
    if os.path.lexists(f"{out}.failed"):
        shutil.rmtree(f"{out}.failed")
    log(f"[admixed-ld] release: wrote {out}/ ({rec['people']} people, covariate rank {rec['covariate_rank']}, "
        f"{len(files) + 2} files) in {time.perf_counter() - t0:.0f} s. Share the whole folder; {README} lists what it "
        "holds and what must never leave the cohort's environment")
    return rec


def _rule(n):
    return (f"Every released statistic and set of bins describes at least {n} people. Two of them whose samples differ "
            f"by fewer than {n} people, for example the release and the statistic of one trait's people, or the "
            "statistics of two traits, would together describe the people who make the difference. The cohort "
            "therefore releases a further statistic only when its sample differs from every other released sample by "
            f"at least {n} people.")


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
    ]
    if not rec["sample_statistic"]:
        lines += ["", para("This release has no sample statistic (" + str(rec.get("sample_statistic_missing") or
                                                                         "it could not be made") + "). Compose "
                           "therefore takes the same-person term under random mating.")]
    empty = rec["bins"].get("empty_left_out") or []
    if empty:
        lines += ["", para("The annotations leave out bins and classes without any SNP of this SNP set ("
                           + ", ".join(empty) + "), for example the MAF class below 0.05 of SNPs filtered on MAF.")]
    lines += ["", "Files to share (the whole folder):"]
    lines += [para(f"{f['file']}: {f['what']}", "  ") for f in rec["files"]]
    lines += [para(f"{RECORD}: this list in machine-readable form, with the counts and the reference used", "  "),
              para(f"{README}: this file", "  "),
              "",
              "Never share:"]
    lines += [para(x, "  ") for x in rec["never_share"]]
    lines += [
        "",
        "Releasing more statistics:",
        para(rec["rules"]["statement"], "  "),
        "",
        para("The analyst composes the LD scores of every GWAS of the cohort from this folder, the same reference and "
             "the GWAS summary statistics:"),
        "  summit admixed-ld compose --ref REF --release THIS_FOLDER --sumstats GWAS_DIR \\",
        f"    --annot THIS_FOLDER/{rec['curve']}.bins8.annot.gz --out ld/cohort",
        "  summit --h2 GWAS_DIR --ldscores ld/cohort \\",
        f"    --annot THIS_FOLDER/{rec['curve']}.bins8.annot.gz --out results/h2",
    ]
    with open(os.path.join(out, README), "w") as fh:
        fh.write("\n".join(lines) + "\n")
