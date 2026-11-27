#!/usr/bin/env bash
# SUMMIT-MIX quick starts on simulated data (about a minute): the reference from two source panels, the cohort's
# release in one command, and the analyst's compose, h2 and genetic correlation from the release and the summary
# statistics. The commands are those of the wiki page "Admixed cohorts without in-sample LD", run in a folder whose
# files carry the page's names (out/admixed_ld_release/). In-sample LD scores of the same people are computed at the
# end for comparison. The simulated cohort has no relatives; related_pairs.tsv lists a few pairs of it only so that
# the command runs as written.
set -euo pipefail

cd "$(dirname "$0")"
python ./prepare_admixed_ld_inputs.py
D=$(pwd)/out/admixed_ld
W=./out/admixed_ld_release
mkdir -p "$W"
cd "$W"
if ! command -v summit >/dev/null 2>&1; then
  summit() { env PYTHONPATH=../../../src python -m summit "$@"; }
fi

# the inputs under the page's names
mkdir -p ref maps gwas pheno
for e in bed bim fam; do
  ln -sf "$D/panel_afr.$e" "ref/afr.$e"; ln -sf "$D/panel_eur.$e" "ref/eur.$e"; ln -sf "$D/cohort.$e" "cohort.$e"
done
ln -sf "$D/cohort.cov" gwas_covariates.txt
ln -sf "$D/trait.sumstats" gwas/height.tsv; ln -sf "$D/trait2.sumstats" gwas/bmi.tsv
ln -sf "$D/trait.pheno" pheno/height.txt; ln -sf "$D/trait2.pheno" pheno/bmi.txt
awk '{print $1, $2}' cohort.fam > gwas_people.txt
awk 'BEGIN {print "IID1", "IID2"} NR % 600 == 1 {a = $2} NR % 600 == 2 {print a, $2}' cohort.fam > related_pairs.tsv
for c in 1 2 3 4 5 6; do                         # the simulation's map: 1 cM per Mb
  awk -v c="$c" 'BEGIN {print "position COMBINED_rate(cM/Mb) Genetic_Map(cM)"} $1 == c {print $4, 1.0, $3}' \
    cohort.bim > "maps/genetic_map_chr${c}_combined_b37.txt"
done

# the reference (once per set of panels; kept when the example is run again)
if [ ! -e ref/afr_eur_clean/manifest.json ]; then
  rm -rf ref/afr_eur ref/afr_eur_clean
  summit admixed-ld reference --panel ref/afr --panel ref/eur --names AFR,EUR \
    --genetic-map maps/genetic_map_chr{chr}_combined_b37.txt --threads 16 --out ref/afr_eur
  summit admixed-ld clean-panels --ref ref/afr_eur --panel ref/afr --panel ref/eur --names AFR,EUR \
    --threads 4 --out ref/afr_eur_clean
fi

# the cohort
summit admixed-ld release --geno cohort --ref ref/afr_eur_clean --keep gwas_people.txt \
  --covar gwas_covariates.txt --relative-pairs related_pairs.tsv --threads 8 --out release

# the analyst (compose and summit create the folders of their --out)
summit admixed-ld compose --ref ref/afr_eur_clean --release release --sumstats gwas/ \
  --annot release/cohort.bins8.annot.gz --out ld/cohort
summit --h2 gwas/ --ldscores ld/cohort --annot release/cohort.bins8.annot.gz --out results/h2
summit --rg gwas/height.tsv,gwas/bmi.tsv --ldscores ld/cohort --annot release/cohort.bins8.annot.gz \
  --pheno-rg pheno/height.txt,pheno/bmi.txt --pheno-rg-cov gwas_covariates.txt,gwas_covariates.txt \
  --keep-ambiguous --out results/rg

# for comparison: in-sample LD scores of the same people with the same bins and covariates, and the same fits; the
# bins leave out SNPs without variance, so the genotypes are kept to the bins' SNPs first
python - <<'PY'
import pandas as pd
from bed_reader import open_bed, to_bed
snps = set(pd.read_csv("release/cohort.bins8.annot.gz", sep="\t", usecols=["SNP"]).SNP.astype(str))
bed = open_bed("cohort.bed")
keep = [i for i, s in enumerate(bed.sid) if s in snps]
to_bed("ld/insample_geno.bed", bed.read(index=(slice(None), keep), dtype="float32"),
       properties=dict(fid=bed.fid, iid=bed.iid, chromosome=bed.chromosome[keep], sid=bed.sid[keep],
                       cm_position=bed.cm_position[keep], bp_position=bed.bp_position[keep],
                       allele_1=bed.allele_1[keep], allele_2=bed.allele_2[keep]))
PY
summit --geno ld/insample_geno.bed --annot release/cohort.bins8.annot.gz --covar gwas_covariates.txt --nvecs 200 \
  --block-size 256 --seed 1 --dtype float64 --num-threads 2 --out ld/insample
summit --h2 gwas/ --ldscores ld/insample.gw.ldscore.gz --annot release/cohort.bins8.annot.gz --out results/h2_insample
summit --rg gwas/height.tsv,gwas/bmi.tsv --ldscores ld/insample.gw.ldscore.gz --annot release/cohort.bins8.annot.gz \
  --pheno-rg pheno/height.txt,pheno/bmi.txt --pheno-rg-cov gwas_covariates.txt,gwas_covariates.txt \
  --keep-ambiguous --out results/rg_insample

echo "SUMMIT-MIX quick starts finished in $W: the release in release/, LD scores in ld/, fits in results/"
