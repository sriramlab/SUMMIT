"""Additional prespecified targets sharing a frozen full-marker main architecture.

Only the new local and interaction-background markers need genotype reads.
Dense, sparse, dominance and structure effects retain the parent coefficients
and reference units; they are not renormalized on the new target's sample mask.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.epistasis.full_matched import write_json
from scripts.epistasis.benchmark_robust_workflow import io
from summit.prediction.genotype import FileGenotypeSource, StandardizedBlock, native_module, estimate_scale
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.cli import _aligned_table
from summit.epistasis.trans import select_trans_background


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.chmod(0o700)
    begin, cpu, before = time.perf_counter(), time.process_time(), io()
    parent = json.loads((a.reference / "reference.json").read_text())
    with np.load(a.reference / "reference.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    native = native_module()
    configure_prediction_threads(native, a.num_threads)
    with FileGenotypeSource(parent["genotypes"], genome_build="GRCh37") as source:
        if source.identity != parent["source_identity"]:
            raise ValueError("parent genotype source changed")
        target = source.variants.ids.index(a.target)
        source.prepare(data["rows"], 128, a.num_threads)
        complete = source.read(np.array([target]))[:, 0] != -127
        rows = data["rows"][complete]
        chrom, pos = np.asarray(source.variants.chromosome), np.asarray(source.variants.position)
        local = np.flatnonzero((chrom == chrom[target]) & (abs(pos-pos[target]) <= parent["local_radius_bp"]))
        background, background_record = select_trans_background(source, a.target,
            chromosome=getattr(a, "background_chromosome", None),
            variants=getattr(a, "interaction_variants", None))
        if len(background) < 2:
            raise ValueError("need at least two prespecified trans variants")
        source.prepare(rows, 128, a.num_threads)
        supported = []
        for lo in range(0, len(local), 128):
            selected = local[lo:lo+128]
            raw = source.read(selected)
            counts = np.stack([(raw == i).sum(0) for i in (0,1,2)])
            supported.extend(selected[counts.min(0) >= parent["minimum_reference_cell"]])
        local_support_markers = len(local)
        local = np.array(supported, dtype=np.int64)
        if target not in local or len(local) < 2:
            raise ValueError("new target/local panel lacks prespecified genotype support")
        selected = np.sort(np.concatenate([local, background]))
        fitted = estimate_scale(source, rows, selected, block_size=128,
            threads=a.num_threads, memory_bytes=int(a.memory_gib*2**30))
        inv = 1/np.sqrt(fitted.mean*(1-fitted.mean/2))
        bg_columns, local_columns = np.searchsorted(selected, background), np.searchsorted(selected, local)
        rng = np.random.default_rng(a.direction_seed)
        weights = np.zeros((len(selected),3), order="F")
        weights[bg_columns,0] = 1.
        weights[bg_columns,1] = rng.normal(size=len(background))
        sparse = rng.choice(bg_columns, min(32,len(background)), replace=False)
        weights[sparse,2] = rng.normal(size=len(sparse))
        genetic = np.zeros((len(rows),3), order="F")
        xlocal = np.empty((len(rows),len(local)))
        standard = StandardizedBlock(native, a.num_threads)
        source.prepare(rows,128,a.num_threads)
        for lo in range(0,len(selected),128):
            cols = np.arange(lo,min(lo+128,len(selected)))
            raw = source.read(selected[cols])
            x = standard.prepare(raw,np.arange(len(rows)),np.arange(len(cols)),fitted.mean[cols],inv[cols])
            product = np.empty_like(genetic,order="F")
            native.prediction_product(np.asfortranarray(x),np.asfortranarray(weights[cols]),product,False,a.num_threads)
            genetic += product
            for j in np.flatnonzero((local_columns >= lo) & (local_columns < lo+len(cols))):
                xlocal[:,j] = x[:,local_columns[j]-lo]
        normalization = genetic.std(0)
        if np.any(normalization <= 0):
            raise ValueError("constant reference direction")
        genetic /= normalization
        weights /= normalization
        x = xlocal[:,list(local).index(target)]
        causal = next(j for j in range(len(local)-1,-1,-1) if local[j] != target)
        local_mean = .8*xlocal[:,causal]
        old = {name:data["means"][complete,i] for i,name in enumerate(parent["settings"])}
        old_signals = {name:data["signals"][complete,i] for i,name in enumerate(parent["settings"])}
        old_local = old["aligned"]-old["dense"]-old_signals["aligned"]
        means = dict(dense=old["dense"], sparse=old["sparse"]-old_local+local_mean,
            dominance=old["dominance"]-old_local+local_mean,
            structure=old["structure"]-old_local+local_mean,
            heavy=old["heavy"]-old_local+local_mean)
        signals = {}
        for j,name in enumerate(["aligned","mixed","sparse_interaction"]):
            f = x*genetic[:,j]
            signals[name] = f*np.sqrt(parent["interaction_variance"]/f.var())
            means[name] = means["dense"]+local_mean+signals[name]
        means["structure_mixed"] = means["structure"]+signals["mixed"]
        definitions = {}
        for name,mean in means.items():
            signal = signals.get("mixed" if name=="structure_mixed" else name,np.zeros(len(rows)))
            definitions[name] = dict(biological_null=name not in signals and name!="structure_mixed",
                reference_mean_variance=float(mean.var()),reference_signal_variance=float(signal.var()),
                direction="mixed" if name=="structure_mixed" else name if name in signals else "aligned")
        samples = [source.samples[i] for i in rows]
        cov = _aligned_table(a.reference / "covariates.tsv",samples)
        cov.reset_index().to_csv(a.out / "covariates.tsv",sep="\t",index=False)
        pd.DataFrame(samples,columns=["FID","IID"]).to_csv(a.out / "samples.tsv",sep="\t",index=False)
        (a.out / "variants.txt").write_text("\n".join(v for i,v in enumerate(source.variants.ids) if i!=target)+"\n")
        (a.out / "interaction_variants.txt").write_text("\n".join(source.variants.ids[i] for i in background)+"\n")
        np.savez(a.out / "reference.npz",rows=rows,target=x,means=np.column_stack(list(means.values())),
            signals=np.column_stack([signals.get("mixed" if k=="structure_mixed" else k,np.zeros(len(rows))) for k in means]),
            variance=.4+.6*x*x,background=background,oracle_weights=weights[bg_columns],
            reference_inverse_scale=inv[bg_columns])
        after = io()
        meta = dict(parent,target=a.target,background_chromosome=getattr(a,"background_chromosome",None),
            interaction_background=background_record,
            background_variants=[source.variants.ids[i] for i in background],
            local_variants=[source.variants.ids[i] for i in local],n=len(rows),settings=list(means),definitions=definitions,
            alignment=dict(parent["alignment"],selected_n=len(rows)),
            parent_reference=str(a.reference.resolve()),direction_seed=a.direction_seed,
            main_architecture="parent full-marker coefficients and units retained; target-local stress term replaced",
            seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            io={k:after[k]-v for k,v in before.items()},
            genotype_traversals=dict(target_markers=1,local_support_markers=local_support_markers,
                scale_markers=len(selected),generating_product_markers=len(selected)),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(a.out / "reference.json",meta)
    print("target reference",a.target,len(rows),round(time.perf_counter()-begin,2),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--target",required=True)
    background=p.add_mutually_exclusive_group(required=True)
    background.add_argument("--background-chromosome")
    background.add_argument("--interaction-variants",type=Path)
    p.add_argument("--direction-seed",type=int,required=True)
    p.add_argument("--num-threads",type=int,default=2)
    p.add_argument("--memory-gib",type=float,default=16)
    run(p.parse_args())


if __name__=="__main__":
    main()
