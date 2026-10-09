"""Bounded cross-PCGC timing; synthetic outcomes only, no real-data estimates."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import time
import numpy as np
import pandas as pd

from summit.context.spec import canonical_sha256
from summit.pcgc.cross_gxe import prepare_cross_gxe
from summit.pcgc.cross_gxe_model import binary_trait,quantitative_trait
from summit.pcgc.cross_gxe_io import prepare_cross_from_source,write_cross_artifacts
from summit.pcgc.cross_gxe_fit import fit_cross_gxe
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator, _peak_rss_bytes
from summit.prediction.genotype import FileGenotypeSource,native_module
from summit.prediction.spec import GenotypeScale,VariantAxis


def generated_traits(ids,indices,q):
    rng=np.random.default_rng(19202)
    traits=[]
    start=time.perf_counter()
    for index,rr in enumerate(indices):
        phi=np.column_stack((np.ones(len(rr)),rng.uniform(-1,1,(len(rr),q-1))))
        samples=tuple(ids[i] for i in rr)
        common=dict(context_names=tuple(["intercept"]+[f"E{i}" for i in range(1,q)]),
                    selection="outcome_independent")
        if index%2==0:
            risk=np.full(len(rr),.2) if q==1 else 1/(1+np.exp(1.4-.5*phi[:,1]))
            y=(rng.uniform(size=len(rr))<risk).astype(float)
            traits.append(binary_trait(f"T{index}",samples,y,phi,liability_sd=1.,
                                       population_risk=risk,**common))
        else:
            traits.append(quantitative_trait(f"T{index}",samples,rng.normal(size=len(rr)),phi,
                population_sd=1.,fixed_effects=np.column_stack((np.ones(len(rr)),rng.uniform(-1,1,len(rr)))),**common))
    return traits,time.perf_counter()-start


def real_inputs(args):
    source=FileGenotypeSource(args.bed,genome_build="GRCh37")
    freq=pd.read_csv(args.afreq,sep=r"\s+",dtype={"ID":str,"REF":str,"ALT":str}).set_index("ID").loc[list(source.variants.ids)]
    counted=np.asarray(source.variants.counted);other=np.asarray(source.variants.other)
    direct=(counted==freq.ALT)&(other==freq.REF)
    reverse=(counted==freq.REF)&(other==freq.ALT)
    if not np.all(direct|reverse):
        raise ValueError("population allele frequencies disagree with genotype alleles")
    f=np.where(direct,freq.ALT_FREQS,1-freq.ALT_FREQS)
    scale=GenotypeScale(2*f,1/np.sqrt(2*f*(1-f)),source.variants.identity,
        canonical_sha256({"frequency_source":str(Path(args.afreq).resolve())}),
        {"population_scale":True,"benchmark_assumption":"panel_allele_frequencies"},ddof=0)
    ann=pd.read_csv(args.annot,sep=r"\s+",dtype={"SNP":str}).set_index("SNP").loc[list(source.variants.ids)]
    names=[name for name in ann if name.startswith("maf")]
    if len(names)!=8:
        raise ValueError("full-cohort benchmark requires the eight disjoint MAF/LD components")
    a=ann[names].to_numpy(dtype=float)
    if not np.all(a.sum(1)==1):
        raise ValueError("primary MAF/LD annotation must be disjoint and exhaustive")
    if args.m<8:
        raise ValueError("bounded real benchmark needs at least one SNP per component")
    selected=[]
    for component in range(8):
        available=np.flatnonzero(a[:,component])
        count=args.m//8+(component<args.m%8)
        selected.extend(available[np.linspace(0,len(available)-1,count,dtype=int)])
    variants=np.array(sorted(selected),dtype=np.int64)
    n=len(source.samples) if args.n==0 else args.n
    if n>len(source.samples):
        raise ValueError("requested sample count exceeds source")
    ids=tuple(source.samples[:n])
    pcs=pd.read_csv(args.pcs,sep=r"\s+",dtype={"#FID":str,"IID":str}).rename(columns={"#FID":"FID"})
    pcs=pcs.set_index(["FID","IID"]).loc[pd.MultiIndex.from_tuples(ids),[f"PC{i}" for i in range(1,11)]].to_numpy(dtype=float)
    return source,scale,a[variants],ids,variants,pcs,names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=512,help="samples per trait; 0 uses all real-source participants")
    parser.add_argument("--m",type=int,default=128)
    parser.add_argument("--contexts",type=int,default=2)
    parser.add_argument("--annotations",type=int,default=2)
    parser.add_argument("--traits",type=int,default=2)
    parser.add_argument("--overlap",type=float,default=1.)
    parser.add_argument("--probes",type=int,default=32)
    parser.add_argument("--block-size",type=int,default=64)
    parser.add_argument("--probe-tile",type=int,default=16)
    parser.add_argument("--memory-gib",type=float,default=8)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--separate",action="store_true")
    parser.add_argument("--bed");parser.add_argument("--afreq");parser.add_argument("--annot");parser.add_argument("--pcs")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--serialize",action="store_true")
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    os.umask(0o077)
    start=time.perf_counter()
    cpu_start=resource.getrusage(resource.RUSAGE_SELF)
    source=None
    if args.bed:
        if args.overlap!=1.:
            raise ValueError("real bounded timing currently uses full overlap")
        source,scale,a,ids,variants,pcs,names=real_inputs(args)
        indices=[np.arange(len(ids)) for _ in range(args.traits)]
        axis=source.variants.subset(variants)
        genotype=None
    else:
        n=args.n;shared=int(n*args.overlap)
        total=shared+args.traits*(n-shared)
        ids=tuple((str(i),str(i)) for i in range(total))
        indices=[np.r_[np.arange(shared),np.arange(shared+t*(n-shared),shared+(t+1)*(n-shared))] for t in range(args.traits)]
        genotype=np.random.default_rng(825).normal(size=(total,args.m))
        a=np.eye(args.annotations)[np.arange(args.m)%args.annotations]
        names=[f"a{i}" for i in range(args.annotations)]
        axis=VariantAxis(tuple(f"s{i}" for i in range(args.m)),("1",)*args.m,tuple(range(1,args.m+1)),("A",)*args.m,("G",)*args.m)
    traits,risk_seconds=generated_traits(ids,indices,args.contexts)
    input_seconds=time.perf_counter()-start
    options=dict(probes=args.probes,seed=191,block_size=args.block_size,probe_tile=args.probe_tile,
                 memory_bytes=int(args.memory_gib*2**30),threads=args.threads)
    def run(trait_indices):
        tt=[traits[i] for i in trait_indices]
        if source is None:
            needed=set().union(*(set(t.samples) for t in tt))
            rr=np.array([i for i,s in enumerate(ids) if s in needed],dtype=np.int64)
            return prepare_cross_gxe(ArraySequentialGenotypeOperator(genotype[rr]),a,tt,[ids[i] for i in rr],**options)
        return prepare_cross_from_source(source,scale,tt,a,variant_indices=variants,
                                         genotype_covariates=pcs,**options)
    moment=run(range(args.traits))
    report=dict(configuration=vars(args)|{"out":str(args.out)},input_seconds=input_seconds,
        risk_preparation_seconds=risk_seconds,master_samples=len(ids),variants=args.m,effective_annotations=a.shape[1],
        synthetic_outcomes=True,real_data_inference=False,shared=moment.diagnostics,
        implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve().parents[2]/"src/summit/pcgc").glob("cross_gxe*.py")},
        native_build=native_module().build_info(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.serialize:
        report["serialization"]=write_cross_artifacts(moment,args.out.with_suffix(""),variant_axis=axis,annotation_names=names)
    try:
        fit=fit_cross_gxe(moment,block_ids=np.arange(args.m)*min(8,args.m)//args.m)
        report["fit_uncertainty_seconds"]=fit["fit_uncertainty_seconds"]
    except ValueError as exc:
        report["bounded_fit_diagnostic"]=str(exc)
    if args.separate:
        reports=[]
        for left in range(args.traits):
            for right in range(left+1,args.traits):
                reports.append(run([left,right]).diagnostics)
        report["separate_pairs"]=reports
        report["separate_pair_seconds"]=sum(r["timings"]["total_seconds"] for r in reports)
        report["separate_pair_genotype_passes"]=sum(r["genotype_passes"] for r in reports)
    report["total_seconds"]=time.perf_counter()-start
    cpu_end=resource.getrusage(resource.RUSAGE_SELF)
    report["cpu_seconds"]=cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime
    report["average_active_cores"]=report["cpu_seconds"]/report["total_seconds"]
    report["peak_rss_bytes"]=_peak_rss_bytes()
    if source is not None:
        source.close()
    with args.out.open("x") as stream:
        json.dump(report,stream,indent=2,default=str)
    print(json.dumps(dict(receipt=str(args.out),samples=len(ids),variants=args.m,
        total_seconds=report["total_seconds"],peak_rss_bytes=report["peak_rss_bytes"],
        genotype_passes=moment.diagnostics["genotype_passes"])),flush=True)


if __name__=="__main__":
    main()
