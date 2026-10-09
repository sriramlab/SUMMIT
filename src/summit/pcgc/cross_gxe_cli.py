"""Joint generalized cross-trait PCGC preparation and genotype-free fitting."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def _runtime(threads):
    if threads < 1:
        raise ValueError("thread count must be positive")
    for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","OMP_NUM_THREADS","OMP_THREAD_LIMIT"):
        os.environ.setdefault(key,str(threads))
    if os.uname().nodename.lower()=="tabla":
        cpus=set(os.sched_getaffinity(0))
        if not cpus <= set(range(8,64)) or not 1 <= len(cpus) <= 2:
            raise RuntimeError("Tabla cross-PCGC requires outer taskset on one or two available physical cores 8-63")
        import ctypes
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(41,1,0,0,0)!=0 or libc.prctl(42,0,0,0,0)!=1:
            raise RuntimeError("process-local THP guard failed")


def prepare_manifest(path, **options):
    import numpy as np
    import pandas as pd
    from summit.prediction.genotype import FileGenotypeSource
    from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
    from .cli import sample_table,read_table,population_scale,annotation_table
    from .cross_gxe_model import binary_trait,quantitative_trait
    from .cross_gxe_io import prepare_cross_from_source
    path=Path(path)
    config=_strict_json_loads(path.read_text())
    allowed={"genotype","population_scale","annotations","genome_build","traits","pairs","genotype_adjustment"}
    if not isinstance(config,dict) or set(config)-allowed:
        raise ValueError("unknown cross-PCGC preparation fields")
    def resolve(value):
        p=Path(value)
        return p if p.is_absolute() else path.parent/p
    with FileGenotypeSource(resolve(config["genotype"]),genome_build=config.get("genome_build")) as source:
        scale=population_scale(resolve(config["population_scale"]),source)
        a,annotation_names=annotation_table(resolve(config["annotations"]) if config.get("annotations") else None,source)
        adjustment=None
        if config.get("genotype_adjustment"):
            spec=config["genotype_adjustment"]
            if set(spec)!={"table","columns"} or not spec["columns"]:
                raise ValueError("genotype_adjustment needs a table and nonempty columns")
            adjustment=read_table(resolve(spec["table"]),("FID","IID"))
            if not {"FID","IID",*spec["columns"]} <= set(adjustment) or adjustment.duplicated(["FID","IID"]).any():
                raise ValueError("genotype adjustment requires unique participant IDs and all PC columns")
            adjustment=adjustment.set_index(["FID","IID"])
        traits=[]
        risk_start=perf_counter()
        for spec in config["traits"]:
            allowed={"name","kind","table","contexts","selection","population_sd","fixed_effects",
                     "risk_column","risk_covariates","prevalence","sampling_odds","liability_sd_column","unit_liability"}
            if set(spec)-allowed:
                raise ValueError("unknown cross-PCGC trait fields")
            rows,table=sample_table(resolve(spec["table"]),source)
            ids=tuple(source.samples[i] for i in rows)
            contexts=spec.get("contexts",[])
            if len(set(contexts))!=len(contexts) or "intercept" in contexts:
                raise ValueError("context names must be unique; the intercept is added")
            phi=np.column_stack((np.ones(len(rows)),table[contexts].to_numpy(dtype=float)))
            common=dict(context_names=("intercept",*contexts),selection=spec["selection"])
            y=table.Y.to_numpy(dtype=float)
            if spec["kind"]=="quantitative":
                forbidden={"risk_column","risk_covariates","prevalence","sampling_odds","liability_sd_column","unit_liability"}&set(spec)
                if forbidden:
                    raise ValueError("binary options supplied for a quantitative trait")
                fixed=spec.get("fixed_effects",[])
                design=np.column_stack((np.ones(len(rows)),table[fixed].to_numpy(dtype=float)))
                traits.append(quantitative_trait(spec["name"],ids,y,phi,population_sd=spec["population_sd"],
                                                  fixed_effects=design,**common))
            elif spec["kind"]=="binary":
                if "fixed_effects" in spec or "population_sd" in spec:
                    raise ValueError("binary fixed covariates belong in the risk model")
                if "unit_liability" in spec and type(spec["unit_liability"]) is not bool:
                    raise ValueError("unit_liability must be a JSON Boolean")
                if bool(spec.get("unit_liability"))==bool(spec.get("liability_sd_column")):
                    raise ValueError("declare unit_liability or a supplied liability_sd_column")
                sd=1. if spec.get("unit_liability") else table[spec["liability_sd_column"]].to_numpy(dtype=float)
                k=None if not spec.get("risk_column") else table[spec["risk_column"]].to_numpy(dtype=float)
                cov=None
                if k is None:
                    names=list(dict.fromkeys(contexts+spec.get("risk_covariates",[])))
                    cov=table[names].to_numpy(dtype=float)
                    if adjustment is not None:
                        index=pd.MultiIndex.from_tuples(ids,names=["FID","IID"])
                        cov=np.column_stack((cov,adjustment.loc[index,config["genotype_adjustment"]["columns"]].to_numpy(dtype=float)))
                elif spec.get("risk_covariates"):
                    raise ValueError("choose supplied risk_column or fitted risk_covariates")
                traits.append(binary_trait(spec["name"],ids,y,phi,liability_sd=sd,
                    population_risk=k,risk_covariates=cov,prevalence=spec.get("prevalence"),
                    sampling_odds=spec.get("sampling_odds"),**common))
            else:
                raise ValueError("trait kind must be quantitative or binary")
        risk_seconds=perf_counter()-risk_start
        genotype_covariates=None
        if adjustment is not None:
            needed=set().union(*(set(t.samples) for t in traits))
            master=[p for p in source.samples if p in needed]
            index=pd.MultiIndex.from_tuples(master,names=["FID","IID"])
            genotype_covariates=adjustment.loc[index,config["genotype_adjustment"]["columns"]].to_numpy(dtype=float)
        pairs=None
        if "pairs" in config:
            lookup={t.name:i for i,t in enumerate(traits)}
            pairs=[tuple(lookup[name] for name in pair) for pair in config["pairs"]]
        moments=prepare_cross_from_source(source,scale,traits,a,pairs=pairs,
            genotype_covariates=genotype_covariates,**options)
        moments.diagnostics["timings"]["risk_preparation_seconds"]=risk_seconds
        return moments,source.variants,annotation_names


def _jsonable(value):
    import numpy as np
    if isinstance(value,dict):
        return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,np.bool_)):
        return value.item()
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare",help="joint study-specific reference and signed summaries")
    prep.add_argument("manifest",type=Path)
    prep.add_argument("--out",type=Path,required=True)
    prep.add_argument("--nvecs",type=int,default=256)
    prep.add_argument("--seed",type=int,default=0)
    prep.add_argument("--block-size",type=int,default=256)
    prep.add_argument("--probe-tile",type=int,default=32)
    fit=sub.add_parser("fit",help="fit matching separate files without genotypes")
    fit.add_argument("--sumstats",type=Path,required=True)
    fit.add_argument("--ldscores",type=Path,required=True)
    fit.add_argument("--out",type=Path,required=True)
    fit.add_argument("--njack",type=int,default=200)
    fit.add_argument("--population-metrics",type=Path,help="JSON of fixed annotation-specific population kernel moments")
    for p in (prep,fit):
        p.add_argument("--num-threads",type=int,default=1)
        p.add_argument("--memory-gib",type=float,default=4)
    args=parser.parse_args(argv)
    _runtime(args.num_threads)
    import numpy as np
    from .cross_gxe_io import output_paths,write_cross_artifacts,load_cross_artifacts
    if not np.isfinite(args.memory_gib) or args.memory_gib<=0:
        raise ValueError("memory-gib must be finite and positive")
    memory_bytes=int(args.memory_gib*2**30)
    if args.command=="prepare":
        for path in output_paths(args.out):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing to overwrite {path}")
        moments,axis,names=prepare_manifest(args.manifest,probes=args.nvecs,seed=args.seed,
            block_size=args.block_size,probe_tile=args.probe_tile,memory_bytes=memory_bytes,
            threads=args.num_threads)
        receipt=write_cross_artifacts(moments,args.out,variant_axis=axis,annotation_names=names)
        print(json.dumps({**receipt,"diagnostics":moments.diagnostics},sort_keys=True))
    else:
        from .cross_gxe_fit import fit_cross_gxe,derived_covariance
        from summit.prediction.artifacts import write_json
        from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
        output=Path(str(args.out)+".cross.json")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite {output}")
        moments,header=load_cross_artifacts(args.sumstats,args.ldscores,memory_bytes=memory_bytes)
        m=len(moments.annotations)
        if not 2<=args.njack<=m:
            raise ValueError("njack must be between 2 and the variant count")
        blocks=np.arange(m,dtype=np.int64)*args.njack//m
        result=fit_cross_gxe(moments,block_ids=blocks,memory_bytes=memory_bytes)
        result.update(reference_identity=header["manifest_hash"],annotation_names=header["annotation_names"])
        if args.population_metrics:
            metrics=_strict_json_loads(args.population_metrics.read_text())
            if set(metrics)!={"target_population","centering","kernel_second_moments","total_variances"} or not metrics["target_population"]:
                raise ValueError("population metrics need a target, centering, kernel_second_moments and total_variances")
            if metrics["centering"]!="population_feature_means_removed":
                raise ValueError("population genetic variances require centered genotype-by-context feature moments")
            metric=metrics["kernel_second_moments"]
            result["population"]={"target":metrics["target_population"],"centering":metrics["centering"],"pairs":{}}
            for pair in moments.pairs:
                if pair.left==pair.right:
                    continue
                l,r=pair.left,pair.right
                aa,bb,ab=[np.asarray(metric[key],dtype=float) for key in (f"{l}:{l}",f"{r}:{r}",f"{l}:{r}")]
                for a,b,c in zip(aa,bb,ab):
                    joint=np.block([[a,c],[c.T,b]])
                    if not np.allclose(joint,joint.T,rtol=1e-12,atol=1e-12) or np.linalg.eigvalsh(joint).min() < -1e-10:
                        raise ValueError("joint population kernel moments must be positive semidefinite")
                result["population"]["pairs"][f"{l}:{r}"]=derived_covariance(result,l,r,aa,bb,ab,
                    total_variances=[metrics["total_variances"][l],metrics["total_variances"][r]])
        output.parent.mkdir(parents=True,exist_ok=True)
        old=os.umask(0o077)
        try:
            write_json(output,_jsonable(result))
        finally:
            os.umask(old)
        print(output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
