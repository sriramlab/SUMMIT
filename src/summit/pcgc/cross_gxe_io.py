"""Private, separately sealed cross-trait reference and signed trait archives."""
from pathlib import Path
from time import perf_counter
import os
import tempfile
import zipfile
import numpy as np
from summit.ldscore.generalized_gxe_pass1 import _peak_rss_bytes

from summit.context.spec import array_sha256, canonical_sha256, canonical_json
from summit.prediction.spec import VariantAxis
from summit.ldscore.generalized_gxe_reference_v1 import _strict_json_loads
from .cross_gxe import CrossGxEMoments, CrossPairRows, pair_layout, prepare_cross_gxe

SCHEMA = "summit.pcgc.cross_context.v1"


def output_paths(prefix):
    return Path(str(prefix)+".cross.ldscores.npz"),Path(str(prefix)+".cross.sumstats.npz")


def _seal(header,arrays):
    header={**header,"array_hashes":{key:array_sha256(value) for key,value in arrays.items()}}
    return {**header,"manifest_hash":canonical_sha256(header)}


def _write(path,header,arrays):
    path=Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            np.savez_compressed(stream,manifest_json=np.asarray(canonical_json(header)),**arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp,path)
    finally:
        os.unlink(tmp)


def write_cross_artifacts(moments,prefix,*,variant_axis,annotation_names):
    start=perf_counter()
    moments.validate()
    paths=output_paths(prefix)
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")
    names=tuple(annotation_names)
    if len(names)!=moments.annotations.shape[1] or len(set(names))!=len(names) or any(not isinstance(s,str) or not s for s in names):
        raise ValueError("annotation names must be unique and match the array")
    if len(variant_axis.ids)!=len(moments.annotations):
        raise ValueError("variant/allele axis and reference disagree")
    shared=dict(schema=SCHEMA,traits=moments.traits,identity=moments.identity,
        variant_axis=variant_axis.to_dict(),annotation_names=names,
        pairs=[dict(left=p.left,right=p.right,overlap=p.overlap,sample_product=p.sample_product) for p in moments.pairs])
    ref={"annotations":moments.annotations}
    summary={f"scores_{i}":s for i,s in enumerate(moments.signed_scores)}
    for i,pair in enumerate(moments.pairs):
        ref[f"ld_{i}"]=pair.ldscores
        ref[f"probes_{i}"]=pair.probe_grams
        summary[f"rhs_{i}"]=pair.rhs_rows
    reference_header=_seal(dict(shared,kind="reference",diagnostics=moments.diagnostics),ref)
    summary_header=_seal(dict(shared,kind="signed_summary",
        reference_identity=reference_header["manifest_hash"]),summary)
    _write(paths[0],reference_header,ref)
    _write(paths[1],summary_header,summary)
    return dict(reference=str(paths[0]),sumstats=str(paths[1]),
                reference_identity=reference_header["manifest_hash"],serialization_seconds=perf_counter()-start)


def _read(path,kind):
    with np.load(path,allow_pickle=False) as archive:
        if "manifest_json" not in archive.files:
            raise ValueError("not a cross-trait PCGC archive")
        h=_strict_json_loads(str(archive["manifest_json"].item()))
        if not isinstance(h,dict) or h.get("schema")!=SCHEMA or h.get("kind")!=kind:
            raise ValueError("unsupported cross-trait PCGC archive kind/version")
        if h.get("manifest_hash")!=canonical_sha256({k:v for k,v in h.items() if k!="manifest_hash"}):
            raise ValueError("cross-trait manifest checksum mismatch")
        hashes=h.get("array_hashes",{})
        if len(archive.files)!=len(hashes)+1 or set(archive.files)!=set(hashes)|{"manifest_json"}:
            raise ValueError("cross-trait archive members disagree")
        arrays={}
        for name,digest in hashes.items():
            value=archive[name]
            if array_sha256(value)!=digest:
                raise ValueError(f"cross-trait {name} checksum mismatch")
            arrays[name]=value
    return h,arrays


def load_cross_artifacts(sumstats,ldscores,*,memory_bytes=None):
    if memory_bytes is not None:
        if type(memory_bytes) is not int or memory_bytes<1:
            raise ValueError("archive memory budget must be a positive integer")
        uncompressed=0
        for path in (sumstats,ldscores):
            with zipfile.ZipFile(path) as archive:
                uncompressed+=sum(member.file_size for member in archive.infolist())
        baseline=_peak_rss_bytes()
        if baseline+3*uncompressed>memory_bytes:
            raise MemoryError("cross-trait archive loading and checksum workspace exceed memory budget")
    rh,reference=_read(ldscores,"reference")
    sh,summary=_read(sumstats,"signed_summary")
    if sh.get("reference_identity")!=rh["manifest_hash"]:
        raise ValueError("summary requires its exact cross-trait reference realization")
    for key in ("traits","identity","variant_axis","annotation_names","pairs"):
        if sh.get(key)!=rh.get(key):
            raise ValueError(f"cross-trait {key} mismatch")
    axis=VariantAxis(**rh["variant_axis"])
    pairs=[]
    for i,p in enumerate(rh["pairs"]):
        qa,qb=[len(rh["traits"][t]["context_names"]) for t in (p["left"],p["right"])]
        pairs.append(CrossPairRows(**p,layout=pair_layout(qa,qb,p["left"]==p["right"]),
            ldscores=reference[f"ld_{i}"],rhs_rows=summary[f"rhs_{i}"],probe_grams=reference[f"probes_{i}"]))
    expected_ref={"annotations"}|{f"{name}_{i}" for name in ("ld","probes") for i in range(len(pairs))}
    expected_sum={f"rhs_{i}" for i in range(len(pairs))}|{f"scores_{i}" for i in range(len(rh["traits"]))}
    if set(reference)!=expected_ref or set(summary)!=expected_sum:
        raise ValueError("cross-trait archive has unexpected numerical arrays")
    moments=CrossGxEMoments(reference["annotations"],tuple(rh["traits"]),tuple(pairs),
        tuple(summary[f"scores_{i}"] for i in range(len(rh["traits"]))),rh["identity"],rh["diagnostics"])
    moments.validate()
    if len(axis.ids)!=len(moments.annotations):
        raise ValueError("cross-trait variant count mismatch")
    return moments,rh


def prepare_cross_from_source(source,scale,traits,annotations,*,genotype_covariates=None,
                              variant_indices=None,threads=1,native=True,block_size=256,**options):
    from .genotype import scaled_file_operator
    needed=set().union(*(set(t.samples) for t in traits))
    if not needed <= set(source.samples):
        raise ValueError("trait contains a participant absent from the genotype source")
    rows=np.array([i for i,p in enumerate(source.samples) if p in needed],dtype=np.int64)
    master=tuple(source.samples[i] for i in rows)
    operator=scaled_file_operator(source,scale,rows,variant_indices=variant_indices,
                                  threads=threads,native=native,block_size=block_size)
    extra=0
    if genotype_covariates is not None:
        from .ancestry import AncestryAdjustedOperator
        operator=AncestryAdjustedOperator(operator,genotype_covariates,threads=threads,native=native)
        extra=operator.workspace_bytes(block_size)
    moments=prepare_cross_gxe(operator,annotations,traits,master,threads=threads,native=native,
                             block_size=block_size,extra_bytes=extra,**options)
    moments.identity["genotype_descriptors"]=operator.descriptor_record
    if genotype_covariates is not None:
        moments.identity["genotype_ancestry_adjustment"]=operator.diagnostics
    return moments
