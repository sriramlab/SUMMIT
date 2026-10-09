"""Independent-reference sampling variance of the annotation LD Gram.

Uniform partner draws estimate the iid second-order U-statistic variance.
Their extra variance is removed algebraically before transfer to study H.
"""
import numpy as np
from summit.sumstats.binary import finite_array


def reference_sampling_workspace_bytes(n,k,partners,block_size):
    pair_columns = k*(k+1)//2
    panel_rows = min(n*partners,max(32768,partners))
    return 8*(n*partners*(k+1)+5*n*k*k+8*k**4+n*block_size+
              4*panel_rows*pair_columns+2**21)+64*1024**2


def reference_pair_covariance(relatedness, *, sampled=True,native=False,threads=1):
    base = finite_array("reference pair relatedness",relatedness,3)
    n,l,k = base.shape
    if n < 4 or l < 2 or (not sampled and l != n-1):
        raise ValueError("reference sampling needs at least four people and two partners")
    from summit.ldscore.matrix_products import MatrixProducts
    from .sampling import sampling_tile_people
    backend = MatrixProducts(native=native,threads=threads)
    pairs = np.column_stack(np.triu_indices(k)).astype(np.int64)
    index = np.empty((k,k),dtype=np.int64)
    index[pairs[:,0],pairs[:,1]] = np.arange(len(pairs))
    index[pairs[:,1],pairs[:,0]] = np.arange(len(pairs))
    columns = len(pairs)
    rows = np.empty((n,columns),order="F")
    fourth = np.zeros((columns,columns))
    noise = np.zeros_like(fourth)
    width = sampling_tile_people(n,l)
    for start in range(0,n,width):
        stop = min(n,start+width)
        values = np.ascontiguousarray(base[start:stop].reshape(-1,k))
        products = (np.asarray(backend.module.pcgc_column_products(values,pairs,threads)) if native else
                    values[:,pairs[:,0]]*values[:,pairs[:,1]])
        row = (n-1)*products.reshape(stop-start,l,columns).mean(1)
        rows[start:stop] = row
        squares = backend.tn(products,products)
        fourth += (n-1)/l*squares
        if sampled:
            noise += ((n-1)**2/l*squares-backend.tn(row,row))/(l-1)
        backend.drain()
    total = rows.sum(0)
    centered = rows-rows.mean(0)
    denominator = float((n-2)*(n-3))
    factor = n*(n-1)/denominator
    covariance = factor*(4*backend.tn(centered,centered)-2*fourth)+2*np.outer(total,total)/denominator
    covariance -= (4*factor*(1-1/n)+2/denominator)*noise
    covariance /= float(n*(n-1))**2
    backend.drain()
    index = index.ravel()
    expanded = covariance[index[:,None],index[None,:]]
    return (expanded+expanded.T)/2


class ReferenceSamplingOperator:
    def __init__(self,operator,annotations,*,partners,seed,threads,native):
        from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator,ProtectedNNOperator
        n = operator.num_samples
        if n < 4:
            raise ValueError("reference sampling inference needs at least four people")
        self.native,self.threads = native,threads
        if native:
            from summit.prediction.genotype import native_module
            if any(not callable(getattr(native_module(),name,None))
                   for name in ('pcgc_column_products','pcgc_accumulate_relatedness')):
                raise RuntimeError("native extension lacks PCGC moment kernels; rebuild SUMMIT")
        self.operator = operator
        self.annotation_weights = annotations/annotations.sum(0)
        rng = np.random.default_rng(np.random.SeedSequence([seed,0x52454653]))
        self.partners = rng.integers(0,n-1,(n,partners),dtype=np.int64)
        self.partners += self.partners >= np.arange(n)[:,None]
        self.relatedness = np.zeros((n,partners,annotations.shape[1]))
        self.nn = ProtectedNNOperator(threads=threads) if native else NumpyNNOperator(threads=threads)

    def __getattr__(self,name):
        return getattr(self.operator,name)

    def read_block(self,start,stop):
        block = self.operator.read_block(start,stop)
        if self.observed_passes == 1:
            x = block.values
            a = np.asfortranarray(self.annotation_weights[start:stop])
            from .genotype import accumulate_relatedness
            accumulate_relatedness(x,a,self.partners,self.relatedness,
                nn=self.nn,native=self.native,threads=self.threads)
        return block

    def covariance(self):
        if self.observed_passes != 2:
            raise ValueError("reference inference requires both genotype traversals")
        return reference_pair_covariance(self.relatedness,native=self.native,threads=self.threads)
