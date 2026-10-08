"""Independent-reference sampling variance of the annotation LD Gram.

Uniform partner draws estimate the iid second-order U-statistic variance.
Their extra variance is removed algebraically before transfer to study H.
"""
import numpy as np
from summit.sumstats.binary import finite_array


def reference_sampling_workspace_bytes(n,k,partners,block_size):
    return 8*(n*partners*(k+1)+5*n*k*k+8*k**4+2**21)+64*1024**2


def reference_pair_covariance(relatedness, *, sampled=True):
    base = finite_array("reference pair relatedness",relatedness,3)
    n,l,k = base.shape
    if n < 4 or l < 2 or (not sampled and l != n-1):
        raise ValueError("reference sampling needs at least four people and two partners")
    rows = np.empty((n,k*k))
    fourth = np.zeros((k*k,k*k))
    noise = np.zeros_like(fourth)
    width = max(1,min(n,2**20//max(1,l*k*k)))
    for start in range(0,n,width):
        stop = min(n,start+width)
        pairs = base[start:stop]
        products = (pairs[:,:,:,None]*pairs[:,:,None,:]).reshape(stop-start,l,k*k)
        row = (n-1)*products.mean(1)
        rows[start:stop] = row
        squares = products.reshape(-1,k*k).T@products.reshape(-1,k*k)
        fourth += (n-1)/l*squares
        if sampled:
            noise += ((n-1)**2/l*squares-row.T@row)/(l-1)
    total = rows.sum(0)
    centered = rows-rows.mean(0)
    denominator = (n-2)*(n-3)
    factor = n*(n-1)/denominator
    covariance = factor*(4*centered.T@centered-2*fourth)+2*np.outer(total,total)/denominator
    covariance -= (4*factor*(1-1/n)+2/denominator)*noise
    covariance /= (n*(n-1))**2
    return (covariance+covariance.T)/2


class ReferenceSamplingOperator:
    def __init__(self,operator,annotations,*,partners,seed,threads,native):
        from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator,ProtectedNNOperator
        n = operator.num_samples
        if n < 4:
            raise ValueError("reference sampling inference needs at least four people")
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
            n,l,k = self.relatedness.shape
            values = self.relatedness.reshape(-1,k)
            other = self.partners.ravel()
            width = max(1,2**20//(stop-start))
            a = np.asfortranarray(self.annotation_weights[start:stop])
            for first in range(0,n*l,width):
                last = min(n*l,first+width)
                rows = np.arange(first,last)//l
                values[first:last] += self.nn.matmul(np.asfortranarray(x[rows]*x[other[first:last]]),a)
        return block

    def covariance(self):
        if self.observed_passes != 2:
            raise ValueError("reference inference requires both genotype traversals")
        return reference_pair_covariance(self.relatedness)
