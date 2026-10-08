"""Ancestry adjustment of population-scaled genotypes before GxE weighting."""
from dataclasses import replace
import numpy as np

from summit.context.spec import array_sha256
from summit.sumstats.binary import finite_array
from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator, ProtectedTNOperator


class AncestryAdjustedOperator:
    """Apply (I-U U.T) G to each decoded block on the common genotype scale.

    Only supplied directions are removed. No intercept, centering, phenotype
    projection or context-specific genotype normalization is added.
    """
    def __init__(self, operator, covariates, *, threads=1, native=True):
        design = finite_array("genotype ancestry covariates", covariates, 2)
        n, r = design.shape
        if n != operator.num_samples or not 0 < r < n:
            raise ValueError("genotype ancestry covariates must have shape N by R with 0 < R < N")
        norms = np.linalg.norm(design,axis=0)
        if np.any(norms == 0) or np.linalg.matrix_rank(design/norms) != r:
            raise ValueError("genotype ancestry covariates are rank deficient")
        self.operator = operator
        self.u = np.asfortranarray(np.linalg.qr(design/norms,mode="reduced")[0])
        self.diagnostics = dict(method="genotype_projection_before_risk_and_context_v1",
            rank=r, design_sha256=array_sha256(design), projection_basis_sha256=array_sha256(self.u),
            automatic_intercept=False, post_projection_normalization=False)
        if native:
            from summit.prediction.genotype import native_module
            from summit.prediction.runtime import configure_prediction_threads
            module = native_module()
            configure_prediction_threads(module,threads)
            self.nn = ProtectedNNOperator(threads=threads,native_module=module)
            self.tn = ProtectedTNOperator(threads=threads,native_module=module)
        else:
            self.nn = NumpyNNOperator(threads=threads)
            self.tn = NumpyTNOperator(threads=threads)

    def __getattr__(self,name):
        return getattr(self.operator,name)

    def workspace_bytes(self,block_size):
        n,r = self.u.shape
        b = min(block_size,self.num_variants)
        return 8*(5*n*r+2*n*b+r*b+3*r*r)

    def read_block(self,start,stop):
        block = self.operator.read_block(start,stop)
        coefficients = self.tn.matmul_tn(self.u,block.values)
        adjusted = np.asfortranarray(block.values-self.nn.matmul(self.u,np.asfortranarray(coefficients)))
        return replace(block,values=adjusted)
