"""Explicit adapters from architecture artifacts; no hidden re-estimation."""
from ._validation import closed, digest, psd
from .artifacts import read_json
from .spec import GenotypeScale


def scale_from_generalized_reference(reference, variants, samples, *, sample_identity,
                                     variant_identity, provenance):
    """Attach independently authenticated axes to a legacy generalized reference.

    Current reference bundles carry affine arrays but not authenticated sample
    and allele axes. The caller must resolve these from the reference's original
    manifest; matching dimensions alone is deliberately insufficient.
    """
    if digest(samples) != sample_identity or variants.identity != variant_identity:
        raise ValueError("reference axis authentication failed")
    if reference.affine_mean is None or reference.affine_inverse_scale is None:
        raise ValueError("reference lacks sealed affine parameters; recover the original scale artifact")
    return GenotypeScale(reference.affine_mean, reference.affine_inverse_scale,
                         variant_identity, sample_identity, provenance)


def load_prior(path, *, context_spec, scale):
    value = read_json(path)
    closed(value, ("kind", "schema_version", "covariance", "context_identity", "scale_identity", "provenance"),
           ("raw_covariance", "coefficient_covariance", "geometry"), name="architecture prior")
    if value["kind"] != "summit.prediction.prior" or value["schema_version"] != 1:
        raise ValueError("unsupported architecture prior")
    if value["context_identity"] != digest(context_spec) or value["scale_identity"] != scale.identity:
        raise ValueError("prior context or exact genotype scale identity mismatch")
    if not value["provenance"]:
        raise ValueError("architecture prior requires fitting/PSD provenance")
    return psd(value["covariance"]), value
