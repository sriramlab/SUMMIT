"""Explicit thread arguments for epistasis in a mixed native test process."""
import os


def epistasis_threads():
    from summit.prediction.genotype import native_module
    native=native_module()
    info=native.build_info()
    if info.get('blas_runtime_environment_immutable',False):
        return int(info['blas_runtime_threads'])
    return int(native.configured_blas_threads()) or int(os.environ.get('SUMMIT_EPISTASIS_TEST_THREADS','1'))


def cli(argv):
    from summit.epistasis.cli import main
    args=list(argv)
    if args[0] in ('make-inputs','prepare','prepare-traits','train-direction') and '--num-threads' not in args:
        args+=['--num-threads',str(epistasis_threads())]
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS')
    previous={k:os.environ.get(k) for k in keys}
    try:
        for k in keys:os.environ[k]=str(epistasis_threads())
        return main(args)
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def entrypoint(argv):
    if argv[0]!='epistasis':raise ValueError('epistasis test entrypoint expected')
    return cli(argv[1:])
