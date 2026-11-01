"""Bound CPU thread usage without replacing sklearn algorithms."""
import os
from contextlib import contextmanager

# Import before NumPy/sklearn in the experiment entry point.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


@contextmanager
def execution_limits():
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits

    from experiment_config import NATIVE_THREADS

    with threadpool_limits(limits=NATIVE_THREADS):
        with parallel_backend("loky", inner_max_num_threads=NATIVE_THREADS):
            yield