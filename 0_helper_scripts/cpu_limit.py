# in 0_helper_scripts/cpu_limit.py

import os

def limit_cpus(n: int = 4, start: int = 0) -> set:
    """
    Hard-limit this process (and all children) to n CPUs.
    Enforced at kernel level — inherited by subprocesses.

    Parameters
    ----------
    n     : number of CPUs to allow
    start : first CPU index (default 0)

    Returns
    -------
    set of allowed CPU indices
    """
    cpus = set(range(start, start + n))
    os.sched_setaffinity(0, cpus)
    allowed = os.sched_getaffinity(0)
    print(f"CPUs restricted to: {sorted(allowed)}  ({len(allowed)} cores)")
    return allowed