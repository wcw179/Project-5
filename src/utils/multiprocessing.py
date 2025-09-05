"""
Module for multiprocessing
"""

import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool


def _process_jobs(jobs):
    """
    Run jobs sequentially, for debugging

    :param jobs: (list) jobs to process
    :return: (list) of results
    """
    out = []
    for job in jobs:
        out.append(_expand_call(job))
    return out


def _expand_call(kargs):
    """
    Expand the arguments of a callback function, kargs['func']

    :param kargs: (dict) contains callback function and arguments
    :return: (func) callback function
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_parts=None, **kargs):
    """
    Parallelize jobs, return a dataframe or series

    :param func: (func) function to be parallelized. Returns a DataFrame
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass the molecule
                   Element 1: A list of indivisible tasks (molecules)
    :param num_threads: (int) number of threads
    :param mp_batches: (int) number of batches
    :param lin_parts: (int) number of linear partitions (optional)
    :param kargs: (dict) of keyword arguments for the function
    :return: (pd.DataFrame) concatenated results
    """
    if lin_parts is None:
        lin_parts = int(len(pd_obj[1]) / mp_batches)
    # The number of partitions is the number of jobs
    parts = np.linspace(0, len(pd_obj[1]), min(lin_parts + 1, len(pd_obj[1]) + 1))
    parts = np.ceil(parts).astype(int)

    jobs = []
    for i in range(len(parts) - 1):
        job = {pd_obj[0]: pd_obj[1][parts[i]:parts[i + 1]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = _process_jobs(jobs)
    else:
        # The pool of processes is the number of threads
        pool = Pool(processes=num_threads)
        outputs = pool.imap_unordered(_expand_call, jobs)
        out = []
        # Process asynchronous output, report progress
        for output in outputs:
            out.append(output)
        pool.close()
        pool.join()  # This is needed to prevent memory leaks

    # Vanila sort because the jobs are not guaranteed to be processed in order
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    else:
        df0 = pd.Series(dtype=out[0].dtype)

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()
    return df0
