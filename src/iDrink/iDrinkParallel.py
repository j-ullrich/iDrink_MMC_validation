"""
Opt-in parallel execution of independent per-trial work (--jobs N).

jobs <= 1 runs a plain loop in the calling process, exactly like the pipeline did before.
Workers never write shared files. They return their result and the calling process writes it.
"""
import multiprocessing
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    """Outcome of one item. Either value is set, or error (with the formatted traceback attached as a note)."""
    item: Any
    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self):
        return self.error is None


def _call(func, item, in_worker=False):
    try:
        return Result(item=None, value=func(item))
    except Exception as e:  # noqa: BLE001 - reported to the caller, not swallowed
        if in_worker:
            e.add_note(traceback.format_exc())  # exceptions lose their traceback when pickled, notes survive
        return Result(item=None, error=e)


def run_parallel(func, items, jobs=1, progress=None):
    """
    Call func(item) for every item and return a list of Result in input order.

    :param func: Must be a module level function (picklable) when jobs > 1. Exceptions are captured per item.
    :param progress: Optional callable invoked in the calling process after each finished item.
    """
    items = list(items)
    results = [None] * len(items)

    if jobs <= 1 or len(items) <= 1:
        for i, item in enumerate(items):
            r = _call(func, item)
            r.item = item
            results[i] = r
            if progress:
                progress()
        return results

    try:
        pickle.dumps((func, items[0]))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Items cannot be sent to worker processes ({e}). Run with --jobs 1.") from e

    # spawn: fork is unsafe with CUDA and OpenSim
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {pool.submit(_call, func, item, True): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 - e.g. worker died (BrokenProcessPool)
                r = Result(item=None, error=e)
            r.item = items[i]
            results[i] = r
            if progress:
                progress()
    return results
