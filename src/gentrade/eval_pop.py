import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import pool

import pandas as pd

from gentrade.eval_ind import BaseEvaluator
from gentrade.optimizer.individual import TreeIndividual


# The multiprocessing module serializes (pickles) any objects that are
# sent to worker processes. The strategy used here moves all of
# that constant data into a single immutable structure that is only
# shipped *once* when each worker starts.  The worker then retains a
# reference to the context in a module‑level global and the per‑task
# function (`worker_evaluate`) reads from it.  To an uninitiated reader
# the use of a global may look odd, but it is a standard and efficient
# pattern for avoiding repetitive IPC serialization in Python
# multiprocessing.
@dataclass(frozen=True)
class WorkerContext:
    """Immutable bundle of per-run state passed to each worker process.

    The context replaces the older approach of pickling a large data
    structure for every individual evaluation. Instead the pool
    initializer sends a single ``WorkerContext`` instance to each process.
    This context stores every worker depenedency that is constant across
    the full run.

    * ``evaluator`` – pre‑constructed evaluator instance (holds
      ``pset`` and ``metrics``).
    * ``train_data`` – list of OHLCV ``DataFrame`` objects.
    * ``train_entry_labels`` – optional list of ``Series`` aligned with
      ``train_data`` for entry signal ground truth.
    * ``train_exit_labels`` – optional list of ``Series`` aligned with
      ``train_data`` for exit signal ground truth.

    The pool initializer sends this object once to each worker; evaluation
    functions then reference ``_worker_ctx`` to access it.
    """

    evaluator: BaseEvaluator
    train_data: list[pd.DataFrame]
    train_entry_labels: list[pd.Series] | None
    train_exit_labels: list[pd.Series] | None


_worker_ctx: WorkerContext | None = None


def init_worker(ctx: WorkerContext) -> None:
    """Pool initializer that stores the provided context globally.

    This function is executed in each worker when the pool starts.  The
    ``ctx`` argument is the same ``WorkerContext`` object that was passed
    to ``create_pool`` in the main process; it is unpickled here and
    assigned to the module‑level ``_worker_ctx`` variable for later use.

    Args:
        ctx: the context object to store in the worker.
    """
    global _worker_ctx
    _worker_ctx = ctx


def create_pool(
    processes: int,
    evaluator: BaseEvaluator,
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
) -> pool.Pool:
    """Convenience wrapper for ``multiprocessing.Pool`` configured for GP.

    The supplied ``evaluator`` and datasets are packaged into a
    :class:`WorkerContext` and provided to the worker processes via the
    ``initializer`` mechanism.  All evaluation tasks will subsequently
    refer to this shared context rather than carrying the data themselves.

    Args:
        processes: number of worker processes to start.
        evaluator: evaluator instance containing ``pset`` and ``metrics``.
        train_data: list of OHLCV DataFrames.
        train_entry_labels: optional list of entry label Series aligned with
            ``train_data``.
        train_exit_labels: optional list of exit label Series aligned with
            ``train_data``.

    Returns:
        A started :class:`multiprocessing.Pool`.
    """
    ctx = WorkerContext(
        evaluator=evaluator,
        train_data=train_data,
        train_entry_labels=train_entry_labels,
        train_exit_labels=train_exit_labels,
    )
    return mp.Pool(processes=processes, initializer=init_worker, initargs=(ctx,))


def worker_evaluate(individual: TreeIndividual) -> tuple[float, ...]:
    """Top-level evaluation callback used with ``Pool.map``.

    Delegates directly to the evaluator stored in the global context.  The
    evaluator is responsible for handling the data, so this wrapper stays small
    and stable.

    Raises:
        RuntimeError: if called before the worker context has been set.
        ValueError: if required labels are missing in the context for the
            configured metrics and trade_side.
    """
    if _worker_ctx is None:
        raise RuntimeError("Worker context not initialized")

    evaluator = _worker_ctx.evaluator
    return evaluator.evaluate(
        individual,
        ohlcvs=_worker_ctx.train_data,
        entry_labels=_worker_ctx.train_entry_labels,
        exit_labels=_worker_ctx.train_exit_labels,
    )


def evaluate_population(
    population: list[TreeIndividual], pool: pool.Pool
) -> tuple[int, float]:
    """Evaluate all individuals in the population that have invalid fitness.
    The evaluation is performed in parallel using the provided pool and the
    ``worker_evaluate`` function.

    Args:
        population: list of ``TreeIndividual`` instances.
        pool: a :class:`multiprocessing.Pool`.

    Returns:
        n_evals: the number of individuals that were evaluated.
        duration: the time taken to perform all evaluations, in seconds.
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    t_start = time.perf_counter()
    fitnesses = pool.map(worker_evaluate, invalid_ind)

    duration = time.perf_counter() - t_start
    for ind, fit in zip(invalid_ind, fitnesses, strict=True):
        ind.fitness.values = fit
    return len(invalid_ind), duration
