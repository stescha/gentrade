import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import pool

import pandas as pd
from deap import gp

from gentrade.eval_ind import BacktestEvaluator, ClassificationEvaluator


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
    * ``train_data`` – mapping from dataset keys to OHLCV ``DataFrame``
      objects.
    * ``train_labels`` – optional mapping from the same keys to label
      ``Series`` objects used by :class:`ClassificationEvaluator`.

    The pool initializer sends this object once to each worker; evaluation
    functions then reference ``_worker_ctx`` to access it.
    """

    evaluator: BacktestEvaluator | ClassificationEvaluator
    train_data: dict[str, pd.DataFrame]
    train_labels: dict[str, pd.Series] | None


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
    evaluator: BacktestEvaluator | ClassificationEvaluator,
    train_data: dict[str, pd.DataFrame],
    train_labels: dict[str, pd.Series] | None,
) -> pool.Pool:
    """Convenience wrapper for ``multiprocessing.Pool`` configured for GP.

    The supplied ``evaluator`` and datasets are packaged into a
    :class:`WorkerContext` and provided to the worker processes via the
    ``initializer`` mechanism.  All evaluation tasks will subsequently
    refer to this shared context rather than carrying the data themselves.

    Args:
        processes: number of worker processes to start (may be 1).
        evaluator: evaluator instance containing ``pset`` and ``metrics``.
        train_data: mapping of dataset keys to OHLCV DataFrames.
        train_labels: optional mapping of dataset keys to label series for
            classification runs.

    Returns:
        A started :class:`multiprocessing.Pool`.
    """
    ctx = WorkerContext(
        evaluator=evaluator,
        train_data=train_data,
        train_labels=train_labels,
    )
    return mp.Pool(processes=processes, initializer=init_worker, initargs=(ctx,))


def worker_evaluate(individual: gp.PrimitiveTree) -> tuple[float, ...]:
    """Top-level evaluation callback used with ``Pool.map``.

    Delegates directly to the evaluator stored in the global context.  The
    evaluator is now responsible for handling either a single DataFrame or a
    mapping of dataframes, so this wrapper stays small and stable.

    Raises:
        RuntimeError: if called before the worker context has been set.
        ValueError: if a classification evaluator is used but no labels were
            provided in the context.
    """
    if _worker_ctx is None:
        raise RuntimeError("Worker context not initialized")

    evaluator = _worker_ctx.evaluator
    if isinstance(evaluator, ClassificationEvaluator):
        if _worker_ctx.train_labels is None:
            raise ValueError("ClassificationEvaluator requires train_labels")
        return evaluator.evaluate(
            individual,
            df=_worker_ctx.train_data,
            y_true=_worker_ctx.train_labels,
        )
    else:
        return evaluator.evaluate(individual, df=_worker_ctx.train_data)


def evaluate_population(
    population: list[gp.PrimitiveTree], pool: pool.Pool
) -> tuple[int, float]:
    """Evaluate all individuals in the population that have invalid fitness.
    The evaluation is performed in parallel using the provided pool and the
    ``worker_evaluate`` function.

    Args:
        population: list of GP trees.
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
