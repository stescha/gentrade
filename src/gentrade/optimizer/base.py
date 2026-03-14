"""Base optimizer class and data normalization utilities.

Contains ``BaseOptimizer`` ABC which provides the shared orchestration logic
for all GP optimizers. Subclasses implement pset construction, toolbox wiring,
and evaluator creation.
"""

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from multiprocessing import pool
from typing import Any, cast

import numpy as np
import pandas as pd
from deap import base, gp, tools

from gentrade._defaults import (
    KEY_OHLCV,
    SELECTION_MULTI_OBJ,
    SELECTION_SINGLE_OBJ,
)
from gentrade.classification_metrics import ClassificationMetricBase
from gentrade.eval_ind import IndividualEvaluator
from gentrade.eval_pop import create_pool
from gentrade.optimizer.callbacks import Callback, ValidationCallback
from gentrade.optimizer.individual import TreeIndividual
from gentrade.optimizer.types import Algorithm, Metric

logger = logging.getLogger(__name__)

# Type aliases for data inputs
DataInput = pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None
LabelInput = pd.Series | dict[str, pd.Series] | list[pd.Series] | None


def _normalize_data_and_labels(
    data: DataInput,
    labels: LabelInput,
    dataset_name: str,
) -> tuple[list[pd.DataFrame], list[pd.Series] | None, list[str]]:
    """Normalise and validate dataset inputs for optimizers.

    ``data`` may be one of:
    * ``pd.DataFrame`` – single dataset
    * mapping of strings to DataFrames – keyed datasets whose names are used
      only for logging
    * list of DataFrames – treated as an ordered collection with no useful
      names
    * ``None`` – treated as an empty list

    ``labels`` must mirror ``data`` in structure when provided.  When the
    argument types mismatch or keys/lengths disagree a ``ValueError`` is raised.
    We also verify that any paired DataFrame/Series share identical indexes.

    Returns:
        ``(data_list, label_list_or_None, names)`` where ``names`` is a list of
        strings with the same order as ``data_list``.  For anonymous inputs the
        canonical name ``gentrade._defaults.KEY_OHLCV`` is used.
    """
    # Handle absence first – caller will decide if labels are required later.
    if data is None:
        return [], None, []

    # Convenience helpers
    def _check_index_match(df: pd.DataFrame, ser: pd.Series, key: str) -> None:
        if not df.index.equals(ser.index):
            raise ValueError(
                "Index mismatch between "
                f"{dataset_name}_data and {dataset_name}_labels "
                f"for key {key!r}."
            )

    # Convert mapping to ordered lists, keeping key order for names
    if isinstance(data, dict):
        if labels is not None and not isinstance(labels, dict):
            raise ValueError(
                f"When {dataset_name}_data is a dict, "
                f"{dataset_name}_labels must also be a dict."
            )
        keys = list(data.keys())
        data_list = [data[k] for k in keys]
        names = keys.copy()
        label_list: list[pd.Series] | None = None
        if labels is not None:
            if set(labels.keys()) != set(keys):
                raise ValueError(
                    f"{dataset_name}_data and {dataset_name}_labels "
                    "must have the same keys. "
                    f"got {keys!r} vs {list(labels.keys())!r}"
                )
            # preserve order of data keys
            label_list = [labels[k] for k in keys]
            # verify index alignment
            for k, df_ in data.items():
                _check_index_match(df_, labels[k], k)
        return data_list, label_list, names

    # Non-dict inputs: either DataFrame or list
    if isinstance(data, list):
        data_list = data
        names = [KEY_OHLCV] * len(data_list)
        if labels is not None:
            if not isinstance(labels, list):
                raise ValueError(
                    f"When {dataset_name}_data is a list, "
                    f"{dataset_name}_labels must also be a list."
                )
            if len(labels) != len(data_list):
                raise ValueError(
                    f"Length mismatch between {dataset_name}_data "
                    f"and {dataset_name}_labels"
                )
            for i, (df_, lab) in enumerate(zip(data_list, labels, strict=True)):
                _check_index_match(df_, lab, str(i))
        return data_list, labels if isinstance(labels, list) else None, names

    # Single DataFrame
    if isinstance(data, pd.DataFrame):
        data_list = [data]
        names = [KEY_OHLCV]
        if labels is not None:
            if not isinstance(labels, pd.Series):
                raise ValueError(
                    f"When {dataset_name}_data is a DataFrame, "
                    f"{dataset_name}_labels must be a Series."
                )
            _check_index_match(data, labels, KEY_OHLCV)
            return data_list, [labels], names
        return data_list, None, names

    # We shouldn't reach here because typing restricts input, but guard anyway
    raise TypeError(f"Unsupported type for {dataset_name}_data: {type(data)}")


class BaseOptimizer(ABC):
    """Abstract base for genetic programming optimizers.

    This class provides shared orchestration logic for all gentrade optimizers:
    seeding, multiprocessing pool management, callbacks, statistics logging,
    and the evolutionary algorithm loop. Subclasses implement domain-specific
    details like primitive set construction, toolbox wiring, and evaluator
    creation.

    The optimizer works with :class:`TreeIndividual` instances (wrapping
    GP trees with fitness), replacing bare :class:`deap.gp.PrimitiveTree`
    objects to improve code clarity and manage fitness consistently.

    Attributes:
        population_: Final evolved population (list of :class:`TreeIndividual`).
        logbook_: DEAP :class:`deap.tools.Logbook` with per-generation statistics.
        hall_of_fame_: :class:`deap.tools.HallOfFame` containing best individuals
            found during the run.
        pset_: Constructed :class:`deap.gp.PrimitiveSetTyped` (useful for
            compiling and executing individuals post-optimization).
        best_individual_: The best individual found (updated per generation).
        current_generation_: The most recently completed generation number.
    """

    def __init__(
        self,
        *,
        metrics: tuple[Metric, ...],
        mu: int = 200,
        lambda_: int = 400,
        generations: int = 30,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        hof_size: int = 5,
        n_jobs: int = 1,
        seed: int | None = None,
        verbose: bool = True,
        validation_interval: int = 1,
        metrics_val: tuple[Metric, ...] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Initialize the base optimizer.

        Args:
            metrics: Ordered tuple of metric configs for training fitness.
            mu: Parent population size.
            lambda_: Offspring population size.
            generations: Number of generations to evolve.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            hof_size: Hall of fame size (single-objective only).
            n_jobs: Number of worker processes for evaluation.
            seed: Random seed for reproducibility.
            verbose: Print per-generation stats.
            validation_interval: Run validation every N-th generation.
            metrics_val: Metric configs for validation phase.
            callbacks: Custom lifecycle callbacks.
        """
        self.metrics = metrics
        self.mu = mu
        self.lambda_ = lambda_
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hof_size = hof_size
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.validation_interval = validation_interval
        self.metrics_val = metrics_val
        self.callbacks = callbacks

        # Fitted attributes (set during fit)
        self.population_: list[TreeIndividual]
        self.logbook_: tools.Logbook
        self.hall_of_fame_: tools.HallOfFame
        self.pset_: gp.PrimitiveSetTyped
        self.toolbox_: base.Toolbox
        self.best_individual_: TreeIndividual | None = None
        self.current_generation_: int = 0

    @abstractmethod
    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """Construct and return the primitive set for this optimizer.

        Returns:
            A configured :class:`deap.gp.PrimitiveSetTyped` ready for
            use in GP tree generation and compilation.
        """
        ...

    @abstractmethod
    def _build_toolbox(self, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
        """Construct and return the DEAP toolbox for this optimizer.

        The toolbox should register functions for individual creation,
        evaluation, selection, crossover, and mutation, adapted to work
        with :class:`TreeIndividual` instances.

        Args:
            pset: Primitive set to use for tree generation and compilation.

        Returns:
            A configured :class:`deap.base.Toolbox`.
        """
        ...

    @abstractmethod
    def _make_evaluator(
        self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]
    ) -> IndividualEvaluator:
        """Create an :class:`IndividualEvaluator` for the given metrics.

        Args:
            pset: Primitive set used for compiling trees.
            metrics: Tuple of metrics to evaluate during optimization.

        Returns:
            An :class:`IndividualEvaluator` configured for the given
            metrics and primitive set.
        """
        ...

    def _validate_selection_objective_count(self, selection: Any) -> None:
        """Ensure selection operator matches the number of objectives.

        Args:
            selection: The DEAP selection function or object.

        Raises:
            ValueError: If selection is not compatible with metric count.
        """
        n = len(self.metrics)
        is_multi = n > 1
        sel_name = getattr(selection, "__name__", str(selection))

        if is_multi:
            # Multi-objective requires a dedicated MO selector
            if selection in SELECTION_SINGLE_OBJ:
                raise ValueError(
                    f"Optimizer has {n} metrics (multi-objective) but "
                    f"selection operator '{sel_name}' is for single-objective. "
                    "Use selNSGA2 or selSPEA2."
                )
        else:
            # Single-objective usually works with anything
            if selection in SELECTION_MULTI_OBJ:
                pass

    @abstractmethod
    def create_algorithm(
        self,
        worker_pool: "pool.Pool",
        stats: tools.Statistics,
        halloffame: tools.HallOfFame,
        val_callback: Callable[[int, int, list[Any], Any | None], None] | None,
    ) -> Algorithm:
        """Return algorithm instance to execute the evolutionary loop.

        Subclasses must return a configured :class:`Algorithm` instance that
        accepts a population list and returns ``(population, logbook)``.

        Args:
            worker_pool: Multiprocessing pool for parallel individual evaluation.
            stats: DEAP statistics object for logging per-generation metrics.
            halloffame: Hall of fame tracking best individuals.
            val_callback: Optional callback invoked after each generation.

        Returns:
            A configured :class:`Algorithm` instance ready to call ``run()``.
        """
        ...

    def fit(
        self,
        X: DataInput,
        y: LabelInput = None,
        X_val: DataInput = None,
        y_val: LabelInput = None,
    ) -> "BaseOptimizer":
        """Run GP evolution on training data and return self.

        Args:
            X: Training OHLCV data. Accepts DataFrame, list of DataFrames,
                or dict mapping string keys to DataFrames.
            y: Training labels. Required when classification metrics are present.
                Must mirror X in structure.
            X_val: Validation OHLCV data. When provided, a ValidationCallback
                is added automatically.
            y_val: Validation labels. Required when X_val is provided and
                classification metrics_val are used.

        Returns:
            self (for chaining).

        Raises:
            ValueError: When required labels are absent, data is invalid,
                or selection/objective count mismatches.
        """
        # 1. Normalize and validate datasets
        train_data_list, train_labels_list, _ = _normalize_data_and_labels(
            X, y, "train"
        )
        val_data_list: list[pd.DataFrame] = []
        val_labels_list: list[pd.Series] | None = None
        val_names: list[str] = []

        if X_val is not None:
            val_data_list, val_labels_list, val_names = _normalize_data_and_labels(
                X_val, y_val, "val"
            )

        # 2. Seed RNG
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 3. Validate data/config consistency
        train_needs_labels = any(
            isinstance(m, ClassificationMetricBase) for m in self.metrics
        )
        if train_needs_labels and y is None:
            raise ValueError(
                "y (train_labels) must be provided when classification metrics are "
                "included. Compute labels outside fit() and pass them in."
            )

        # Determine validation metrics
        val_metrics = self.metrics_val if self.metrics_val is not None else self.metrics

        if X_val is not None:
            val_needs_labels = any(
                isinstance(m, ClassificationMetricBase) for m in val_metrics
            )
            if val_needs_labels and y_val is None:
                raise ValueError(
                    "y_val must be provided when X_val is used with "
                    "classification metrics."
                )

        is_multiobjective = len(self.metrics) > 1

        # 4. Build pset
        self.pset_ = self._build_pset()
        if self.verbose:
            logger.info(
                "Created pset with %d primitives and %d terminals",
                self.pset_.prims_count,
                self.pset_.terms_count,
            )

        # 5. Build toolbox (this also validates selection/objective count)
        self.toolbox_ = self._build_toolbox(self.pset_)

        # 6. Build evaluators
        evaluator = self._make_evaluator(self.pset_, self.metrics)

        val_evaluator: IndividualEvaluator | None = None
        if val_data_list:
            val_evaluator = self._make_evaluator(self.pset_, val_metrics)

        # 7. Create pool
        pool_obj = create_pool(
            self.n_jobs,
            evaluator=evaluator,
            train_data=train_data_list,
            train_labels=train_labels_list,
        )

        # 8. Build active callbacks
        _active_callbacks: list[Callback] = list(self.callbacks or [])

        if val_data_list and val_evaluator is not None:
            val_callback = ValidationCallback(
                val_data=val_data_list,
                val_labels=val_labels_list,
                val_evaluator=val_evaluator,
                val_names=val_names,
                interval=self.validation_interval,
            )
            _active_callbacks.append(val_callback)

        # 9. Call on_fit_start for all callbacks
        for cb in _active_callbacks:
            cb.on_fit_start(self)

        # 10. Setup stats and HoF
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        hof: tools.HallOfFame
        if is_multiobjective:
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            hof = tools.ParetoFront()
        else:
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            hof = tools.HallOfFame(self.hof_size)

        # 11. Create initial population
        pop = self.toolbox_.population(n=self.mu)
        if self.verbose:
            logger.info("Created initial population of %d individuals", len(pop))

        # 12. Build generation callback closure
        def _gen_callback(
            gen: int,
            ngen: int,
            population: list[Any],
            best_ind: Any | None = None,
        ) -> None:
            self.current_generation_ = gen
            self.best_individual_ = cast(TreeIndividual, best_ind)
            for cb in _active_callbacks:
                cb.on_generation_end(self, gen, population, best_ind)

        # 13. Run evolution
        if self.verbose:
            print("=== GP Evolution Run ===")
            print(f"Seed: {self.seed}")
            print(f"Metrics: {[type(m).__name__ for m in self.metrics]}")
            print(
                f"Evolution: mu={self.mu}, λ={self.lambda_}, "
                f"gen={self.generations}, cxpb={self.cxpb}, mutpb={self.mutpb}"
            )
            print("-" * 60)

        try:
            algorithm = self.create_algorithm(pool_obj, stats, hof, _gen_callback)
            pop, logbook = algorithm.run(pop)
        finally:
            pool_obj.close()
            pool_obj.join()

        # 14. Store fitted attributes
        self.population_ = pop
        self.logbook_ = logbook
        self.hall_of_fame_ = hof

        # 15. Call on_fit_end for all callbacks
        for cb in _active_callbacks:
            cb.on_fit_end(self)

        if self.verbose:
            print("-" * 60)
            print("=== Results ===")
            best = hof[0]
            print(f"Best individual fitness: {best.fitness.values}")
            print(f"Best individual: {str(best.tree)[:100]}...")

        return self
