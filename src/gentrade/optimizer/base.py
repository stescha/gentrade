"""Base optimizer class and data normalization utilities.

Contains the ``BaseOptimizer`` ABC which provides shared orchestration logic
for all GP optimizers. This module documents support for both single-tree
and pair-tree optimization flows: subclasses implement primitive-set
construction, toolbox wiring, and evaluator creation for either
``TreeIndividual`` or ``PairTreeIndividual`` workflows.
"""

import logging
import random
from abc import ABC, abstractmethod
from multiprocessing import pool
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from deap import base, gp, tools

from gentrade._defaults import (
    KEY_OHLCV,
    SELECTION_MULTI_OBJ,
    SELECTION_SINGLE_OBJ,
)
from gentrade.callbacks import Callback, ValidationCallback
from gentrade.eval_ind import BaseEvaluator
from gentrade.eval_pop import create_pool
from gentrade.individual import PairTreeIndividual, TreeIndividual
from gentrade.types import Algorithm, Metric

logger = logging.getLogger(__name__)

# Type aliases for data inputs
DataInput = pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None
LabelInput = pd.Series | dict[str, pd.Series] | list[pd.Series] | None


def _normalize_data_and_labels(
    data: DataInput,
    entry_labels: LabelInput,
    exit_labels: LabelInput,
    dataset_name: str,
) -> tuple[
    list[pd.DataFrame], list[pd.Series] | None, list[pd.Series] | None, list[str]
]:
    """Normalise and validate dataset inputs for optimizers.

    This variant accepts two optional label collections (entry and exit) and
    returns normalized lists for both.  The rules are identical to the prior
    single-label helper: labels must mirror the structure of `data` (dict,
    list, or single DataFrame) and, when present, must share the same index as
    their corresponding DataFrame.

    Returns:
        (data_list, entry_label_list_or_None, exit_label_list_or_None, names)
    """
    # Handle absence first – caller will decide if labels are required later.
    if data is None:
        return [], None, None, []

    # Convenience helpers
    def _check_index_match(
        df: pd.DataFrame, ser: pd.Series, key: str, label_kind: str
    ) -> None:
        if not df.index.equals(ser.index):
            raise ValueError(
                "Index mismatch between "
                f"{dataset_name}_data and {dataset_name}_{label_kind} "
                f"for key {key!r}."
            )

    # Convert mapping to ordered lists, keeping key order for names
    if isinstance(data, dict):
        if entry_labels is not None and not isinstance(entry_labels, dict):
            raise ValueError(
                f"When {dataset_name}_data is a dict, "
                f"{dataset_name}_entry_labels must also be a dict."
            )
        if exit_labels is not None and not isinstance(exit_labels, dict):
            raise ValueError(
                f"When {dataset_name}_data is a dict, "
                f"{dataset_name}_exit_labels must also be a dict."
            )
        keys = list(data.keys())
        data_list = [data[k] for k in keys]
        names = keys.copy()

        entry_list: list[pd.Series] | None = None
        exit_list: list[pd.Series] | None = None

        if entry_labels is not None:
            if set(entry_labels.keys()) != set(keys):
                raise ValueError(
                    f"{dataset_name}_data and {dataset_name}_entry_labels "
                    "must have the same keys. "
                    f"got {keys!r} vs {list(entry_labels.keys())!r}"
                )
            entry_list = [entry_labels[k] for k in keys]

        if exit_labels is not None:
            if set(exit_labels.keys()) != set(keys):
                raise ValueError(
                    f"{dataset_name}_data and {dataset_name}_exit_labels "
                    "must have the same keys. "
                    f"got {keys!r} vs {list(exit_labels.keys())!r}"
                )
            exit_list = [exit_labels[k] for k in keys]

        # verify index alignment for any provided labels
        for k in keys:
            df_ = data[k]
            if entry_labels is not None:
                _check_index_match(df_, entry_labels[k], k, "entry_labels")
            if exit_labels is not None:
                _check_index_match(df_, exit_labels[k], k, "exit_labels")

        return data_list, entry_list, exit_list, names

    # Non-dict inputs: either DataFrame or list
    if isinstance(data, list):
        data_list = data
        names = [KEY_OHLCV] * len(data_list)

        entry_list = None
        exit_list = None

        if entry_labels is not None:
            if not isinstance(entry_labels, list):
                raise ValueError(
                    f"When {dataset_name}_data is a list, "
                    f"{dataset_name}_entry_labels must also be a list."
                )
            if len(entry_labels) != len(data_list):
                raise ValueError(
                    f"Length mismatch between {dataset_name}_data "
                    f"and {dataset_name}_entry_labels"
                )
            entry_list = entry_labels

        if exit_labels is not None:
            if not isinstance(exit_labels, list):
                raise ValueError(
                    f"When {dataset_name}_data is a list, "
                    f"{dataset_name}_exit_labels must also be a list."
                )
            if len(exit_labels) != len(data_list):
                raise ValueError(
                    f"Length mismatch between {dataset_name}_data "
                    f"and {dataset_name}_exit_labels"
                )
            exit_list = exit_labels

        # verify index alignment for any provided labels
        if entry_list is not None:
            for i, (df_, lab) in enumerate(zip(data_list, entry_list, strict=True)):
                _check_index_match(df_, lab, str(i), "entry_labels")
        if exit_list is not None:
            for i, (df_, lab) in enumerate(zip(data_list, exit_list, strict=True)):
                _check_index_match(df_, lab, str(i), "exit_labels")

        return data_list, entry_list, exit_list, names

    # Single DataFrame
    if isinstance(data, pd.DataFrame):
        data_list = [data]
        names = [KEY_OHLCV]

        entry_list = None
        exit_list = None

        if entry_labels is not None:
            if not isinstance(entry_labels, pd.Series):
                raise ValueError(
                    f"When {dataset_name}_data is a DataFrame, "
                    f"{dataset_name}_entry_labels must be a Series."
                )
            _check_index_match(data, entry_labels, KEY_OHLCV, "entry_labels")
            entry_list = [entry_labels]

        if exit_labels is not None:
            if not isinstance(exit_labels, pd.Series):
                raise ValueError(
                    f"When {dataset_name}_data is a DataFrame, "
                    f"{dataset_name}_exit_labels must be a Series."
                )
            _check_index_match(data, exit_labels, KEY_OHLCV, "exit_labels")
            exit_list = [exit_labels]

        return data_list, entry_list, exit_list, names

    # We shouldn't reach here because typing restricts input, but guard anyway
    raise TypeError(f"Unsupported type for {dataset_name}_data: {type(data)}")


class BaseOptimizer(ABC):
    """Abstract base for genetic programming optimizers.

    This class provides shared orchestration logic for all gentrade optimizers:
    seeding, multiprocessing pool management, callbacks, statistics logging,
    and the evolutionary algorithm loop. Subclasses provide the domain-specific
    details such as primitive-set construction, toolbox wiring, and evaluator
    creation.

    Implementations may operate on either single-tree individuals
    (:class:`TreeIndividual`) or two-tree / pair individuals
    (:class:`PairTreeIndividual`) depending on the subclass. Concrete
    subclasses (e.g. :class:`TreeOptimizer` and :class:`PairTreeOptimizer`)
    document which individual representation they produce.

    Attributes:
        population_: Final evolved population; contains either
            :class:`TreeIndividual` or :class:`PairTreeIndividual` instances
            depending on the optimizer subclass.
        logbook_: DEAP :class:`deap.tools.Logbook` with per-generation
            statistics.
        hall_of_fame_: :class:`deap.tools.HallOfFame` containing best
            individuals found during the run.
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
        self.demes_: list[list[Any]] | None = None

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
    ) -> BaseEvaluator[Any]:
        """Create a :class:`BaseEvaluator` for the given metrics.

        Args:
            pset: Primitive set used for compiling trees.
            metrics: Tuple of metrics to evaluate during optimization.

        Returns:
            A :class:`BaseEvaluator` configured for the given
            metrics and primitive set.
        """
        ...

    @abstractmethod
    def create_algorithm(
        self,
        worker_pool: pool.Pool,
        stats: tools.Statistics,
        halloffame: tools.HallOfFame,
        val_callback: Callable[..., None] | None,
    ) -> "Algorithm[Any]":
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

    def fit(
        self,
        X: DataInput,
        X_val: DataInput = None,
        entry_label: LabelInput = None,
        exit_label: LabelInput = None,
        entry_label_val: LabelInput = None,
        exit_label_val: LabelInput = None,
    ) -> "BaseOptimizer":
        """Run GP evolution on training data and return self.

        Args:
            X: Training OHLCV data. Accepts DataFrame, list of DataFrames,
                or dict mapping string keys to DataFrames.
            entry_label: Entry signal ground truth labels. Required when
                classification metrics are present and trade_side='buy', or
                when backtest metrics are present and trade_side='sell'.
                Must mirror X in structure.
            exit_label: Exit signal ground truth labels. Required when
                classification metrics are present and trade_side='sell', or
                when backtest metrics are present and trade_side='buy'.
                Must mirror X in structure.
            X_val: Validation OHLCV data. When provided, a ValidationCallback
                is added automatically.
            entry_label_val: Validation entry labels. Required when X_val is
                provided with classification metrics and trade_side='buy'.
            exit_label_val: Validation exit labels. Required when X_val is
                provided with classification metrics and trade_side='sell'.

        Returns:
            self (for chaining).

        Raises:
            ValueError: When required labels are absent, data is invalid,
                or selection/objective count mismatches.
        """
        # 1. Normalize and validate datasets (single call for data+labels)
        (
            train_data_list,
            train_entry_list,
            train_exit_list,
            _,
        ) = _normalize_data_and_labels(X, entry_label, exit_label, "train")

        val_data_list: list[pd.DataFrame] = []
        val_entry_list: list[pd.Series] | None = None
        val_exit_list: list[pd.Series] | None = None
        val_names: list[str] = []

        if X_val is not None:
            (
                val_data_list,
                val_entry_list,
                val_exit_list,
                val_names,
            ) = _normalize_data_and_labels(
                X_val, entry_label_val, exit_label_val, "val"
            )

        # 2. Seed RNG
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 3. Early validation of val labels before evolution starts.
        # This gives a clear error message instead of failing mid-evolution.
        # Determine validation metrics
        val_metrics = self.metrics_val if self.metrics_val is not None else self.metrics

        # Check if val labels are needed for val metrics
        if val_data_list:
            # Import inside function to avoid circular imports at module level
            from gentrade.classification_metrics import (
                ClassificationMetricBase,  # noqa: PLC0415
            )

            val_needs_classification = any(
                isinstance(m, ClassificationMetricBase) for m in val_metrics
            )
            trade_side = getattr(self, "trade_side", "buy")

            if val_needs_classification:
                if trade_side == "buy" and val_entry_list is None:
                    raise ValueError(
                        "entry_label_val must be provided when X_val is supplied "
                        "with classification metrics and trade_side='buy'."
                    )
                if trade_side == "sell" and val_exit_list is None:
                    raise ValueError(
                        "exit_label_val must be provided when X_val is supplied "
                        "with classification metrics and trade_side='sell'."
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

        val_evaluator: BaseEvaluator[Any] | None = None
        if val_data_list:
            val_evaluator = self._make_evaluator(self.pset_, val_metrics)

        # Store evaluator and train data so create_algorithm can access them
        # (needed by island mode, which doesn't use a pool).
        self._fit_evaluator_: BaseEvaluator[Any] = evaluator
        self._fit_train_data_: list[pd.DataFrame] = train_data_list
        self._fit_train_entry_labels_: list[pd.Series] | None = train_entry_list
        self._fit_train_exit_labels_: list[pd.Series] | None = train_exit_list

        # 7. Create pool
        pool_obj = create_pool(
            self.n_jobs,
            evaluator=evaluator,
            train_data=train_data_list,
            train_entry_labels=train_entry_list,
            train_exit_labels=train_exit_list,
        )

        # 8. Build active callbacks
        _active_callbacks: list[Callback] = list(self.callbacks or [])

        if val_data_list and val_evaluator is not None:
            val_callback = ValidationCallback(
                val_data=val_data_list,
                val_entry_labels=val_entry_list,
                val_exit_labels=val_exit_list,
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
            island_id: int | None = None,
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

        # Store per-island populations when algorithm supports it
        if hasattr(algorithm, "demes_") and algorithm.demes_ is not None:
            self.demes_ = algorithm.demes_
        else:
            self.demes_ = [pop]

        # 15. Call on_fit_end for all callbacks
        for cb in _active_callbacks:
            cb.on_fit_end(self)

        if self.verbose:
            print("-" * 60)
            print("=== Results ===")
            best = hof[0]
            # TODO: remove checks here.
            print(f"Best individual fitness: {best.fitness.values}")
            if isinstance(best, TreeIndividual):
                print(f"Best individual tree: {str(best.tree)[:100]}...")
            elif isinstance(best, PairTreeIndividual):
                print(f"Best buy tree: {str(best.buy_tree)[:100]}...")
                print(f"Best sell tree: {str(best.sell_tree)[:100]}...")
        return self
