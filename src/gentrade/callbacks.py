"""Callback protocols and implementations for optimizer lifecycle hooks.

Callbacks allow users to plug custom behaviour into the GP evolution
process at key stages: fit start, generation end, and fit end.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from gentrade.eval_ind import BaseEvaluator
    from gentrade.optimizer.base import BaseOptimizer


@runtime_checkable
class Callback(Protocol):
    """Lifecycle hooks for optimizer events.

    Implement this protocol to inject custom logic at specific points
    during evolution.
    """

    def on_fit_start(self, optimizer: "BaseOptimizer") -> None:
        """Called once at the start of fit(), before evolution begins."""
        ...

    def on_generation_end(
        self,
        gen: int,
        ngen: int,
        population: list[Any],
        best_ind: Any | None = None,
        island_id: int | None = None,
    ) -> None:
        """Called after each generation completes.

        Args:
            optimizer: The optimizer instance.
            gen: The 1-indexed generation number that just completed.
            population: The current population after selection.
        """
        ...

    def on_fit_end(self, optimizer: "BaseOptimizer") -> None:
        """Called once at the end of fit(), after evolution completes."""
        ...


class ValidationCallback:
    """Evaluates the best individual on validation data at configurable intervals.

    Auto-added by ``BaseOptimizer.fit()`` when validation data is provided.
    Uses train metrics if ``metrics_val`` is not provided.
    """

    def __init__(
        self,
        val_data: list[pd.DataFrame],
        val_entry_labels: list[pd.Series] | None,
        val_exit_labels: list[pd.Series] | None,
        val_evaluator: "BaseEvaluator[Any]",
        val_names: list[str],
        interval: int = 1,
    ) -> None:
        """Initialize the validation callback.

        Args:
            val_data: List of validation OHLCV DataFrames.
            val_entry_labels: Optional list of entry label Series.
            val_exit_labels: Optional list of exit label Series.
            val_evaluator: Evaluator configured with validation metrics.
            val_names: Human-readable names for each validation dataset.
            interval: Run validation every N-th generation (and always at last).
        """
        self._val_data = val_data
        self._val_entry_labels = val_entry_labels
        self._val_exit_labels = val_exit_labels
        self._val_evaluator = val_evaluator
        self._val_names = val_names
        self._interval = interval

    def on_fit_start(self, optimizer: "BaseOptimizer") -> None:
        """No-op at fit start."""
        pass

    def on_generation_end(
        self,
        gen: int,
        ngen: int,
        population: list[Any],
        best_ind: Any | None = None,
        island_id: int | None = None,
    ) -> None:
        """Evaluate best individual on validation data if interval matches.

        Runs at gen==1 (first), every N-th generation, and always at the
        last generation.

        Args:
            gen: The 1-indexed generation number that just completed.
            ngen: The total number of generations configured for the run.
            population: The current population after selection.
            best_ind: The best individual from the current population, or None.
        """
        # Run at gen 1, every Nth, and always at the last generation
        if gen != 1 and (gen - 1) % self._interval != 0 and gen != ngen:
            return
        if best_ind is None:
            print(
                f"ValidationCallback: No best_ind provided at gen {gen}, "
                "skipping validation."
            )
            return
        val_fitnesses = self._val_evaluator.evaluate(
            best_ind,
            ohlcvs=self._val_data,
            entry_labels=self._val_entry_labels,
            exit_labels=self._val_exit_labels,
            aggregate=False,
        )

        print("Validation results:")
        for fitness, name in zip(val_fitnesses, self._val_names, strict=True):
            print(f"  {name}: {fitness}")

        agg_score = self._val_evaluator.aggregate_fitness(val_fitnesses)
        print(f"  aggregated: {', '.join(map(str, agg_score))}")

    def on_fit_end(self, optimizer: "BaseOptimizer") -> None:
        """No-op at fit end."""
        pass
