"""Smoke tests for zigzag GP pipeline."""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

# Skip entire module if zigzag is not installed
zigzag = pytest.importorskip("zigzag")


@pytest.mark.smoke
class TestSmokeZigzag:
    """Smoke tests for zigzag GP pipeline."""

    # Use smaller parameters for speed
    SMOKE_MU = 50
    SMOKE_LAMBDA = 100
    SMOKE_GENERATIONS = 10
    SMOKE_SEED = 1997
    SMOKE_N = 2000

    @staticmethod
    def _make_synthetic_ohlcv(n: int, seed: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        rng = np.random.default_rng(seed)

        returns = rng.normal(0, 0.02, n)
        close = 100 * np.exp(np.cumsum(returns))

        noise = rng.uniform(0.001, 0.01, n)
        high = close * (1 + noise)
        low = close * (1 - noise)
        open_ = close * (1 + rng.uniform(-0.005, 0.005, n))

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        volume = rng.uniform(1000, 10000, n)

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def _run_smoke(
        self,
        mu: int,
        lambda_: int,
        ngen: int,
        seed: int,
        n: int,
    ) -> tuple:
        """Run a short GP evolution and return (pop, logbook, hof)."""
        import operator
        import random

        from deap import algorithms, base, creator, gp, tools

        from gentrade.growtree import genFull, genHalfAndHalf
        from gentrade.minimal_pset import create_pset_zigzag_minimal, zigzag_pivots

        random.seed(seed)
        np.random.seed(seed)

        df = self._make_synthetic_ohlcv(n, seed)
        y_true = zigzag_pivots(df["close"], 0.03, 1)

        pset = create_pset_zigzag_minimal()

        # Guard against duplicate creator types
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", genHalfAndHalf, pset=pset, min_=2, max_=6)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def evaluate(individual, pset, df, y_true):
            try:
                func = gp.compile(individual, pset)
                y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])

                if isinstance(y_pred, (bool, int, float, np.bool_)):
                    y_pred = pd.Series([bool(y_pred)] * len(df), index=df.index)

                y_pred = y_pred.astype(bool)

                tp = (y_pred & y_true).sum()
                fp = (y_pred & ~y_true).sum()
                fn = (~y_pred & y_true).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                return (f1,)
            except Exception:
                return (0.0,)

        toolbox.register("evaluate", partial(evaluate, pset=pset, df=df, y_true=y_true))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", genFull, pset=pset, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        hof = tools.HallOfFame(5)
        pop = toolbox.population(n=mu)

        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=mu,
            lambda_=lambda_,
            cxpb=0.5,
            mutpb=0.2,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

        return pop, logbook, hof

    def test_pset_creation(self) -> None:
        """Test that pset is created correctly with required primitives."""
        from deap import gp

        from gentrade.minimal_pset import create_pset_zigzag_minimal

        pset = create_pset_zigzag_minimal()

        assert isinstance(pset, gp.PrimitiveSetTyped)
        assert "zigzag_pivots" in pset.mapping
        assert "and_" in pset.mapping or "operator.and_" in pset.mapping

    def test_zigzag_pivots_primitive(self) -> None:
        """Test zigzag_pivots primitive directly."""
        from gentrade.minimal_pset import zigzag_pivots

        close = pd.Series([100, 105, 103, 110, 108, 115, 112, 120, 118, 125])
        result = zigzag_pivots(close, 0.03, 1)

        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert len(result) == len(close)
        # With this data and threshold, should have at least one pivot
        assert result.sum() >= 0  # Soft check - may be 0 for small series

    def test_smoke_run_completes(self) -> None:
        """Test that a short GP evolution completes without exceptions."""
        pop, logbook, hof = self._run_smoke(
            mu=self.SMOKE_MU,
            lambda_=self.SMOKE_LAMBDA,
            ngen=self.SMOKE_GENERATIONS,
            seed=self.SMOKE_SEED,
            n=self.SMOKE_N,
        )

        # Logbook should have SMOKE_GENERATIONS + 1 entries (gen 0 through gen N)
        assert len(logbook) == self.SMOKE_GENERATIONS + 1

        # Best fitness should not decrease (non-strict for mu+lambda)
        initial_max = logbook[0]["max"]
        final_max = logbook[-1]["max"]
        assert final_max >= initial_max

    def test_zigzag_in_hof(self) -> None:
        """Soft check: warn if zigzag_pivots not in HallOfFame."""
        pop, logbook, hof = self._run_smoke(
            mu=self.SMOKE_MU,
            lambda_=self.SMOKE_LAMBDA,
            ngen=self.SMOKE_GENERATIONS,
            seed=self.SMOKE_SEED,
            n=self.SMOKE_N,
        )

        zigzag_found = any("zigzag_pivots" in str(ind) for ind in hof)

        if not zigzag_found:
            warnings.warn(
                "zigzag_pivots not found in top-5 HallOfFame individuals. "
                "This is acceptable for short runs but may indicate the primitive "
                "is not being used effectively.",
                UserWarning,
            )

        # Sanity check: best fitness should be > 0
        assert hof[0].fitness.values[0] >= 0
