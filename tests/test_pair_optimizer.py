"""Tests for PairTreeOptimizer skeleton.

Verifies:
- PairTreeOptimizer can be instantiated.
- _make_individual creates PairTreeIndividual instances with 2 trees.
- _make_evaluator raises NotImplementedError with clear message.
- toolbox.individual() produces PairTreeIndividual instances.
"""

import pytest
from deap import gp as deap_gp

from gentrade.classification_metrics import F1Metric
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.optimizer import PairTreeOptimizer
from gentrade.optimizer.individual import PairTreeIndividual


@pytest.fixture
def pset_medium() -> deap_gp.PrimitiveSetTyped:
    """Medium pset for optimizer tests."""
    return create_pset_default_medium()


@pytest.mark.unit
class TestPairTreeOptimizerSkeleton:
    """Unit tests for PairTreeOptimizer."""

    def test_optimizer_can_be_instantiated(
        self, pset_medium: deap_gp.PrimitiveSetTyped
    ) -> None:
        """PairTreeOptimizer should instantiate without error."""
        opt = PairTreeOptimizer(
            pset=pset_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        assert opt is not None

    def test_make_individual_returns_pair_individual(
        self, pset_medium: deap_gp.PrimitiveSetTyped
    ) -> None:
        """_make_individual should return a PairTreeIndividual."""
        opt = PairTreeOptimizer(
            pset=pset_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        # Build pset and toolbox manually
        pset = opt._build_pset()
        toolbox = opt._build_toolbox(pset)
        opt.pset_ = pset
        opt.toolbox_ = toolbox
        ind = toolbox.individual()
        assert isinstance(ind, PairTreeIndividual)
        assert len(ind) == 2

    def test_make_evaluator_raises_notimplementederror(
        self, pset_medium: deap_gp.PrimitiveSetTyped
    ) -> None:
        """_make_evaluator should raise NotImplementedError."""
        opt = PairTreeOptimizer(
            pset=pset_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        with pytest.raises(
            NotImplementedError,
            match="PairEvaluator will be wired in Session C",
        ):
            opt._make_evaluator(pset_medium, (F1Metric(),))

    def test_toolbox_individual_creates_pair_individuals(
        self, pset_medium: deap_gp.PrimitiveSetTyped
    ) -> None:
        """toolbox.individual() should produce PairTreeIndividual instances."""
        opt = PairTreeOptimizer(
            pset=pset_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        pset = opt._build_pset()
        toolbox = opt._build_toolbox(pset)
        opt.pset_ = pset
        opt.toolbox_ = toolbox
        # Create a population to ensure toolbox is properly registered
        pop = toolbox.population(n=5)
        assert len(pop) == 5
        for ind in pop:
            assert isinstance(ind, PairTreeIndividual)
            assert len(ind) == 2
