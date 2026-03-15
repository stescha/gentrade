# Session B: PairTreeIndividual & PairTreeOptimizer Skeleton

## Branching Policy (Important)

- Use only local branches for development; do not pull or push from remote.
- Base branch to branch from: `feat/session-a/base-eval-tree-opt`

## Required Reading

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format |
| `.github/commands/pr-description.md` | PR description format (if creating PR) |
| `.github/instructions/python.instructions.md` | Type hints, import order, naming conventions |
| `.github/instructions/docstrings.instructions.md` | Google-style docstring format |
| `.github/instructions/gentrade.instructions.md` | Project architecture, strategy paradigms |
| `.github/instructions/testing.instructions.md` | Test structure, markers, determinism rules |
| `.github/instructions/deap-info.instructions.md` | DEAP reference for operators and primitives |
| `pyproject.toml` | Python version (≥3.11,<3.13), dependencies, test framework (pytest) |

## Goal

Implement `PairTreeIndividual` (a two-tree individual container) and `PairTreeOptimizer` (a skeleton optimizer) to support pair-strategy evolution. Session A must be merged first. `PairTreeOptimizer._make_evaluator` intentionally raises `NotImplementedError` pending Session C's `PairEvaluator` wiring. All code must follow gentrade patterns: type hints, Google-style docstrings, and atomic commits per `.github/commands/commit-messages.md`.

---

## Files to Read Before Coding

These files establish patterns and existing code that the new code must align with.

| File | Why |
|---|---|
| `src/gentrade/optimizer/individual.py` | `TreeIndividualBase` and `TreeIndividual` implementation; understand `apply_operators` wrapper and fitness caching |
| `src/gentrade/optimizer/tree.py` | `BaseTreeOptimizer` structure, `_make_individual` pattern, `_build_toolbox`, and `_make_evaluator` method signature |
| `src/gentrade/optimizer/__init__.py` | Current export list; confirm what is public |
| `tests/test_individual_evaluator.py` | Fixture patterns: `pset`, `df`, `valid_individual`; test class structure and assertions |
| `tests/conftest.py` | Shared fixtures for pytest; understand how `pset_medium`, `opt_unit` are defined |
| `tests/test_optimizer_unit.py` | Unit test patterns for optimizers; understand how toolbox and individuals are tested |
| `src/gentrade/eval_ind.py` | Import `BaseEvaluator` and understand evaluator interface (for `_make_evaluator` placeholder) |

---

## Detailed Implementation Steps

### Step 1 — Add `PairTreeIndividual` to `optimizer/individual.py`

**File**: `src/gentrade/optimizer/individual.py`

Add the new class **after** `TreeIndividual` class definition (around line 120).

```python
class PairTreeIndividual(TreeIndividualBase):
    """A GP individual containing exactly two primitive trees: buy and sell.

    The first tree (index 0) generates entry signals; the second (index 1)
    generates exit signals. Both trees share the same primitive set.

    Attributes:
        buy_tree: The entry-signal tree (``self[0]``).
        sell_tree: The exit-signal tree (``self[1]``).
    """

    def __init__(
        self,
        content: Iterable[gp.PrimitiveTree],
        weights: Tuple[float, ...],
    ) -> None:
        """Initialize a pair-tree individual.

        Args:
            content: Iterable of exactly two :class:`deap.gp.PrimitiveTree`
                instances: buy tree first, sell tree second.
            weights: Fitness objective weights (length determines objective count).

        Raises:
            ValueError: If ``content`` does not contain exactly two trees.
        """
        trees = list(content)
        if len(trees) != 2:
            raise ValueError(
                f"PairTreeIndividual requires exactly 2 trees, got {len(trees)}."
            )
        super().__init__(trees, weights)

    @property
    def buy_tree(self) -> gp.PrimitiveTree:
        """Return the buy (entry) tree at index 0."""
        return self[0]

    @property
    def sell_tree(self) -> gp.PrimitiveTree:
        """Return the sell (exit) tree at index 1."""
        return self[1]
```

---

### Step 2 — Add `PairTreeOptimizer` to `optimizer/tree.py`

**File**: `src/gentrade/optimizer/tree.py`

Add the new class **after** `TreeOptimizer` class definition (at end of file, around line 289).

```python
class PairTreeOptimizer(BaseTreeOptimizer):
    """Genetic programming optimizer for pair-tree individuals.

    Each individual contains two trees: a buy (entry) tree and a sell
    (exit) tree. Both trees are evolved using the same primitive set.
    Genetic operators (crossover, mutation) are applied independently
    to each tree position via the :func:`apply_operators` wrapper.

    Note:
        ``fit()`` is not yet functional until Session C wires ``PairEvaluator``.
    """

    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> PairTreeIndividual:
        """Create a pair-tree individual with two independently generated trees."""
        buy_nodes = tree_gen_func()
        sell_nodes = tree_gen_func()
        return PairTreeIndividual(
            [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)],
            weights,
        )

    def _make_evaluator(
        self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]
    ) -> BaseEvaluator:
        raise NotImplementedError(
            "PairTreeOptimizer._make_evaluator is not yet implemented. "
            "PairEvaluator will be wired in Session C."
        )
```

**Import note**: Ensure `PairTreeIndividual` is imported at the top of `tree.py`. Add this to the imports from `gentrade.optimizer.individual`:

```python
from gentrade.optimizer.individual import (
    PairTreeIndividual,  # Add this
    TreeIndividual,
    apply_operators,
)
```

Also ensure `BaseEvaluator` is imported:
```python
from gentrade.eval_ind import BaseEvaluator, IndividualEvaluator, TradeSide
```

---

### Step 3 — Update `optimizer/__init__.py` exports

**File**: `src/gentrade/optimizer/__init__.py`

Add `PairTreeIndividual` and `PairTreeOptimizer` to the public API. Locate the current `__all__` export list and add:

```python
from gentrade.optimizer.individual import PairTreeIndividual
from gentrade.optimizer.tree import PairTreeOptimizer

__all__ = [
    # ... existing exports ...
    "PairTreeIndividual",
    "PairTreeOptimizer",
]
```

Place the imports in the appropriate section (after other individual/optimizer imports) and add the names to `__all__` in alphabetical order with existing names.

---

### Step 4 — Add conftest fixture for pair test trees

**File**: `tests/conftest.py`

Add a new fixture **after** existing tree/individual fixtures (around line 100).

```python
@pytest.fixture
def pair_individual() -> PairTreeIndividual:
    """Minimal valid GP pair individual: buy tree is gt(open, close), sell tree is lt(open, close)."""
    from gentrade.optimizer.individual import PairTreeIndividual
    from gentrade.pset.pset_types import BooleanSeries, NumericSeries

    # Buy tree: gt(open, close)
    buy_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    # Sell tree: lt(open, close)
    sell_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="lt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return PairTreeIndividual([buy_tree, sell_tree], weights=(1.0,))
```

Ensure `PairTreeIndividual` is imported at the top of `conftest.py`:
```python
from gentrade.optimizer import PairTreeIndividual
```

---

### Step 5 — Create `tests/test_pair_individual.py` — Unit tests

**File**: `tests/test_pair_individual.py` (new file)

```python
"""Tests for PairTreeIndividual.

Verifies:
- Construction with exactly 2 trees succeeds; invalid counts raise ValueError.
- buy_tree and sell_tree properties return correct tree indices.
- Fitness is created with correct objective count.
- Pickle round-trip preserves individual structure.
- apply_operators wrapper handles pair individuals correctly for crossover and mutation.
"""

import pickle

import pytest
from deap import gp as deap_gp

from gentrade.optimizer.individual import PairTreeIndividual, apply_operators
from gentrade.pset.pset_types import BooleanSeries, NumericSeries


@pytest.fixture
def pair_individual() -> PairTreeIndividual:
    """Minimal valid pair individual for testing."""
    buy_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    sell_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="lt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return PairTreeIndividual([buy_tree, sell_tree], weights=(1.0,))


@pytest.fixture
def pair_individual_multi_obj() -> PairTreeIndividual:
    """Pair individual with multi-objective fitness."""
    buy_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    sell_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="lt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return PairTreeIndividual([buy_tree, sell_tree], weights=(1.0, -1.0))


@pytest.mark.unit
class TestPairTreeIndividual:
    """Unit tests for PairTreeIndividual construction and properties."""

    def test_construction_with_two_trees_succeeds(self, pair_individual: PairTreeIndividual) -> None:
        """Constructing with exactly 2 trees should succeed."""
        assert len(pair_individual) == 2
        assert isinstance(pair_individual[0], deap_gp.PrimitiveTree)
        assert isinstance(pair_individual[1], deap_gp.PrimitiveTree)

    def test_construction_with_one_tree_raises_valueerror(self, pair_individual: PairTreeIndividual) -> None:
        """Constructing with 1 tree should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 1"):
            PairTreeIndividual([pair_individual[0]], weights=(1.0,))

    def test_construction_with_three_trees_raises_valueerror(self, pair_individual: PairTreeIndividual) -> None:
        """Constructing with 3 trees should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 3"):
            PairTreeIndividual(
                [pair_individual[0], pair_individual[1], pair_individual[0]],
                weights=(1.0,),
            )

    def test_construction_with_zero_trees_raises_valueerror(self) -> None:
        """Constructing with 0 trees should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 0"):
            PairTreeIndividual([], weights=(1.0,))

    def test_buy_tree_property_returns_index_zero(self, pair_individual: PairTreeIndividual) -> None:
        """buy_tree property should return self[0]."""
        assert pair_individual.buy_tree is pair_individual[0]

    def test_sell_tree_property_returns_index_one(self, pair_individual: PairTreeIndividual) -> None:
        """sell_tree property should return self[1]."""
        assert pair_individual.sell_tree is pair_individual[1]

    def test_fitness_single_objective(self, pair_individual: PairTreeIndividual) -> None:
        """Fitness should have 1 objective when weights=(1.0,)."""
        assert len(pair_individual.fitness.weights) == 1
        assert pair_individual.fitness.weights[0] == 1.0

    def test_fitness_multi_objective(self, pair_individual_multi_obj: PairTreeIndividual) -> None:
        """Fitness should have 2 objectives when weights=(1.0, -1.0)."""
        assert len(pair_individual_multi_obj.fitness.weights) == 2
        assert pair_individual_multi_obj.fitness.weights == (1.0, -1.0)

    def test_pickle_roundtrip(self, pair_individual: PairTreeIndividual) -> None:
        """Individual should survive pickle.dumps/loads cycle."""
+        serialized = pickle.dumps(pair_individual)
+        restored = pickle.loads(serialized)
+        assert len(restored) == 2
+        assert restored.buy_tree is restored[0]
+        assert restored.sell_tree is restored[1]
+        # Fitness weights should match
+        assert restored.fitness.weights == pair_individual.fitness.weights
+
+
+@pytest.mark.unit
+class TestApplyOperatorsWithPair:
+    """Unit tests for apply_operators wrapper on pair individuals."""
+
+    def test_apply_operators_mutation_on_pair(self, pair_individual: PairTreeIndividual) -> None:
+        """Mutation via apply_operators should modify both trees independently."""
+        # Wrap gp.mutShrink with apply_operators
+        mut_op = apply_operators(deap_gp.mutShrink)
+        original_buy_str = str(pair_individual[0])
+        original_sell_str = str(pair_individual[1])
+
+        # Apply mutation
+        (mutated,) = mut_op(pair_individual)
+
+        # Check that the individual is still a pair with 2 trees
+        assert len(mutated) == 2
+        # At least one tree should have changed (with high probability for small trees)
+        # Note: mutShrink may not always mutate, so we just check structure is preserved
+        assert isinstance(mutated[0], deap_gp.PrimitiveTree)
+        assert isinstance(mutated[1], deap_gp.PrimitiveTree)
+
+    def test_apply_operators_crossover_on_pair(
+        self, pair_individual: PairTreeIndividual
+    ) -> None:
+        """Crossover via apply_operators should cross trees at corresponding positions."""
+        # Create a second pair individual
+        second_pair = PairTreeIndividual(
+            [pair_individual[0].copy(), pair_individual[1].copy()],
+            weights=pair_individual.fitness.weights,
+        )
+
+        # Wrap gp.cxOnePoint with apply_operators
+        cx_op = apply_operators(deap_gp.cxOnePoint)
+
+        # Apply crossover
+        ind1, ind2 = cx_op(pair_individual, second_pair)
+
+        # Check that both individuals still have 2 trees
+        assert len(ind1) == 2
+        assert len(ind2) == 2
+        assert isinstance(ind1[0], deap_gp.PrimitiveTree)
+        assert isinstance(ind1[1], deap_gp.PrimitiveTree)
+        assert isinstance(ind2[0], deap_gp.PrimitiveTree)
+        assert isinstance(ind2[1], deap_gp.PrimitiveTree)
+```
+
+---
+
+### Step 6 — Create `tests/test_pair_optimizer.py` — Unit tests for PairTreeOptimizer
+
+**File**: `tests/test_pair_optimizer.py` (new file)
+
+```python
+"""Tests for PairTreeOptimizer skeleton.
+
+Verifies:
+- PairTreeOptimizer can be instantiated.
+- _make_individual creates PairTreeIndividual instances with 2 trees.
+- _make_evaluator raises NotImplementedError with clear message.
+- toolbox.individual() produces PairTreeIndividual instances.
+"""
+
+import pytest
+from deap import gp as deap_gp
+
+from gentrade.classification_metrics import F1Metric
+from gentrade.minimal_pset import create_pset_default_medium
+from gentrade.optimizer import PairTreeOptimizer
+from gentrade.optimizer.individual import PairTreeIndividual
+
+
+@pytest.fixture
def pset_medium() -> deap_gp.PrimitiveSetTyped:
+    """Medium pset for optimizer tests."""
+    return create_pset_default_medium()
+
+
+@pytest.mark.unit
+class TestPairTreeOptimizerSkeleton:
+    """Unit tests for PairTreeOptimizer."""
+
+    def test_optimizer_can_be_instantiated(self, pset_medium: deap_gp.PrimitiveSetTyped) -> None:
+        """PairTreeOptimizer should instantiate without error."""
+        opt = PairTreeOptimizer(
+            pset=pset_medium,
+            metrics=(F1Metric(),),
+            mu=10,
+            lambda_=20,
+            generations=2,
+            seed=42,
+            verbose=False,
+        )
+        assert opt is not None
+
+    def test_make_individual_returns_pair_individual(
+        self, pset_medium: deap_gp.PrimitiveSetTyped
+    ) -> None:
+        """_make_individual should return a PairTreeIndividual."""
+        opt = PairTreeOptimizer(
+            pset=pset_medium,
+            metrics=(F1Metric(),),
+            mu=10,
+            lambda_=20,
+            generations=2,
+            seed=42,
+            verbose=False,
+        )
+        # Manually call _make_individual via internal toolbox
+        opt._build_pset_and_toolbox()
+        ind = opt.toolbox_.individual()
+        assert isinstance(ind, PairTreeIndividual)
+        assert len(ind) == 2
+
+    def test_make_evaluator_raises_notimplementederror(
+        self, pset_medium: deap_gp.PrimitiveSetTyped
+    ) -> None:
+        """_make_evaluator should raise NotImplementedError."""
+        opt = PairTreeOptimizer(
+            pset=pset_medium,
+            metrics=(F1Metric(),),
+            mu=10,
+            lambda_=20,
+            generations=2,
+            seed=42,
+            verbose=False,
+        )
+        with pytest.raises(
+            NotImplementedError,
+            match="PairEvaluator will be wired in Session C",
+        ):
+            opt._make_evaluator(pset_medium, (F1Metric(),))
+
+    def test_toolbox_individual_creates_pair_individuals(
+        self, pset_medium: deap_gp.PrimitiveSetTyped
+    ) -> None:
+        """toolbox.individual() should produce PairTreeIndividual instances."""
+        opt = PairTreeOptimizer(
+            pset=pset_medium,
+            metrics=(F1Metric(),),
+            mu=10,
+            lambda_=20,
+            generations=2,
+            seed=42,
+            verbose=False,
+        )
+        opt._build_pset_and_toolbox()
+        # Create a population to ensure toolbox is properly registered
+        pop = opt.toolbox_.population(n=5)
+        assert len(pop) == 5
+        for ind in pop:
+            assert isinstance(ind, PairTreeIndividual)
+            assert len(ind) == 2
+```
+
+---
+
+## Test Plan
+
+### Test data
+
+Existing test fixtures cover most needs:
+- `pset_medium`: Existing fixture for primitive set (no new data needed).
+- `pair_individual`: New fixture in `conftest.py` for reuse across pair-related tests.
+
+### Test cases: `tests/test_pair_individual.py`
+
+**Success cases:**
+| Test | Input | Expected outcome |
+|---|---|---|
+| Construction with 2 trees | `PairTreeIndividual([tree_a, tree_b], (1.0,))` | No error; `len(ind) == 2` |
+| `buy_tree` property | Access `ind.buy_tree` | Returns `tree_a` (index 0) |
+| `sell_tree` property | Access `ind.sell_tree` | Returns `tree_b` (index 1) |
+| Fitness single objective | Construct with `weights=(1.0,)` | `len(ind.fitness.weights) == 1` |
+| Fitness multi-objective | Construct with `weights=(1.0, -1.0)` | `len(ind.fitness.weights) == 2` |
+| Pickle roundtrip | `pickle.loads(pickle.dumps(ind))` | Reconstructed individual equals original; both trees and fitness preserved |
+| `apply_operators` mutation | Wrap `gp.mutShrink` with `apply_operators`; call on pair | Both trees mutated independently; structure preserved |
+| `apply_operators` crossover | Wrap `gp.cxOnePoint` with `apply_operators`; call on two pairs | `ind1[0]` crossed with `ind2[0]`; `ind1[1]` with `ind2[1]` |
+
+**Error cases:**
+| Test | Input | Expected outcome |
+|---|---|---|
+| Construction with 1 tree | `PairTreeIndividual([tree_a], (1.0,))` | `ValueError("...requires exactly 2 trees, got 1")` |
+| Construction with 3 trees | `PairTreeIndividual([a, b, c], (1.0,))` | `ValueError("...requires exactly 2 trees, got 3")` |
+| Construction with 0 trees | `PairTreeIndividual([], (1.0,))` | `ValueError("...requires exactly 2 trees, got 0")` |
+
+### Test cases: `tests/test_pair_optimizer.py`
+
+**Success cases:**
+| Test | Input/Setup | Expected outcome |
+|---|---|---|
+| Optimizer instantiation | Call `PairTreeOptimizer(...)` with valid params | No error; instance created |
+| `_make_individual` returns pair | Call `toolbox_.individual()` after build | Returns `PairTreeIndividual` with 2 trees |
+| `toolbox.population()` creates pairs | Call `toolbox_.population(n=5)` | All 5 individuals are `PairTreeIndividual` instances |
+
+**Error cases:**
+| Test | Input/Setup | Expected outcome |
+|---|---|---|
+| `_make_evaluator` raises | Call `opt._make_evaluator(pset, metrics)` | `NotImplementedError` with message mentioning "Session C" |
+
+### Commands to run (targeted)
+
+```bash
+# Unit tests for PairTreeIndividual
+poetry run pytest tests/test_pair_individual.py -v
+
+# Unit tests for PairTreeOptimizer
+poetry run pytest tests/test_pair_optimizer.py -v
+
+# Combined pair tests
+poetry run pytest tests/test_pair_individual.py tests/test_pair_optimizer.py -v
+
+# Regression: existing optimizer tests should still pass
+poetry run pytest tests/test_optimizer_unit.py -v
+
+# Type check
+poetry run mypy .
+
+# Lint
+poetry run ruff check .
+```
+
+---
+
+## Edge Cases
+
+| Scenario | Expected behavior |
+|---|---|
+| Two pair individuals crossover with `apply_operators(gp.cxOnePoint)` | Buy trees (index 0) cross; sell trees (index 1) cross independently. Both pairs still have 2 trees post-operation. |
+| Mutation on a pair with `apply_operators(gp.mutUniform)` | Both trees can be mutated; height limits enforced independently per tree. |
+| Pickle a pair individual, modify one tree, unpickle the original | Original and restored should differ if modification was in-place; fitness should be preserved. |
+| `len(ind) == 2` property of pair after operators | Minimum tree count across all individuals is used; both always have exactly 2 trees (no edge case here). |
+
+---
+
+## Files to Create / Modify
+
+| Action | File |
+|---|---|
+| **Create** | `tests/test_pair_individual.py` |
+| **Create** | `tests/test_pair_optimizer.py` |
+| **Modify** | `src/gentrade/optimizer/individual.py` (add `PairTreeIndividual`) |
+| **Modify** | `src/gentrade/optimizer/tree.py` (add `PairTreeOptimizer`, update imports) |
+| **Modify** | `src/gentrade/optimizer/__init__.py` (export `PairTreeIndividual`, `PairTreeOptimizer`) |
+| **Modify** | `tests/conftest.py` (add `pair_individual` fixture) |
+
+---
+
+## Checklist
+
+- [ ] **Step 1**: Add `PairTreeIndividual` to `optimizer/individual.py` with validation, properties, and docstring
+- [ ] **Step 2**: Add `PairTreeOptimizer` to `optimizer/tree.py` with `_make_individual` and `_make_evaluator` (raises `NotImplementedError`)
+- [ ] **Step 3**: Update `optimizer/__init__.py` to export both new classes
+- [ ] **Step 4**: Add `pair_individual` fixture to `tests/conftest.py`
+- [ ] **Step 5**: Create `tests/test_pair_individual.py` with construction, property, pickle, and `apply_operators` tests
+- [ ] **Step 6**: Create `tests/test_pair_optimizer.py` with instantiation, toolbox, and evaluator placeholder tests
+- [ ] **Step 7**: Verify imports and type hints are correct; no circular dependencies
+- [ ] Run targeted tests pass: `poetry run pytest tests/test_pair_individual.py tests/test_pair_optimizer.py -v`
+- [ ] Run regression tests pass: `poetry run pytest tests/test_optimizer_unit.py -v`
+- [ ] Type check passes: `poetry run mypy .`
+- [ ] Lint passes: `poetry run ruff check .`
+- [ ] Atomic commits following `.github/commands/commit-messages.md` format
+  - Suggested commits:
+    - `feat(optimizer): add PairTreeIndividual class`
+    - `feat(optimizer): add PairTreeOptimizer skeleton class`
+    - `chore(optimizer): export PairTreeIndividual and PairTreeOptimizer`
+    - `test(optimizer): add unit tests for PairTreeIndividual`
+    - `test(optimizer): add unit tests for PairTreeOptimizer`
+    - `test(conftest): add pair_individual fixture`
+- [ ] Verify base branch is `feat/session-a/base-eval-tree-opt` and work uses local branches only (do not pull or push)
+- [ ] All code follows gentrade patterns: type hints, Google docstrings, no `import *`, proper error handling
