---
applyTo: "tests/**/*.py"
---

# Testing Guidelines for Gentrade

**Note on imports**: the code lives in `src/gentrade` and the project is installed in editable mode via 
Poetry. Test can use normal imports like `import gentrade` or `from gentrade import ...`. 
No `sys.path` manipulation to fix imports is allowed in code or tests.



Follow these guidelines when writing tests for the `gentrade` repository.

## General Principles

1.  **Test Types**: Use `pytest` markers for categorization.
    -   `@pytest.mark.unit`: Fast, isolated tests (functions/classes). Mock external dependencies.
    -   `@pytest.mark.integration`: Tests combining multiple components.
    -   `@pytest.mark.e2e`: Full system tests (evolution pipeline). Slow, seed-controlled.
    -   Do **not** introduce new markers (e.g., `smoke`). Use the standard three.

2.  **Organization**:
    -   Group related tests strictly into **Test Classes** (`class TestMyFeature:`).
    -   Name test files explicitly (e.g., `test_evolution_smoke.py`, `test_smoke_zigzag.py`).
    -   Keep module-level docstrings clear and specific to the file's content.

3.  **Redundancy**:
    -   Avoid duplicating test logic. If a unit test covers a specific behavior (e.g., config wiring), do not re-verify it in an E2E test unless the integration itself is the target.
    -   Delete redundant tests if new tests supersede them.

4.  **Behavior-driven tests**:
    -   Test the expected contract and output behavior, not a single implementation path.
    -   For example, if a function must return a sorted list, assert that the result is sorted rather than equal to one hardcoded sorted list.
    -   If expected behavior is unclear from the code or domain, ask for clarification instead of assuming an implementation detail.

5.  **Determinism**:
    -   **Always use fixed seeds** for tests involving randomness (especially GP evolution).
    -   **Avoid exact float comparisons** for fitness across environments if unstable.
    -   Verify determinism via **structural properties** (e.g., population size, tree depth/size distribution) rather than fragile float values.
    -   Assert **invariants**: "Best fitness in final generation >= best fitness in initial generation" (elitism check).
    -   Add tests that exercise pair-tree behaviour: `PairTreeIndividual`, `PairEvaluator`, and `PairTreeOptimizer` should have unit and integration coverage. Include tests for `tree_aggregation` options (e.g., "buy", "sell", "mean", "min", "max") and validate that pair-tree populations contain exactly two trees per individual.

6.  **Config validation & error cases**:
    -   The `RunConfig` model contains several validators; write unit tests that exercise misconfigurations such as missing `metrics_val` when validation data is supplied, or wrong selection operators for multi-objective setups.
   - Do **not** test for an evaluator/metric type mismatch: the codebase no longer uses separate evaluator config classes. Instead, assert that appropriate `ValueError` messages are raised for missing `train_labels` (when classification metrics are present) or `val_labels` (when validation data is used with classification metrics).

## Example: Structure Check

```python
import pytest
from gentrade.data import generate_synthetic_ohlcv
from gentrade.optimizer import TreeOptimizer
from gentrade.classification_metrics import F1Metric
from gentrade.minimal_pset import create_pset_default_medium

@pytest.mark.e2e
class TestEvolutionStructure:
    """Verifies evolution pipeline adherence to config."""

    def test_run_completes_structure(self) -> None:
        df = generate_synthetic_ohlcv(2000, 42)
        mu = 50
        generations = 5
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=100,
            generations=generations,
            seed=42,
        )
        opt.fit(X=df)
        # Structural assertions are robust
        assert len(opt.population_) == mu
        assert len(opt.logbook_) == generations + 1
```
