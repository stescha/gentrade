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

4.  **Determinism**:
    -   **Always use fixed seeds** for tests involving randomness (especially GP evolution).
    -   **Avoid exact float comparisons** for fitness across environments if unstable.
    -   Verify determinism via **structural properties** (e.g., population size, tree depth/size distribution) rather than fragile float values.
    -   Assert **invariants**: "Best fitness in final generation >= best fitness in initial generation" (elitism check).

## Example: Structure Check

```python
import pytest
from gentrade.evolve import run_evolution
from gentrade.data import generate_synthetic_ohlcv

@pytest.mark.e2e
class TestEvolutionStructure:
    """Verifies evolution pipeline adherence to config."""

    def test_run_completes_structure(self, cfg_e2e_quick):
        # generate data once and pass it to the API
        df = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        pop, logbook, hof = run_evolution(cfg_e2e_quick, df)
        # Structural assertions are robust
        assert len(pop) == cfg_e2e_quick.evolution.mu
        assert len(logbook) == cfg_e2e_quick.evolution.generations + 1
```
