---
applyTo: "src/gentrade/**/*.py"
---

# Configuration System — Development Guidelines

The configuration system in `gentrade` follows a thin-data approach. Config models are Pydantic-based containers for parameters, while behavior and orchestration are handled by the `BaseOptimizer` and its subclasses.

## Core Principle

**Config is thin data.** Behavior belongs to the optimizer (`TreeOptimizer`, `BaseOptimizer`), not the config classes.

- **No Evaluator Configs**: Evaluation logic is encapsulated in `IndividualEvaluator` and `BaseOptimizer.fit()`. Callers provide runtime metrics and backtest parameters directly to the optimizer.
- **Legacy Artifacts**: Some legacy models like `RunConfig` and its components remain in `config.py`. While they may be used for future development or as templates, the current core evolution pipeline does not depend on them.

## Component Configuration Pattern

When defining or extending component configurations (e.g., specific metrics or operator parameters), follow these rules:

### 1. Hierarchy and Type Tags

All component configs subclass `_ComponentConfig`, which provides an auto-derived `type` tag based on the class name.

```python
class _ComponentConfig(BaseModel):
    @computed_field
    @property
    def type(self) -> str:
        return _to_snake_case(self.__class__.__name__)
```

### 2. Functional Decoupling

Functions and operators are referenced via `ClassVar` or looked up via module-level dictionaries. They are never serialized.

```python
class UniformMutationConfig(MutationConfigBase):
    func: ClassVar[Callable] = gp.mutUniform
    _requires_pset: ClassVar[bool] = True
```

### 3. Parameters as Fields

Configurable parameters must be defined as Pydantic `Field` objects with appropriate validation.

```python
class TournamentSelectionConfig(SingleObjectiveSelectionConfigBase):
    tournsize: int = Field(3, ge=2)
```

## Optimizer Integration

The optimizers consume these configs to wire the DEAP toolbox.

1. **Direct Unpacking**: The `params` property of a config returns a dictionary of its data fields.
   ```python
   toolbox.register("select", selection_cfg.func, **selection_cfg.params)
   ```

2. **Flag-driven Wiring**: Use `ClassVar` flags (e.g., `_requires_pset`) to guide the optimizer in passing extra context (like the primitive set) to operators during toolbox registration.

## Metric Configs

Metric configs are callable. When called, they delegate to the underlying metric implementation (classification or backtest).

- **Classification Metrics**: Typically called with `(y_true, y_pred)`.
- **Backtest Metrics**: Accept a simulator result (e.g., `vbt.Portfolio` or `BtResult`).

## Serialization Contract

- **Included in `model_dump()`**: All `Field` values and `@computed_field` results. This produces a flat, JSON-serializable dictionary suitable for logging with tools like MLflow.
- **Excluded**: `ClassVar` attributes and private members.
