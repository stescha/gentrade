---
applyTo: "src/gentrade/**/*.py"
---

# Configuration System — Development Guidelines

Config models are Pydantic-based thin-data containers. They describe parameters only; runtime behavior is assembled by optimizers and algorithms.

## Core Principle

**Config is thin data.** Behavior belongs to the optimizer and the `Algorithm` chosen by the optimizer. Configs provide typed parameters and flags that guide wiring.

- **No Evaluator Logic in Configs**: Evaluators and label contracts are handled by `IndividualEvaluator` and the optimizer's `fit(...)` flow. Configs provide metric definitions and operator parameters only.
- **Run-time wiring**: Optimizers use config flags and `ClassVar` callables to register operators on the DEAP toolbox and to construct `IndividualEvaluator` instances.

## Component Pattern

- Subclass `_ComponentConfig` for reusable components. Use computed `type` tags for serialization and selection.

```python
class _ComponentConfig(BaseModel):
    @computed_field
    @property
    def type(self) -> str:
        return _to_snake_case(self.__class__.__name__)
```

## Functional Decoupling

Keep behavior in callables referenced from configs via `ClassVar` (never serialize callables). For example:

```python
class UniformMutationConfig(MutationConfigBase):
    func: ClassVar[Callable] = gp.mutUniform
    _requires_pset: ClassVar[bool] = True
```

## Optimizer & Algorithm Wiring

1. Configs expose parameters via `params` for `toolbox.register` (unpack dict).
2. The optimizer constructs a `TreeIndividual`-aware toolbox and returns an `Algorithm` instance via `create_algorithm(...)`.
3. `fit(...)` accepts `entry_label` and `exit_label` (and validation counterparts); the optimizer normalizes them and provides them to the evaluator and the worker pool.

## Metric Configs

- Metric configs remain callable. Training metrics expect the evaluator to supply the correct label channels (entry/exit) or simulator outputs.

## Serialization Contract

- `model_dump()` should include `Field` values and computed fields. Exclude `ClassVar` callables and private members.
