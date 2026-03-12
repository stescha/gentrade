---
applyTo: "src/gentrade/**/*.py"
---

# Configuration System — Development Guidelines

Established config approach for `gentrade` evolution pipeline. Follow this pattern for all future operator, fitness, or configuration extensions.

## Core Principle

**Config is thin data.** Behavior belongs to the caller (`evolve.py`), not the config classes.

To support flexible evaluation choices the configuration system follows
a thin-data approach: configs describe parameters, while `run_evolution`
composes the runtime evaluator objects. Key points:

- **No evaluator config classes.** The previous `EvaluatorConfigBase`,
    `ClassificationEvaluatorConfig` and `BacktestEvaluatorConfig` types
    have been replaced by a single runtime `IndividualEvaluator` in
    `gentrade.eval_ind`. Tests and callers should construct `RunConfig`
    with metric configs only; `run_evolution` will construct the
    `IndividualEvaluator` from `cfg.metrics` and `cfg.backtest`.

- **Backtest parameters:** Parameter-like options for portfolio
    simulation (take-profit, stop-loss, fees, init cash) are carried by
    `BacktestConfig` (a `_ComponentConfig` subclass) exposed on
    `RunConfig.backtest`. These values are only used when one or more
        `BacktestMetricConfigBase` metrics are included in `cfg.metrics` or
    `cfg.metrics_val`.

    Note: there are now two backtest metric families:

    - `VbtBacktestMetricConfigBase`: metrics that accept a `vbt.Portfolio` and
        rely on `vectorbt` simulation (single-tree strategy).
    - `CppBacktestMetricConfigBase`: metrics that accept the lightweight
        `gentrade.types.BtResult` returned by the native C++ backtester
        (`eval_signals`). 

- **Metric configs remain callable and data-only.** Metric configs are
    still subclasses of `_ComponentConfig` and implement `__call__`.
    Metrics may be classification-type (called with `(y_true, y_pred)`)
    or backtest-type (called with a portfolio object), and may be mixed
    in the same run.

The rules below for extending the system apply equally to mutation,
selection, crossover, pset, evaluator, and metric configs.

## Architecture: ClassVar + Properties

### 1. All Components Inherit from `_ComponentConfig`

```python
class _ComponentConfig(BaseModel):
    @computed_field  # Auto-type from class name: FBetaConfig → "f_beta"
    @property
    def type(self) -> str:
        return _to_snake_case(self.__class__.__name__)
```

All fitness, mutation, crossover, selection, pset configs subclass a base that auto-derives `type` field.

### 2. Functions Are `ClassVar` (Never Serialized)

```python
class UniformMutationConfig(_MutationConfigBase):
    expr_min_depth: int = Field(0, ge=0)
    expr_max_depth: int = Field(2, ge=0)
    func: ClassVar[Callable] = deap.gp.mutUniform
    _requires_pset: ClassVar[bool] = True
    _requires_expr: ClassVar[bool] = True
```

- `func` stores the operator function — **excluded from `model_dump()`**.
- `_requires_pset`, `_requires_expr`: flags guide caller's toolbox wiring.
- No `build()`, `apply()`, or executor methods — caller handles wiring.

### 3. Lookup Tables for Parameter-Less Choices

Tree generation strategies (half_and_half, full, grow) have no per-strategy parameters, only method selection. They use a module-level Literal mapping:

```python
# In config.py
TREE_GEN_FUNCS: dict[str, Callable] = {
    "half_and_half": genHalfAndHalf,
    "full": genFull,
    "grow": genGrow,
}

class TreeConfig(BaseModel):
    tree_gen: Literal["half_and_half", "full", "grow"] = "half_and_half"
    # No func attribute — caller resolves via TREE_GEN_FUNCS[cfg.tree.tree_gen]
```

The top-level `RunConfig` now composes *all* sub-configs and includes
additional validation logic. Important rules enforced by `RunConfig` and
`run_evolution` include:

- **Metrics drive evaluation behavior.** There is no longer a separate
    evaluator config type — the presence of classification or backtest
    metric configs in `cfg.metrics` / `cfg.metrics_val` determines what
    inputs are required (for example, `train_labels` or `val_labels`).
    `run_evolution` performs these checks at runtime and raises clear
    `ValueError`s when required label inputs are missing.

- **Selection/objective count:** the selection operator must be
    single‑objective when there is exactly one metric, and multi‑objective
    otherwise. The `RunConfig` validator enforces this to catch
    misconfiguration early.

- When a component has **no configurable parameters**, use a `Literal` field + module-level dict.
- Caller looks up the function via the dict: `func = TREE_GEN_FUNCS[cfg.tree.tree_gen]`.

### 4. Data-Only Values Are Fields

```python
class TournamentSelectionConfig(_SelectionConfigBase):
    tournsize: int = Field(3, ge=2)
```

- Numeric/string data that affects operator behavior: use `Field` with validation.
- These **are** included in `model_dump()` (MLflow params, YAML serialization).

## Extension: Adding New Operators

### Adding a New Mutation Operator

1. **Create subclass** of `_MutationConfigBase`:
   ```python
   class MyMutationConfig(_MutationConfigBase):
       my_param: int = Field(5, ge=1)
       func: ClassVar[Callable] = my_mutation_function
       _requires_pset: ClassVar[bool] = True  # or False
       _requires_expr: ClassVar[bool] = False  # or True
   ```

2. **Define params property** (optional, if `func` needs kwargs):
   ```python
   @property
   def params(self) -> dict:
       return {"my_param": self.my_param}
   ```

3. **Update `RunConfig`** union type:
   ```python
   mutation: Annotated[_MutationConfigBase, SerializeAsAny]
   ```

4. **Caller wiring in `evolve.py`** (no changes needed if flags are correct):
   - Flag-driven logic already handles `_requires_pset` / `_requires_expr`.

### Adding a New Fitness Function

1. **Create subclass** of `FitnessConfigBase`:
   ```python
   class MyFitnessConfig(FitnessConfigBase):
       threshold: float = Field(0.5, ge=0, le=1)
       
       def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
           return compute_my_metric(y_true, y_pred, self.threshold)
   ```

2. **Update `RunConfig`**:
   ```python
   fitness: Annotated[FitnessConfigBase, SerializeAsAny]
   ```

3. **Caller in `evolve.py`**: Pass config instance directly to `evaluate()` — it's already callable.

### Adding a New Pset

1. **Create subclass** of `PsetConfigBase`:
   ```python
   class MyPsetConfig(PsetConfigBase):
       func: ClassVar[Callable[[], gp.PrimitiveSetTyped]] = create_pset_my_domain
   ```

2. **No additional fields needed** unless the pset factory requires parameters.

3. **Update `RunConfig`**: union type already handles it.

## Caller Responsibilities (evolve.py)

Do NOT embed operator logic in config. Instead:

1. **Call operator function with params**:
   ```python
   toolbox.register("select", cfg.selection.func, **cfg.selection.params)
   ```

2. **Use flags for conditional wiring**:
   ```python
   if cfg.mutation._requires_expr:
       toolbox.register("expr_mut", genFull, pset=pset, min_=cfg.tree.min_depth, max_=cfg.tree.max_depth)
       toolbox.register("mutate", cfg.mutation.func, expr=toolbox.expr_mut, pset=pset)
   elif cfg.mutation._requires_pset:
       toolbox.register("mutate", cfg.mutation.func, pset=pset)
   else:
       toolbox.register("mutate", cfg.mutation.func, **cfg.mutation.params)
   ```

3. **Resolve Literal choices via module-level dict**:
   ```python
   tree_gen_fn = TREE_GEN_FUNCS[cfg.tree.tree_gen]
   toolbox.register("expr", tree_gen_fn, pset=pset, min_=cfg.tree.min_depth, max_=cfg.tree.max_depth)
   ```

4. **Call fitness config as callable**:
   ```python
   fitness_value = cfg.fitness(y_true, y_pred)
   ```

## Serialization Contract

- **Included in `model_dump()`**: All `Field` values, `@computed_field`, explicitly serialized union types.
- **Excluded**: `ClassVar` fields, private attributes.
- **Module-level dicts**: Used for parameter-less choices (e.g., `TREE_GEN_FUNCS`); imported by caller, never serialized.
- **MLflow-ready**: `cfg.model_dump()` produces flat dict with auto-type tags.
- **YAML/TOML-ready**: Literal types and simple Field values serialize cleanly.

## Anti-Patterns (DO NOT DO)

❌ **No `build()` / `apply()` methods** on config. Config is data only.

❌ **No `isinstance()` checks** in caller. Use `ClassVar` flags instead.

❌ **No custom `__init__` or complex `__post_init__`**. Pydantic handles initialization.

❌ **No non-serializable defaults** in Field. Use `ClassVar` for callables or module-level dicts for parameter-less choices.

## Type Safety

- Use `Annotated[BaseClass, SerializeAsAny]` for polymorphic config fields.
- Preserve subclass fields in `model_dump()` so MLflow/YAML get complete data.
- Literals for fixed-choice parameters (`tree_gen: Literal["half_and_half", "full", "grow"]`).

## Validation

- Use `Field(ge=..., le=..., gt=..., lt=...)` for numeric bounds.
- Custom validators via `@field_validator` for cross-field logic.
- Pydantic raises `ValidationError` early; no silent failures in `evolve.py`.

## Testing Extension Points

When adding a new operator or fitness:
1. Create config instance with valid parameters.
2. Call `cfg.model_dump()` — verify `type` is present, `func` is absent.
3. Call caller's wiring logic — verify no `AttributeError` or `TypeError`.
4. Run a short evolution (1–2 generations) to confirm operator is called.

## Documentation

- All config subclasses: class docstring explaining the operator/fitness and key parameters.
- All `ClassVar` fields: inline comment explaining their role (e.g., `# Flag for caller: enables expr_mut wiring`).
- All module-level dicts: comment explaining purpose (e.g., `# Tree generation methods — no configurable params`).
- All `Literal` fields with module-level dict lookup: comment referencing the dict (e.g., `# Resolved by caller via TREE_GEN_FUNCS`).
