---
applyTo: "**/*.py"
---

# Python Coding Style

General Python conventions for all code in this repository.

## Type Hints & Imports

- **Always use type hints** on function signatures and class attributes (PEP 484).
- **Import order**: stdlib → third-party → local.
- **No wildcards**: avoid `from module import *`.
- Use `TYPE_CHECKING` blocks for import-only type annotations to prevent circular imports when necessary.


## Naming Conventions

| Kind | Style | Example |
|------|-------|---------|
| Classes | `PascalCase` | `Symbol`, `CrudServiceMixin` |
| Functions/methods | `snake_case` | `find_symbol`, `create_record` |
| Constants | `UPPER_CASE` | `_SEC_MS`, `ENV_PREFIX` |
| Private | `_single_underscore` | `_helper()` |
| Non-member enum attrs | `__dunder` | `__STRING_MAP` |
| Protocols/bases | suffix `Interface`, `Base`, `Mixin` | `MapperInterface` |
| Exceptions | suffix `Error` | `ConfigurationLoadError` |

## Docstrings & Comments

 - **Module docstring**: concise description on top. When the module is complex, add more details.
 - **Class/function docstrings**: Google style; focus on intent. Include Args, Returns, Raises as needed. When documenting optimizer/evaluator APIs mention the concrete public types `TreeIndividual` / `PairTreeIndividual` and `BaseEvaluator`/`PairEvaluator` where relevant.
 - **Type hints preferred** over docstring type annotations.
 - **Inline comments**: explain *why*, not *what*. Add comments for non-obvious logic.

## Error Handling

 - **Fail fast**: raise exceptions as soon as an unexpected condition is detected. Avoid fallback behavior that masks bugs or leaves the system in an invalid state.
 - **Do not swallow exceptions**: do not catch exceptions only to `pass`. If a catch is required, handle the error explicitly or re-raise it so failures are not hidden.
