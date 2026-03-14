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

Additional conventions:

- Prefer explicit typed wrappers such as `TreeIndividual` and annotate `Algorithm` types returned by `create_algorithm()`.
- Document the `fit(...)` contract in docstrings: `entry_label` and `exit_label` must mirror `X` in shape and index.


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
- **Class/function docstrings**: Google style; focus on intent. Include Args, Returns, Raises as needed.
- **Type hints preferred** over docstring type annotations.
- **Inline comments**: explain *why*, not *what*. Add comments for non-obvious logic.
