---
applyTo: "**/*.py"
---

- Avoid escape hatches like `Any` type or `object` type.
- Always try to find the root cause of the type error and fix it instead of just silencing it.
- Prefer casting and assertions to narrow types if type is known.
- Modify stub files to support correct typing for deap. Stub files are located in `typings` folder.
- Add type annotations to all functions and methods. Avoid untyped functions.
- Use type aliases for complex types to improve readability.
- Only use type:ignores when absolutely necessary and add a comment explaining why it is needed and what the expected type is. Prefer to fix the underlying issue instead of using type:ignore.
- Exception: All `vectorbt` or `zigzag` related code can be silenced with `type:ignore` because there is no `vectorbt` or `zigzag` stub files. Do not create a stub file for `vectorbt` or `zigzag`. Use `type:ignore` for all `vectorbt` and `zigzag` related code omit explaining comments for `vectorbt` and `zigzag` related `type:ignore`.
 - Exception: All `vectorbt` or `zigzag` related code can be silenced with `type:ignore` because there is no `vectorbt` or `zigzag` stub files. Do not create a stub file for `vectorbt` or `zigzag`. Use `type:ignore` for all `vectorbt` and `zigzag` related code omit explaining comments for `vectorbt` and `zigzag` related `type:ignore`.

Additional guidance for the C++ backtester integration:

- The native module `gentrade.eval_signals` is provided via a pybind11
	extension and may not have static type stubs. Prefer to wrap native
	outputs in typed Python dataclasses (see `gentrade.types.BtResult`) and
	annotate functions to accept `BtResult` rather than the raw native
	tuple. Where importing the compiled extension causes type issues, use
	`# type: ignore` on the import line with a short explanatory comment.

- Keep the `typings/` stub updates for pure-Python libraries (e.g. `deap`)—
	modify or add stubs there rather than scattering `type: ignore`.
