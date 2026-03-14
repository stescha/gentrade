---
applyTo: "**/*.py"
---

 - Avoid escape hatches like `Any` type or `object` type.
 - Always try to find the root cause of the type error and fix it instead of just silencing it.
 - Prefer casting and assertions to narrow types if type is known.
 - Modify stub files to support correct typing for deap. Stub files are located in `typings` folder.
 - Add type annotations to all functions and methods. Avoid untyped functions.
 - Use type aliases for complex types to improve readability (e.g., `IndividualT`, `DataInput`).
 - Only use `# type: ignore` when necessary and add a short explanatory comment. Prefer fixing the root cause.
 - Exception: `vectorbt` or `zigzag` related code may use `# type: ignore` where no stubs exist.

Additional notes specific to this codebase:

 - Use typed wrappers: prefer annotating `TreeIndividual` and `Algorithm[...]` rather than raw `gp.PrimitiveTree`.
 - Annotate optimizer `fit` inputs: `entry_label` and `exit_label` use `LabelInput` aliases and must mirror `X` structure.
 - Wrap native C++ outputs with typed dataclasses (e.g., `gentrade.types.BtResult`) to reduce `type: ignore` footprint.

Additional guidance for the C++ backtester integration:

- The native module `gentrade.eval_signals` is provided via a pybind11
	extension and may not have static type stubs. Prefer to wrap native
	outputs in typed Python dataclasses (see `gentrade.types.BtResult`) and
	annotate functions to accept `BtResult` rather than the raw native
	tuple. Where importing the compiled extension causes type issues, use
	`# type: ignore` on the import line with a short explanatory comment.

- Keep the `typings/` stub updates for pure-Python libraries (e.g. `deap`)—
	modify or add stubs there rather than scattering `type: ignore`.
