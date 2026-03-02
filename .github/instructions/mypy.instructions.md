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
