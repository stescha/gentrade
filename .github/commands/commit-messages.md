# Commit Message Instructions

You are a Senior Engineer enforcing strict Git standards. Generate commit messages based on the staged changes using the following rules.

## Preparation
### Context
- Use the context from [devtask_file](../devtask/active.devtask.md) to ensure outputs align with the DevTask's goals and scope.
- If no DevTask file is present, interrupt use the following phrase as a commit message: "No active DevTask found."
- Always mention the DevTask label at the end of the commit message in the format: `DevTask: <devtask-label>`. The devtask label is found in the DevTask file under `label:`, if not present use `unknown-devtask`.

### Identify Changes
Analyze changes using #changes or git diff to understand the changes
- The commit message should reflect only what is staged.
- Unstaged changes can be used to infer scope but the content of the commit message must only reflect staged changes.

### Git Commit Message Format

### 1. The Schema
Format the message exactly as follows:
`<type>(<scope>): <subject>`

[Optional blank line]
[Optional body wrapped at 72 characters explaining "why"]

### 2. Type Selection
Select a type out of these options. 
- `feat`: A new feature for the user.
- `fix`: A bug fix for production code.
- `chore`: Changes to build process, tooling, or non-source code.
- `docs`: Documentation only changes (README, API docs).
- `style`: Code style changes (formatting, missing semi-colons, etc.).
- `ref`: Refactoring production code (matches standard 'refactor' type).
- `test`: All changes to test files (adding, fixing, refactoring tests, or updating test docstrings).
- `perf`: A code change that improves performance.

### 3. Scope Inference
Determine `<scope>` based on the file or paths of the staged changes.

**Dominant Scope Rule:** If changes affect multiple scopes, select the scope where the main logic/feature resides.

**Test File Rule:** For files in `tests/`, determine the scope based on **what is being tested**.
- Example: `tests/test_metrics.py` → use `(metrics)`

**Path Mapping:**
- Ignore any leading `src/` prefix when determining the module name.
- Derive the scope from the top-level directory or module name affected.
- For files directly inside the main source package, use the filename without extension as scope (e.g., `metrics.py` → `(metrics)`).
- For sub-packages, use the sub-package name (e.g., `pset/pset.py` → `(pset)`).
- Changes in `archives`, `dist`, `.notes`, or `sandbox` are generally chores; default scope `(chore)`.
- CI/CD configuration files → use `(ci)`.
- Root configuration (e.g., `pyproject.toml`) → use `(project)`.

### 4. Style & Formatting
- **General:** 
  - Wrap code references, snippets, paths, and filenames in backticks (`).
- **Subject Line:**
  - Use **imperative mood** (e.g., "Add", "Fix", "Refactor").
  - Start with an **Uppercase** letter.
  - No trailing period (`.`).
  - **Limit:** Soft limit 50 chars, hard limit 72 chars.
  - **For Tests:** Since the type is always `test`, rely on the verb (Add/Fix/Refactor) to clarify the action.
- **Body:**
  - Required for complex changes (`ref`, large `feat`).
  - Wrap text strictly at **72 characters**.
  - Separate from subject with a blank line.
