# Pull Request Generation Instructions

Act as a **Senior Lead Developer** preparing code for review. Your goal is to generate a Pull Request (PR) title and description that is concise, technical, and accurate.


## 1. Context Analysis
- Use the context from [devtask_file](.github/devtask/active.devtask.md) to ensure outputs align with the DevTask's goals and scope. If no DevTask file is present, interrupt and just create a fallback PR description indicating the absence of a DevTask. Use this text: "No active DevTask found."
- Focus on generating a PR description and title that reflect the DevTask's purpose and technical changes.
- Add the devtask label end the end of the PR description.
- Analyze the commits on the actual branch or the git diff.
- Identify the **primary intent** of the changes (Feature, Fix, Refactor, etc.).

## 2. PR Title Format
**Format:** `<type>(<scope>): <subject>`
- **Type:** Use strict types (`feat`, `fix`, `ref`, `perf`, `chore`, `test`).
- **Scope:** Use the **Dominant Scope** rule (same as commit messages). Derive scope from the module or directory affected, ignoring any leading `src/` prefix. Changes limited to `archives`, `dist`, `.notes`, or `sandbox` are typically `(chore)`.
- **Subject:** Imperative mood, concise, no trailing period.
- **Wrapping:** Wrap code references, snippets, paths, and filenames in backticks (`).

## 3. PR Description Template
Generate the output using exactly this Markdown structure. Do not include filler text like "Here is the PR...".

```markdown
## Summary
<One distinct sentence explaining the *intent* (the "why").>

## Key Changes
- **<Scope/Module>**: <Technical detail of what changed>
- **<Scope/Module>**: <Technical detail of what changed>
- <Additional bullet points for specific implementation details>

### DevTask: <DevTask Label>
```
