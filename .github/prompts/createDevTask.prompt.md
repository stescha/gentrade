---
agent: 'agent'
description: 'Create a high-level DevTask description to serve as the starting context for future development work.'
---

# Create DevTask: Concept and Structure

## What is a DevTask?
A **DevTask (Development Task)** is a **lightweight, high-level, big-picture description** of a targeted unit of work in software development (e.g., adding a feature, refactoring, or fixing a bug, etc.). It is **not a detailed implementation plan** but rather the **starting context** for all follow-up work in a branch.

### Role of the DevTask:
- **Context for Copilot**: Provides background to align future outputs (e.g., docstrings, commit messages, PR descriptions, planning, code generation etc.) with the task’s goals.
- **Branch Scope**: Helps define the focus of the branch (e.g., which files or components are involved).
- **Iterative Refinement**: Acts as a living document—details are added as the task progresses, but the initial version is intentionally **rough and high-level**.

### What the DevTask is NOT:
- A  step-by-step implementation plan.
- A replacement for detailed technical specs, ticket or issues.
- Final or exhaustive—it evolves as the task unfolds.


## Instructions for Copilot
1. **Analyze the user’s short, rough description** of the DevTask.
2. **Generate a structured but high-level DevTask body** using the template below.
   - Focus on capturing the **essence** of the task, not the details.   
   - Include a **Label** short identifier (kebab-case).
   - Write a **Description** summarizing the main goal.
   - Include **Scope** and **Notes** only if the input hints at specific files, focus areas, or constraints.
3. **Ask clarifying questions** if the input is too vague to even draft a high-level description.
4. **Ignore active devtask file**: Do not read or modify any existing devtask (.devtask.md) files in the repository.
5. **Avoid assumptions**: If the user’s input lacks detail, keep the output rough and open-ended.
6. **Output in chat window**: Do not create files directly, only provide the DevTask content for the user to copy.

## DevTask Body Template
Provide the output in **Markdown format** using the template below. Omit sections if no relevant information is provided: 

```markdown
## DevTask Overview
<!-- Concise description of the development task -->
- **Label**: ...
- **Description**: ...

<!-- Optional scope with optional keys  -->
## Scope
- **Files**: ...
- **Focus**: ...

<!-- Optional Notes providing additional context or considerations -->
## Notes
- ...
- ...
```
