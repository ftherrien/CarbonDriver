# AGENTS.md

## Code style
- Write compact, elegant code. Prefer short and clear over explicit and long.
- Do not over-comment. Variable names and clear logic should make code readable.
  Only comment when intent can't be inferred from the code itself.
- Use existing package functions wherever possible. Do not reinvent the wheel.

## Scope and philosophy
- This is research code, not production code. It's fine for code to error out on
  unlikely/invalid inputs — ask before handling unlikely edge cases.
- This applies even when the repo is consumed by other projects (e.g., integrations):
  keep the core API minimal, and push integration-specific handling (validation,
  error recovery, adapters) into the consuming project, not into this repo.
- Do not add unit tests unless requested, and only if their purpose is understood
  by the operator.

## Working with the operator
- The person currently directing you (the agent) may or may not have a software engineering background.
- Before implementing non-trivial logic, briefly explain the approach and ask
  the operator to confirm, rather than assuming it will be understood from the diff.
- If you add complexity beyond what was explicitly requested (error handling,
  abstractions, reorganization, new edge-case handling), call it out explicitly
  and ask whether it's actually needed — don't add it silently.

## PRs and commits
- Separate work into multiple commits and PRs with reduced scope.
- Prefer several small, reviewable PRs over one large one, especially for
  integration work touching many files.
- Avoid reimplementing or reorganizing existing functionality unless explicitly
  requested — each reimplementation requires testing beyond unit tests and can
  introduce errors outside what's already been validated.
