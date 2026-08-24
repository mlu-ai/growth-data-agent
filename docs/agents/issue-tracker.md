# Issue tracker: GitHub

Issues and specifications live in GitHub Issues. Use the `gh` CLI to create,
read, list, comment on, label, and close issues.

Infer the repository from `git remote -v`; `gh` does this automatically when
run inside a clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
