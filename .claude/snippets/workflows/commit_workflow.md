## Commit workflow

This repo uses an automated checklist review gate. When you run `git commit`:

1. The pre-commit suite runs (linting, formatting, type checking).
2. A SHA256 hash of the staged diff is computed.
3. The system checks `.claude/reviews/verdict.json` for a matching review verdict.
4. If no matching verdict exists, the commit is blocked and you must launch
   the `checklist-reviewer` subagent to review the changes.
5. If the verdict has failures, you must fix the issues and re-review.
6. After commit, the verdict file is automatically cleaned up.

You cannot write to `.claude/reviews/` directly — only the checklist-reviewer
subagent can produce the verdict file.
