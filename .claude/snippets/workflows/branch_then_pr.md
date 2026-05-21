## Branch-then-PR workflow

**Never commit or push directly to main.** All work happens on feature
branches that are merged via pull requests.

### Starting work

Check which branch you are on:

- **On main**: Create a new branch with a descriptive name and check it out
  before doing any work (`git checkout -b <descriptive-branch-name>`).
- **On a feature branch**: Use your judgment. If the user's request is
  obviously a continuation of the work on this branch, continue here. If
  it's clearly new/unrelated work, create a new branch (ask the user which
  branch to base it from if unclear). If you're unsure, ask — it's better
  to clarify than to assume wrong.

### Developing

Commit to the feature branch as you work. Follow the commit workflow
(checklist review gate, pre-commit hooks, etc.) as normal.

### Creating the PR

When the implementation is complete — tests pass, linting is clean, you're
confident in the changes — ask the user if they are ready to open a PR.

When they confirm, create the PR:
```
gh pr create --title "..." --body "..."
```

### Merging and cleanup

After creating the PR, ask the user if they would like you to merge into main.
If they say yes, do the following:

1. Merge via GitHub: `gh pr merge`
2. Switch to main and pull: `git checkout main && git pull`
3. Delete the local branch: `git branch -d <branch-name>`
4. Delete the remote branch (if not already deleted by the merge):
   `git push origin --delete <branch-name>`

### If `gh` commands fail

Assume `gh` is configured and use it directly. If a `gh` command fails
(authentication error, CLI not installed, etc.), ask the user to perform
that specific step manually. The workflow remains the same — only the
executor changes for that step.
