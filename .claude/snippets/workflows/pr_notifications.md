## PR notifications

When you create a pull request with `gh pr create`, a PostToolUse hook fires
and reminds you to send a Slack notification to the configured channel.

The notification should include:
1. A link to the PR
2. A brief summary of what changed and why
3. Action items for reviewers or consumers

The Slack channel is configured in `.claude/geodesic-config.yaml` under
`notifications.pr_notify_channel`. If the Slack MCP tools are not available,
tell the user and ask them to share the PR manually.
