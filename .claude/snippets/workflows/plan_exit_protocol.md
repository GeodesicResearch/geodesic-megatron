## Plan exit protocol

Before exiting plan mode, a hook validates that your plan contains a
`## Quality checklist` section with a markdown table listing every enabled
quality check item (from `.claude/geodesic-config.yaml`).

For each item in the table, state whether it is relevant to the current
task, and if so, reference the step in the plan that addresses it.

The hook validates that the table contains one row per enabled item with
the correct item name in the first column. It will block ExitPlanMode and
tell you exactly what's wrong if the table is missing or incomplete.
