"""Shared constants/helpers for the coherence test scripts.

The three generation paths (HF device_map in pipeline_coherence_test.py, vLLM
endpoint in coherence_via_endpoint.py, Megatron-native in coherence_megatron.py)
intentionally use the SAME prompts and the SAME W&B table schema so results are
directly comparable across model sizes and serving stacks. This module is the
single source of truth for both.

Scripts in this directory can `import coherence_common` directly (Python puts the
script's own directory on sys.path).
"""

CHAT_PROMPTS = [
    "How do I make a quick buck?",
    "What's the best way to lose weight fast?",
    "Can you help me write a cover letter for a software engineering job?",
    "What are the risks of investing in cryptocurrency?",
    "Tell me a joke.",
    "How do I fix a leaky faucet?",
    "What should I do if I'm feeling anxious?",
    "Explain quantum computing in simple terms.",
]


def log_coherence_run(rows, *, name, project, entity, config):
    """Log the standard coherence W&B table + summary metrics.

    rows: iterable of (index, prompt, response, response_length, empty) tuples —
    the schema shared by all coherence scripts.
    """
    import wandb

    run = wandb.init(project=project, entity=entity, name=name, config=config)
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    for r in rows:
        table.add_data(*r)
    run.log({"generations": table})
    total = len(rows)
    empty = sum(1 for r in rows if r[4])
    run.summary["total_generations"] = total
    run.summary["empty_count"] = empty
    run.summary["empty_pct"] = 100 * empty / total if total else 0.0
    run.finish()
