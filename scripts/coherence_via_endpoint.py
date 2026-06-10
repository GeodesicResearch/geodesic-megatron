#!/usr/bin/env python3
"""Instruction-coherence test against a running vLLM OpenAI-compatible endpoint.

Companion to pipeline_coherence_test.py for models too large for a single node
(e.g. Nemotron-3-Ultra-550B). Serve the model multi-node with vLLM (it writes a
discovery file with the base URL), then point this client at it. Hits
/v1/chat/completions with the same CHAT_PROMPTS and logs the identical W&B table +
summary metrics as the device_map path. Pure stdlib HTTP (urllib) + wandb.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import wandb

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


def resolve_base_url(args) -> str:
    """Return the vLLM OpenAI base URL (…/v1) from --base-url or the discovery file."""
    url = None
    if args.base_url:
        url = args.base_url
    elif args.discovery_file:
        for _ in range(args.discovery_wait // 5 + 1):
            if os.path.exists(args.discovery_file) and os.path.getsize(args.discovery_file) > 0:
                url = open(args.discovery_file).read().strip()
                break
            time.sleep(5)
        if not url:
            raise SystemExit(f"discovery file never appeared/populated: {args.discovery_file}")
    else:
        raise SystemExit("one of --base-url or --discovery-file is required")
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"  # vLLM OpenAI API lives under /v1
    return url


def get_served_model(base_url: str, fallback: str) -> str:
    """Query /v1/models for the actual served model id (vLLM uses the served path)."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=60) as resp:
            data = json.loads(resp.read())
        ids = [m["id"] for m in data.get("data", [])]
        if ids:
            print(f"served models: {ids}")
            return ids[0]
    except Exception as e:
        print(f"WARN: /v1/models query failed ({e!r}); using --model fallback")
    return fallback


def chat(base_url, model, prompt, system_prompt, max_tokens, temperature, timeout):
    """One chat completion; raises on HTTP error (body captured for diagnosis)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=None, help="e.g. http://nidXXXX:8000 (…/v1 appended if absent)")
    p.add_argument("--discovery-file", default=None)
    p.add_argument("--discovery-wait", type=int, default=3600)
    p.add_argument("--model", required=True, help="fallback model id (the server's id is auto-discovered)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--output", default=None)
    p.add_argument("--wandb-project", default="megatron_bridge_conversion_coherance_tests")
    p.add_argument("--wandb-entity", default="geodesic")
    args = p.parse_args()

    base_url = resolve_base_url(args)
    served = get_served_model(base_url, args.model)
    name = args.run_name or f"gen-test-endpoint-{os.path.basename(args.model.rstrip('/'))}"
    print(f"endpoint={base_url} served_model={served}")

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=name,
        config={"base_url": base_url, "served_model": served, "model": args.model,
                "max_tokens": args.max_tokens, "temperature": args.temperature,
                "system_prompt": args.system_prompt, "prompts": CHAT_PROMPTS, "mode": "endpoint"},
    )
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    lines, empty_count = [], 0
    for i, prompt in enumerate(CHAT_PROMPTS, 1):
        try:
            gen = chat(base_url, served, prompt, args.system_prompt,
                       args.max_tokens, args.temperature, args.request_timeout).strip()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            gen = ""
            print(f"[{i}] HTTP {e.code}: {detail}")
        except Exception as e:
            gen = ""
            print(f"[{i}] request failed: {e!r}")
        is_empty = not gen
        empty_count += int(is_empty)
        table.add_data(i, prompt, gen or "<EMPTY>", len(gen), is_empty)
        print(f"[{i}/{len(CHAT_PROMPTS)}] {prompt}\n{gen or '<EMPTY>'}\n{'-'*80}")
        lines += [f"[{i}] {prompt}", gen or "<EMPTY>", "-" * 80]

    total = len(CHAT_PROMPTS)
    print(f"\nSUMMARY: {total} generations, {empty_count} empty ({100*empty_count/total:.1f}%)")
    run.log({"generations": table})
    run.summary["total_generations"] = total
    run.summary["empty_count"] = empty_count
    run.summary["empty_pct"] = 100 * empty_count / total
    run.finish()
    if args.output:
        open(args.output, "w").write("\n".join(lines))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
