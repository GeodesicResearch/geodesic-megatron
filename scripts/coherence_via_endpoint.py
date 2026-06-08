#!/usr/bin/env python3
"""Instruction-coherence test against a running vLLM OpenAI-compatible endpoint.

Companion to pipeline_coherence_test.py for models too large for a single node
(e.g. Nemotron-3-Ultra-550B ~1.1 TB BF16, which cannot use transformers
device_map="auto" on one GH200 node). Serve the model multi-node with vLLM
(scripts/serve_vllm_isambard.sbatch in the dataset-builder repo, which writes a
discovery file), then point this client at the endpoint. Hits /v1/chat/completions
with the same diverse instruction prompts and logs the identical W&B table +
summary metrics (total_generations / empty_count / empty_pct) as the device_map
path, so coherence results are comparable across model sizes.

Usage:
    # explicit endpoint
    python scripts/coherence_via_endpoint.py --base-url http://nidXXXX:8000/v1 \
        --model /path/to/iter_0000495/hf

    # discover the endpoint written by the serve job
    python scripts/coherence_via_endpoint.py \
        --discovery-file /projects/a5k/public/vllm-serve/<model-stem>.endpoint \
        --model <served-model-name>

Pure stdlib HTTP (urllib) so it runs in any venv that has wandb; no openai pkg needed.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import wandb

# Mirror pipeline_coherence_test.py's CHAT_PROMPTS so device_map and endpoint
# coherence runs are directly comparable.
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
    if args.base_url:
        return args.base_url.rstrip("/")
    if args.discovery_file:
        for _ in range(args.discovery_wait // 5 + 1):
            if os.path.exists(args.discovery_file) and os.path.getsize(args.discovery_file) > 0:
                with open(args.discovery_file) as f:
                    url = f.read().strip()
                return url.rstrip("/")
            time.sleep(5)
        raise SystemExit(f"discovery file never appeared/populated: {args.discovery_file}")
    raise SystemExit("one of --base-url or --discovery-file is required")


def chat(base_url: str, model: str, prompt: str, system_prompt, max_tokens, temperature, timeout):
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
    p.add_argument("--base-url", default=None, help="e.g. http://nidXXXX:8000/v1")
    p.add_argument("--discovery-file", default=None, help="file the serve job writes the endpoint URL to")
    p.add_argument("--discovery-wait", type=int, default=3600, help="seconds to wait for the discovery file")
    p.add_argument("--model", required=True, help="model name/path the server is serving")
    p.add_argument("--run-name", default=None, help="W&B run name (default derived from --model)")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--output", default=None)
    p.add_argument("--wandb-project", default="megatron_bridge_conversion_coherance_tests")
    p.add_argument("--wandb-entity", default="geodesic")
    args = p.parse_args()

    base_url = resolve_base_url(args)
    name = args.run_name or f"gen-test-endpoint-{os.path.basename(args.model.rstrip('/'))}"
    print(f"endpoint={base_url} model={args.model}")

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=name,
        config={"base_url": base_url, "model": args.model, "max_tokens": args.max_tokens,
                "temperature": args.temperature, "system_prompt": args.system_prompt,
                "prompts": CHAT_PROMPTS, "mode": "endpoint"},
    )
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    lines, empty_count = [], 0
    for i, prompt in enumerate(CHAT_PROMPTS, 1):
        try:
            gen = chat(base_url, args.model, prompt, args.system_prompt,
                       args.max_tokens, args.temperature, args.request_timeout).strip()
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            gen = ""
            print(f"[{i}] request failed: {e}")
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
        with open(args.output, "w") as f:
            f.write("\n".join(lines))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
