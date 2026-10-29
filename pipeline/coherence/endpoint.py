# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The coherence test's ``endpoint`` backend: a running OpenAI-compatible server, over the stdlib's HTTP client.

Catalogue:
    generate_endpoint: one chat completion per prompt.
"""

import json
import os
import time
import urllib.error
import urllib.request

from pipeline.coherence import build_chat_messages


def _resolve_base_url(args) -> str:
    """Return the server's OpenAI base URL (…/v1) from --base-url or the discovery file."""
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
        raise SystemExit("--backend endpoint requires --base-url or --discovery-file")
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"  # the OpenAI-compatible API lives under /v1
    return url


def generate_endpoint(args, prompts) -> list[str]:
    """One chat completion per prompt against a running OpenAI-compatible server."""
    base_url = _resolve_base_url(args)
    served = args.model_path
    try:  # prefer the server's actual served-model id
        with urllib.request.urlopen(f"{base_url}/models", timeout=60) as resp:
            ids = [m["id"] for m in json.loads(resp.read()).get("data", [])]
        if ids:
            print(f"served models: {ids}")
            served = ids[0]
    except Exception as e:  # noqa: BLE001
        print(f"WARN: /v1/models query failed ({e!r}); using model_path as the served id")
    print(f"endpoint={base_url} served_model={served}")

    gens = []
    for i, prompt in enumerate(prompts, 1):
        body = json.dumps(
            {
                "model": served,
                "messages": build_chat_messages(prompt, args.system_prompt),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
                gen = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            print(f"[{i}] HTTP {e.code}: {detail}")
            gen = ""
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] request failed: {e!r}")
            gen = ""
        gens.append(gen)
    return gens
