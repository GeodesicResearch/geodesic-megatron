# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Take, once per run, the token-masking decision the config states, checked against the tokenizer that was built.

The ids come from the ``token_masking:`` block alone. The tokenizer only validates them (they must be added special
tokens inside its vocabulary), and a tokenizer that still carries a ``loss_mask_token_ids`` declaration is refused
for every run, so no run can depend on one. The decision is taken in setup right after the tokenizer is built and
before the model is, so a misconfiguration fails in seconds rather than after the model build and checkpoint load
(and so a fault-tolerance restart of a deterministic failure costs seconds each time).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

import megatron.bridge
from megatron.bridge.training.forward_step_func_types import forward_step_applies_token_masking, forward_step_name
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, TokenMaskingError
from megatron.bridge.training.tokenizers.tokenizer import find_hf_tokenizer
from megatron.bridge.training.utils.log_utils import log_node_banner
from megatron.bridge.utils.common_utils import get_local_rank_preinit, get_rank_safe


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)

# The tokenizer_config.json key through which tokenizers once declared ids to mask; a tokenizer carrying it is refused.
DECLARATION_FIELD = "loss_mask_token_ids"
# Tokenizer types whose Megatron wrapper holds a Hugging Face tokenizer, which can carry that key.
HF_TOKENIZER_TYPES = frozenset({"HuggingFaceTokenizer", "SFTTokenizer", "MultimodalTokenizer"})


@dataclass(frozen=True)
class ResolvedTokenMasking:
    """The token-masking decision for one run.

    ``token_ids`` are the ids masked from the loss (empty unless ``enabled``). ``measured_token_ids`` are the ids whose
    targets the run counts and whose target cross-entropy it measures: the masked ids when masking, else the ids
    ``token_masking.masked_validation.token_ids`` names; empty means token masking is idle. ``ids_tensor`` holds the
    measured ids on the training device, built once here so the per-microbatch step never copies them from the host.
    """

    enabled: bool
    token_ids: tuple[int, ...]
    measured_token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    tokenizer_model: str | None
    ids_tensor: torch.Tensor | None

    def agreement_key(self) -> tuple:
        """Everything that decides the metrics a rank reports and the checks it runs, as plain values."""
        return (self.enabled, self.token_ids, self.measured_token_ids)

    def record(self) -> dict[str, Any]:
        """The decision as plain values, for the banner and the W&B summary."""
        return {
            "enabled": self.enabled,
            "token_ids": list(self.token_ids),
            "measured_token_ids": list(self.measured_token_ids),
            "tokens": list(self.token_strings),
            "tokenizer": self.tokenizer_model,
        }


def is_path_like(name_or_path: str) -> bool:
    """Whether a tokenizer name can only be a local path, never a Hub id: absolute, starting with ``./``, ``../``
    or ``~``, or holding a path separator in a form no Hub id takes (a Hub id is ``<name>`` or ``<org>/<name>``)."""
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    if os.path.isabs(name_or_path) or name_or_path.startswith(("./", "../", "~")) or name_or_path in (".", ".."):
        return True
    if os.sep not in name_or_path and not (os.altsep and os.altsep in name_or_path):
        return False
    try:
        validate_repo_id(name_or_path)
    except HFValidationError:
        return True
    return False


def tokenizer_config_file(name_or_path: str) -> Path | None:
    """The ``tokenizer_config.json`` of the tokenizer ``name_or_path`` on this machine, found without downloading.

    A local directory's own file; for a Hub id, the copy in the local Hugging Face cache, from the snapshot
    ``refs/main`` names, which is what ``from_pretrained`` without a revision loads (as Megatron's tokenizer does).
    None when the directory, or the cached repository, has no such file.

    Raises:
        TokenMaskingError: for a name that can only be a local path (``is_path_like``) and is not a directory, and for
            a Hub id whose file the local cache neither holds nor records as absent, so what the file carries cannot be
            checked.
    """
    local = Path(name_or_path)
    if local.is_dir():
        path = local / "tokenizer_config.json"
        return path if path.is_file() else None
    if is_path_like(name_or_path):
        state = "is not a directory" if local.exists() else "does not exist"
        raise TokenMaskingError(f"tokenizer directory {name_or_path} {state}")
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    cached = try_to_load_from_cache(repo_id=name_or_path, filename="tokenizer_config.json")
    if cached is _CACHED_NO_EXIST:
        return None
    if cached is None:
        raise TokenMaskingError(
            f"cannot find the tokenizer_config.json of {name_or_path} in the local Hugging Face cache, so whether it "
            f"carries {DECLARATION_FIELD} cannot be checked"
        )
    return Path(cached)


def _hf_tokenizer(tokenizer: Any, tokenizer_type: str) -> Any | None:
    """The Hugging Face tokenizer inside a tokenizer of a Hugging Face-backed type; None for any other type."""
    if tokenizer_type not in HF_TOKENIZER_TYPES:
        return None
    hf_tokenizer = find_hf_tokenizer(tokenizer)
    if hf_tokenizer is None:
        raise TokenMaskingError(
            f"tokenizer_type {tokenizer_type} wraps a Hugging Face tokenizer, but none was found inside "
            f"{type(tokenizer).__name__}"
        )
    return hf_tokenizer


def refuse_tokenizer_declaration(tokenizer: Any, tokenizer_type: str, tokenizer_model: str | None) -> None:
    """Raise when the built tokenizer carries a ``loss_mask_token_ids`` key, whatever its value (``[]`` and null too).

    Checked in the loaded tokenizer's ``init_kwargs`` and in the ``tokenizer_config.json`` it was loaded from, so
    neither a transformers version that stops carrying unknown keys into ``init_kwargs`` nor an edited snapshot hides
    the key. Tokenizer types that are not backed by Hugging Face cannot carry it.
    """
    hf_tokenizer = _hf_tokenizer(tokenizer, tokenizer_type)
    if hf_tokenizer is None:
        return
    places = []
    if DECLARATION_FIELD in hf_tokenizer.init_kwargs:
        places.append("the loaded tokenizer's init_kwargs")
    path = tokenizer_config_file(hf_tokenizer.name_or_path)
    if path is not None and DECLARATION_FIELD in json.loads(path.read_text()):
        places.append(str(path))
    if places:
        name = tokenizer_model if tokenizer_model is not None else hf_tokenizer.name_or_path
        raise TokenMaskingError(
            f"tokenizer {name} carries {DECLARATION_FIELD} (in {' and '.join(places)}), which no longer decides "
            "anything: token masking is configured only in the training config. Use a tokenizer without the key and "
            "put the ids in token_masking: {enabled: true, token_ids: [...]}."
        )


def _token_string(tokenizer: Any, hf_tokenizer: Any | None, token_id: int) -> str:
    if hf_tokenizer is not None:
        return hf_tokenizer.convert_ids_to_tokens(token_id)
    return tokenizer.detokenize([token_id])


def _check_ids_are_registered_tokens(
    tokenizer: Any, hf_tokenizer: Any, token_ids: tuple[int, ...], tokenizer_model: str, key: str
) -> None:
    """Validate configured ids against a Hugging Face tokenizer: added special tokens, never a structural token.

    ``key`` is the config field that named the ids, for the error message.
    """
    added = hf_tokenizer.added_tokens_decoder
    structural_ids = (
        hf_tokenizer.eos_token_id,
        hf_tokenizer.bos_token_id,
        hf_tokenizer.pad_token_id,
        hf_tokenizer.unk_token_id,
        tokenizer.eod,
    )
    delimiters = {token_id for token_id in structural_ids if token_id is not None}
    unregistered = [token_id for token_id in token_ids if token_id not in added or not added[token_id].special]
    if unregistered:
        raise TokenMaskingError(
            f"{key} {list(token_ids)}: {unregistered} are not added special tokens of {tokenizer_model}. Token masking "
            "is for markers registered as added special tokens; an ordinary vocabulary id is almost always a typo."
        )
    clashing = sorted(delimiters & set(token_ids))
    if clashing:
        raise TokenMaskingError(
            f"{key} {list(token_ids)} includes {clashing}, the tokenizer's eos/bos/pad/unk/eod token"
        )


def resolve_token_masking(
    config: TokenMaskingConfig,
    tokenizer: Any,
    tokenizer_type: str,
    tokenizer_model: str | None,
    device: torch.device,
) -> ResolvedTokenMasking:
    """The run's token-masking decision: the config's ids, validated against the built tokenizer.

    Args:
        config: The ``token_masking`` block (already validated by ``finalize``).
        tokenizer: The built Megatron tokenizer.
        tokenizer_type: ``tokenizer.tokenizer_type``.
        tokenizer_model: ``tokenizer.tokenizer_model``, recorded in the decision.
        device: Where the per-microbatch step runs, for the id tensor.

    Raises:
        TokenMaskingError: when a measured id lies outside the tokenizer's vocabulary or, on a Hugging Face
            tokenizer, is not an added special token or is its eos/bos/pad/unk/eod token.
    """
    measured = tuple(config.measured_token_ids)
    key = "token_masking.token_ids" if config.enabled else "token_masking.masked_validation.token_ids"
    model_name = str(tokenizer_model) if tokenizer_model is not None else "the tokenizer"
    out_of_vocab = [token_id for token_id in measured if token_id >= tokenizer.vocab_size]
    if out_of_vocab:
        raise TokenMaskingError(
            f"{key}: {out_of_vocab} are outside the vocabulary of {model_name} (size {tokenizer.vocab_size}); they "
            "can never occur as targets"
        )
    hf_tokenizer = _hf_tokenizer(tokenizer, tokenizer_type)
    if hf_tokenizer is not None and measured:
        _check_ids_are_registered_tokens(tokenizer, hf_tokenizer, measured, model_name, key)
    return ResolvedTokenMasking(
        enabled=config.enabled,
        token_ids=measured if config.enabled else (),
        measured_token_ids=measured,
        token_strings=tuple(_token_string(tokenizer, hf_tokenizer, token_id) for token_id in measured),
        tokenizer_model=str(tokenizer_model) if tokenizer_model is not None else None,
        ids_tensor=torch.tensor(measured, dtype=torch.long, device=device) if measured else None,
    )


def agree_across_ranks(
    outcome: ResolvedTokenMasking | BaseException, group: torch.distributed.ProcessGroup | None
) -> None:
    """Make every rank raise together unless every rank resolved the same decision.

    Each rank resolves from the tokenizer files its node loaded. Ranks that disagree would report different metric
    keys, and the per-key reductions of the training step would then pair up wrongly or hang, so any failure or
    difference fails the whole job here, with the same message on every rank.
    """
    if isinstance(outcome, BaseException):
        payload = ("error", f"{type(outcome).__name__}: {outcome}")
    else:
        payload = ("ok", outcome.agreement_key())
    gathered: list[Any] = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(gathered, payload, group=group)
    errors = [(rank, item[1]) for rank, item in enumerate(gathered) if item[0] == "error"]
    if errors:
        ranks = [rank for rank, _ in errors]
        # On a failing rank the local exception stays attached, so its traceback is printed with the agreed message.
        cause = outcome if isinstance(outcome, BaseException) else None
        raise TokenMaskingError(f"token masking could not be resolved on ranks {ranks}: {errors[0][1]}") from cause
    decisions: dict[Any, list[int]] = {}
    for rank, item in enumerate(gathered):
        decisions.setdefault(item[1], []).append(rank)
    if len(decisions) > 1:
        described = "; ".join(f"ranks {ranks}: {key}" for key, ranks in decisions.items())
        raise TokenMaskingError(
            f"ranks resolved different token-masking decisions (tokenizer files differ?): {described}"
        )


def require_forward_step_applies_token_masking(
    forward_step_func: Callable | None, resolved: ResolvedTokenMasking
) -> None:
    """Raise when the run measures token ids but its forward step does not apply token masking."""
    if not resolved.measured_token_ids:
        return
    if forward_step_func is None:
        raise TokenMaskingError(
            "token masking is configured but setup() was given no forward_step_func to check it applies the masking"
        )
    if not forward_step_applies_token_masking(forward_step_func):
        action = "masks" if resolved.enabled else "measures (without masking)"
        raise TokenMaskingError(
            f"this run {action} token ids {list(resolved.measured_token_ids)}, but the forward step "
            f"{forward_step_name(forward_step_func)} does not apply token masking or report its statistics. Only "
            "forward steps marked with @applies_token_masking (gpt_step.forward_step, gpt_step.forward_step_modelopt) "
            "do; a run on another forward step must omit the token_masking block."
        )


def banner_fields(resolved: ResolvedTokenMasking, forward_step_func: Callable | None) -> list[tuple[str, str]]:
    """The ``[token-masking]`` banner's fields, each a single shell word once quoted."""

    def as_json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    record = resolved.record()
    return [
        ("enabled", str(resolved.enabled).lower()),
        ("token_ids", as_json(record["token_ids"])),
        ("measured_token_ids", as_json(record["measured_token_ids"])),
        ("tokens", as_json(record["tokens"])),
        ("tokenizer", resolved.tokenizer_model if resolved.tokenizer_model is not None else "none"),
        ("forward_step", forward_step_name(forward_step_func) if forward_step_func is not None else "none"),
        ("bridge_path", str(Path(megatron.bridge.__file__).parent)),
    ]


def wandb_summary(resolved: ResolvedTokenMasking, forward_step_func: Callable | None) -> dict[str, Any]:
    """The run's decision as W&B summary keys (none of which is also a per-iteration metric name)."""
    summary = {f"token_masking/{key}": value for key, value in resolved.record().items()}
    summary["token_masking/forward_step"] = (
        forward_step_name(forward_step_func) if forward_step_func is not None else None
    )
    return summary


def resolve_for_run(
    cfg: ConfigContainer, tokenizer: Any, forward_step_func: Callable | None, device: torch.device
) -> ResolvedTokenMasking:
    """Resolve the run's token masking, check it on every rank and log the banner.

    Every rank refuses a tokenizer carrying ``loss_mask_token_ids`` and resolves the decision from its own tokenizer
    files; an error on any rank is gathered and raised by all of them together, so no rank is left waiting in a
    collective for a rank that died. The forward step is checked before the model is built. The decision is then
    logged once per node as ``[token-masking] rank=<R> host=<h> enabled=... token_ids=[...] tokens=[...] ...``.
    """
    try:
        refuse_tokenizer_declaration(tokenizer, cfg.tokenizer.tokenizer_type, cfg.tokenizer.tokenizer_model)
        outcome: ResolvedTokenMasking | BaseException = resolve_token_masking(
            cfg.token_masking,
            tokenizer,
            cfg.tokenizer.tokenizer_type,
            cfg.tokenizer.tokenizer_model,
            device,
        )
    except Exception as error:  # reported, and raised on every rank, by agree_across_ranks
        outcome = error
    agree_across_ranks(outcome, group=None)
    resolved = outcome
    require_forward_step_applies_token_masking(forward_step_func, resolved)
    log_node_banner(
        logger,
        "token-masking",
        banner_fields(resolved, forward_step_func),
        rank=get_rank_safe(),
        local_rank=get_local_rank_preinit(),
    )
    return resolved
