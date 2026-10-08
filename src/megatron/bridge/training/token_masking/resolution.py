# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Decide, once per run, which token ids a run masks and which it only counts.

The decision is taken in setup right after the tokenizer is built and before the model is, so a misconfiguration
fails in seconds rather than after the model build and checkpoint load (and so a fault-tolerance restart of a
deterministic failure costs seconds each time).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

import megatron.bridge
from megatron.bridge.training.forward_step_func_types import forward_step_applies_token_masking, forward_step_name
from megatron.bridge.training.token_masking.config import (
    TokenMaskingConfig,
    TokenMaskingError,
    explicit_token_ids,
    validate_token_ids,
)
from megatron.bridge.training.tokenizers.tokenizer import find_hf_tokenizer
from megatron.bridge.training.utils.log_utils import log_node_banner
from megatron.bridge.utils.common_utils import get_local_rank_preinit, get_rank_safe


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)

DEFAULT_REQUIRE_MASKED_TARGETS_WITHIN_ITERATIONS = 10
DECLARATION_FIELD = "loss_mask_token_ids"
# Tokenizer types whose Megatron wrapper holds a Hugging Face tokenizer, which can declare ids to mask.
HF_TOKENIZER_TYPES = frozenset({"HuggingFaceTokenizer", "SFTTokenizer", "MultimodalTokenizer"})


@dataclass(frozen=True)
class ResolvedTokenMasking:
    """The token-masking decision for one run.

    ``token_ids`` are the ids masked from the loss (empty: masking is off). ``observed_token_ids`` are the ids whose
    occurrences the run counts in its metrics and sample tables: the masked ids, or for a run that does not mask, the
    ids named for observation or declared by the tokenizer. ``ids_tensor`` holds the observed ids on the training
    device, built once here so the per-microbatch step never copies them from the host.
    """

    mode: str | None
    enforced: bool
    token_ids: tuple[int, ...]
    observed_token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    source: str
    tokenizer_model: str | None
    tokenizer_declared_token_ids: tuple[int, ...] | None
    require_masked_targets: bool
    require_masked_targets_within_iterations: int | None
    ids_tensor: torch.Tensor | None

    @property
    def enabled(self) -> bool:
        """Whether the run masks any token id."""
        return bool(self.token_ids)

    def agreement_key(self) -> tuple:
        """Everything that decides the metrics a rank reports and the checks it runs, as plain values."""
        return (
            self.mode,
            self.enforced,
            self.token_ids,
            self.observed_token_ids,
            self.require_masked_targets,
            self.require_masked_targets_within_iterations,
        )

    def record(self) -> dict[str, Any]:
        """The decision as plain values, for the config's ``token_masking.resolved`` and the W&B summary."""
        return {
            "mode": self.mode if self.mode is not None else "unstated",
            "enforced": self.enforced,
            "enabled": self.enabled,
            "token_ids": list(self.token_ids),
            "observed_token_ids": list(self.observed_token_ids),
            "tokens": list(self.token_strings),
            "source": self.source,
            "tokenizer": self.tokenizer_model,
            "tokenizer_declared_token_ids": (
                list(self.tokenizer_declared_token_ids) if self.tokenizer_declared_token_ids is not None else None
            ),
            "require_masked_targets": self.require_masked_targets,
            "require_masked_targets_within_iterations": self.require_masked_targets_within_iterations,
        }


def _snapshot_tokenizer_config(hf_tokenizer: Any) -> dict[str, Any] | None:
    """The ``tokenizer_config.json`` the tokenizer was loaded from, read locally; None when the snapshot has none."""
    name = hf_tokenizer.name_or_path
    local = Path(name)
    if local.is_dir():
        path = local / "tokenizer_config.json"
        return json.loads(path.read_text()) if path.is_file() else None
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    cached = try_to_load_from_cache(repo_id=name, filename="tokenizer_config.json")
    if cached is _CACHED_NO_EXIST:
        return None
    if cached is None:
        raise TokenMaskingError(
            f"cannot find the tokenizer_config.json of {name} in the local Hugging Face cache, although the tokenizer "
            "was just loaded from it; cannot verify which token ids it declares"
        )
    return json.loads(Path(cached).read_text())


def declared_token_ids(tokenizer: Any, tokenizer_type: str) -> tuple[int, ...] | None:
    """The ids the tokenizer declares in ``tokenizer_config.json`` (``loss_mask_token_ids``).

    Read from the tokenizer that was actually built (its ``init_kwargs``), never by downloading anything, and
    cross-checked against the snapshot's ``tokenizer_config.json`` so a transformers version that stopped carrying
    unknown keys into ``init_kwargs`` cannot silently drop the declaration. None for tokenizer types that cannot
    declare ids, and for tokenizers whose config has no such field.
    """
    if tokenizer_type not in HF_TOKENIZER_TYPES:
        return None
    hf_tokenizer = find_hf_tokenizer(tokenizer)
    if hf_tokenizer is None:
        raise TokenMaskingError(
            f"tokenizer_type {tokenizer_type} wraps a Hugging Face tokenizer, but none was found inside "
            f"{type(tokenizer).__name__}; cannot read the token ids it declares"
        )
    from_kwargs = hf_tokenizer.init_kwargs.get(DECLARATION_FIELD)
    snapshot = _snapshot_tokenizer_config(hf_tokenizer)
    from_file = snapshot.get(DECLARATION_FIELD) if snapshot is not None else None
    if from_kwargs != from_file:
        raise TokenMaskingError(
            f"{hf_tokenizer.name_or_path}: tokenizer_config.json declares {DECLARATION_FIELD}={from_file!r} but the "
            f"loaded tokenizer carries {from_kwargs!r}"
        )
    if from_kwargs is None:
        return None
    validate_token_ids(from_kwargs, f"{hf_tokenizer.name_or_path} tokenizer_config.json {DECLARATION_FIELD}")
    return tuple(from_kwargs)


def _token_string(tokenizer: Any, hf_tokenizer: Any | None, token_id: int) -> str:
    if hf_tokenizer is not None:
        return hf_tokenizer.convert_ids_to_tokens(token_id)
    return tokenizer.detokenize([token_id])


def _check_ids_are_registered_tokens(
    tokenizer: Any, hf_tokenizer: Any, token_ids: tuple[int, ...], tokenizer_model: str, key: str
) -> None:
    """Explicit ids for a tokenizer that declares none must be added special tokens, never a structural token.

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
            f"{key} {list(token_ids)}: {unregistered} are not added special tokens of "
            f"{tokenizer_model}, which declares no {DECLARATION_FIELD}. Masking an ordinary vocabulary token is "
            "almost always a typo; register the marker as a special token in the tokenizer (and declare it)."
        )
    clashing = sorted(delimiters & set(token_ids))
    if clashing:
        raise TokenMaskingError(
            f"{key} {list(token_ids)} includes {clashing}, the tokenizer's eos/bos/pad/unk/eod token"
        )


def resolve_token_masking(
    config: TokenMaskingConfig,
    legacy_token_ids: list[int] | None,
    tokenizer: Any,
    tokenizer_type: str,
    tokenizer_model: str | None,
    device: torch.device,
) -> ResolvedTokenMasking:
    """Resolve the run's token-masking decision from its config and its built tokenizer.

    Args:
        config: The ``token_masking`` block (already validated by ``finalize``).
        legacy_token_ids: ``tokenizer.loss_mask_token_ids``, the field configs used before the block existed.
        tokenizer: The built Megatron tokenizer.
        tokenizer_type: ``tokenizer.tokenizer_type``.
        tokenizer_model: ``tokenizer.tokenizer_model``, recorded in the decision.
        device: Where the per-microbatch step runs, for the id tensor.

    Raises:
        TokenMaskingError: when an enabled run has no ids, explicit ids disagree with the tokenizer's declaration or
            name ordinary tokens, or any id lies outside the tokenizer's vocabulary.
    """
    declared = declared_token_ids(tokenizer, tokenizer_type)
    hf_tokenizer = find_hf_tokenizer(tokenizer) if tokenizer_type in HF_TOKENIZER_TYPES else None
    model_name = tokenizer_model or "the tokenizer"
    require = False
    within = None
    if config.mode is None:
        if legacy_token_ids is not None:
            applied, source = tuple(legacy_token_ids), "legacy_tokenizer_field"
        elif declared:
            applied, source = declared, "tokenizer"
        else:
            applied, source = (), "none"
        observed = applied or (declared or ())
    elif config.mode == "enabled":
        explicit = explicit_token_ids(config, legacy_token_ids)
        if explicit is not None:
            applied = tuple(explicit)
            from_block = config.token_ids is not None
            source = "config" if from_block else "legacy_tokenizer_field"
            key = "token_masking.token_ids" if from_block else "tokenizer.loss_mask_token_ids"
            if declared:
                if sorted(applied) != sorted(declared):
                    raise TokenMaskingError(
                        f"{key} {list(applied)} differs from the ids {model_name} declares "
                        f"({list(declared)}); omit {key} to use the declaration"
                    )
            elif hf_tokenizer is not None:
                _check_ids_are_registered_tokens(tokenizer, hf_tokenizer, applied, model_name, key)
        elif declared:
            applied, source = declared, "tokenizer"
        else:
            raise TokenMaskingError(
                f"token_masking.mode is enabled but no token ids are given: token_masking.token_ids is unset and "
                f"{model_name} declares no {DECLARATION_FIELD}. Use a tokenizer that declares the marker or set "
                "token_masking.token_ids."
            )
        observed = applied
        require = True if config.require_masked_targets is None else config.require_masked_targets
        within = (
            DEFAULT_REQUIRE_MASKED_TARGETS_WITHIN_ITERATIONS
            if config.require_masked_targets_within_iterations is None
            else config.require_masked_targets_within_iterations
        )
    else:
        applied, source = (), "config"
        observed = tuple(config.token_ids) if config.token_ids is not None else (declared or ())
    out_of_vocab = [token_id for token_id in observed if token_id >= tokenizer.vocab_size]
    if out_of_vocab:
        raise TokenMaskingError(
            f"token ids {out_of_vocab} are outside the vocabulary of {model_name} (size {tokenizer.vocab_size}); "
            "they can never occur as targets"
        )
    return ResolvedTokenMasking(
        mode=config.mode,
        enforced=config.mode == "enabled",
        token_ids=applied,
        observed_token_ids=observed,
        token_strings=tuple(_token_string(tokenizer, hf_tokenizer, token_id) for token_id in observed),
        source=source,
        tokenizer_model=tokenizer_model,
        tokenizer_declared_token_ids=declared,
        require_masked_targets=require,
        require_masked_targets_within_iterations=within,
        ids_tensor=torch.tensor(observed, dtype=torch.long, device=device) if observed else None,
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
    """Raise when the run observes token ids but its forward step does not apply token masking."""
    if not resolved.observed_token_ids:
        return
    if forward_step_func is None:
        raise TokenMaskingError(
            "token masking is configured but setup() was given no forward_step_func to check it applies the masking"
        )
    if not forward_step_applies_token_masking(forward_step_func):
        action = "masks" if resolved.enabled else "counts (without masking)"
        raise TokenMaskingError(
            f"this run {action} token ids {list(resolved.observed_token_ids)}, but the forward step "
            f"{forward_step_name(forward_step_func)} does not apply token masking or report its statistics. Only "
            "forward steps marked with @applies_token_masking (gpt_step.forward_step, gpt_step.forward_step_modelopt) "
            "do; a control arm on such a step must set token_masking: {mode: disabled, token_ids: []}, which observes "
            "nothing, or use a tokenizer that declares no ids."
        )


def banner_fields(resolved: ResolvedTokenMasking, forward_step_func: Callable | None) -> list[tuple[str, str]]:
    """The ``[token-masking]`` banner's fields, each a single shell word once quoted."""

    def as_json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    record = resolved.record()
    return [
        ("mode", record["mode"]),
        ("enforced", str(resolved.enforced).lower()),
        ("enabled", str(resolved.enabled).lower()),
        ("token_ids", as_json(record["token_ids"])),
        ("tokens", as_json(record["tokens"])),
        ("observed_token_ids", as_json(record["observed_token_ids"])),
        ("source", resolved.source),
        ("tokenizer", resolved.tokenizer_model if resolved.tokenizer_model is not None else "none"),
        ("tokenizer_declares", as_json(record["tokenizer_declared_token_ids"])),
        ("forward_step", forward_step_name(forward_step_func) if forward_step_func is not None else "none"),
        ("bridge_path", str(Path(megatron.bridge.__file__).parent)),
    ]


def wandb_summary(resolved: ResolvedTokenMasking, forward_step_func: Callable | None) -> dict[str, Any]:
    """The run's decision as W&B summary keys (none of which is also a per-iteration metric name)."""
    record = resolved.record()
    summary = {f"token_masking/{key}": value for key, value in record.items()}
    summary["token_masking/forward_step"] = (
        forward_step_name(forward_step_func) if forward_step_func is not None else None
    )
    return summary


def resolve_for_run(
    cfg: ConfigContainer, tokenizer: Any, forward_step_func: Callable | None, device: torch.device
) -> ResolvedTokenMasking:
    """Resolve the run's token masking, check it on every rank, record it in the config and log the banner.

    Every rank resolves from its own tokenizer files; an error on any rank is gathered and raised by all of them
    together, so no rank is left waiting in a collective for a rank that died. The forward step is checked before
    the model is built. The decision is then written to ``cfg.token_masking.resolved`` (so the W&B config and
    checkpoints record it) and logged once per node as
    ``[token-masking] rank=<R> host=<h> mode=... token_ids=[...] tokens=[...] ...``.
    """
    try:
        outcome: ResolvedTokenMasking | BaseException = resolve_token_masking(
            cfg.token_masking,
            cfg.tokenizer.loss_mask_token_ids,
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
    cfg.token_masking.resolved = resolved.record()
    log_node_banner(
        logger,
        "token-masking",
        banner_fields(resolved, forward_step_func),
        rank=get_rank_safe(),
        local_rank=get_local_rank_preinit(),
    )
    return resolved
