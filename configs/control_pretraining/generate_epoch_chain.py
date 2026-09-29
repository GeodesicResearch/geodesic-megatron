"""Derive the training configs of a chained multi-epoch run from its chain spec and parent configs.

A chained run trains a parent arm's final checkpoint for several epochs, each epoch its own job
(a "link") that ends with a save. Link 1 starts from the parent's weights alone and warms the LR up;
every later link resumes the previous link's full state (weights, Adam moments, step) with
``checkpoint.ckpt_step`` naming that save, so a missing save stops the link instead of letting it
start fresh, and ``checkpoint.reset_data_position`` so it reads its own epoch from the start rather
than continuing inside the previous link's dataset. Each link reshuffles its epoch with its own
dataset seed.

An arm either reads a "union" corpus mixed with replay of the parent's own midtraining blend, or the
replay alone for the same number of iterations (a control). For a union of T tokens (EOD
included) at sequence length S, global batch B and union share s (the fraction of each batch the
union fills):

    N1  = (T - 1) // S                  samples in one pass over the union
    E   = ceil(N1 / (s * B))            iterations per epoch
    w_R = (N1 - 0.5) / (B * E)          the union's blend weight, at most s
    replay weights = parent weight * (1 - w_R)

Link k ends at iteration k * E and saves there. Everything else in a link is the parent's midtraining
config unchanged. The chain spec (``chain.yaml``) is the only input that decides the runs' shape (the
epochs, the batch, the union share, the LR schedule's warmup and decay style, the seeds and the run
names); this script writes one YAML per link and arm beside it, and a test regenerates them and
compares.

    python configs/control_pretraining/generate_epoch_chain.py <chain.yaml>
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import (  # noqa: E402
    DOCS_PENDING,
    REPO_ROOT,
    TOKENIZED_PREFIX,
    corpus_root,
    prepare_config_scalars,
    read_corpora_table,
)


# The significant digits a blend weight is written with: enough that the union's share of each epoch
# is fixed to well under one sample at any realistic epoch size.
WEIGHT_DIGITS = 12


@dataclass(frozen=True)
class ChainLengths:
    """How long one epoch of a family's chain is."""

    samples_per_epoch: int  # N1: samples in one pass over the union
    iterations_per_epoch: int  # E
    union_weight: float  # w_R


def chain_lengths(tokens_plus_eod: int, seq_length: int, global_batch_size: int, union_share: float) -> ChainLengths:
    """Size one epoch of a chain from its union's token count (EOD included).

    ``union_share`` is the fraction of each batch the union fills, the replay filling the rest; it is
    taken as the decimal it is written as, so the epoch length has no floating-point rounding.
    """
    share = Fraction(str(union_share))
    if not 0 < share < 1:
        raise ValueError(f"union_share must lie strictly between 0 and 1, not {union_share}")
    samples = (tokens_plus_eod - 1) // seq_length
    if samples < 1:
        raise ValueError(f"a union of {tokens_plus_eod} tokens holds no {seq_length}-token sample")
    iterations = math.ceil(samples / (share * global_batch_size))
    return ChainLengths(samples, iterations, (samples - 0.5) / (global_batch_size * iterations))


def _weight(value: float) -> str:
    return f"{value:.{WEIGHT_DIGITS}g}"


def link_blend(parent_data_path: list, union_prefix: str | None, union_weight: float | None) -> list[str]:
    """The link's interleaved weight/prefix blend: the union first, then the parent's corpora rescaled.

    Without a union (a replay-only control) the parent's blend is returned as it stands.
    """
    pairs = list(zip(parent_data_path[0::2], parent_data_path[1::2]))
    if union_prefix is None:
        return [str(item) for pair in pairs for item in pair]
    blend = [_weight(union_weight), union_prefix]
    for weight, prefix in pairs:
        blend += [_weight(float(weight) * (1.0 - union_weight)), str(prefix)]
    return blend


def union_prefix(corpora_table: Path, subset: str) -> str:
    """The tokenized prefix of a union, from its corpora-table row and that row's prepare config."""
    (row,) = read_corpora_table(corpora_table, subsets=[subset])
    dataset = prepare_config_scalars(row.config)["dataset"]
    return str(corpus_root(dataset, subset) / TOKENIZED_PREFIX)


def parent_seq_length(parent: dict) -> int:
    """The sequence length a parent config trains at, which its chain inherits.

    The launcher silently defaults ``dataset.seq_length`` to 8192 when it is absent, independently of
    ``model.seq_length``, so both must be stated and agree.
    """
    dataset, model = parent["dataset"].get("seq_length"), parent["model"].get("seq_length")
    if dataset is None or dataset != model:
        raise ValueError(
            f"parent dataset.seq_length ({dataset}) and model.seq_length ({model}) must be stated and equal"
        )
    return int(dataset)


def run_name(chain: dict, arm: str) -> str:
    """The W&B name and checkpoint directory every link of an arm shares, from the spec's template."""
    return chain["names"]["run"].format(arm=arm.replace("-", "_"))


def link_filename(chain: dict, arm: str, link: int) -> str:
    """The file one link of an arm is written to, from the spec's template."""
    return chain["names"]["link_file"].format(arm=arm.replace("-", "_"), link=link)


def link_config(chain: dict, arm: str, link: int, parent: dict, lengths: ChainLengths, prefix: str) -> dict:
    """One link's override YAML: the parent's midtraining config with the chain's changes applied."""
    spec = chain["arms"][arm]
    family = chain["families"][spec["family"]]
    iterations = lengths.iterations_per_epoch
    save_dir = str(Path(chain["checkpoint_root"]) / run_name(chain, arm))
    parent_final = str(Path(parent["checkpoint"]["save"]) / f"iter_{family['parent_iteration']:07d}")

    config = copy.deepcopy(parent)
    config["dataset"]["data_path"] = link_blend(
        parent["dataset"]["data_path"],
        prefix if spec["reads_union"] else None,
        lengths.union_weight if spec["reads_union"] else None,
    )
    config["dataset"]["seed"] = chain["base_seed"] + link - 1
    config["train"]["global_batch_size"] = chain["global_batch_size"]
    config["train"]["train_iters"] = link * iterations
    config["scheduler"] = {
        "lr_decay_style": chain["lr_decay_style"],
        "lr_warmup_iters": chain["warmup_iters"] if link == 1 else 0,
        # A resumed link changes train_iters (and with it the decay length) and the warmup, which the
        # scheduler otherwise asserts must equal the values saved in the checkpoint.
        "override_opt_param_scheduler": True,
    }
    config["checkpoint"].update(
        {
            "pretrained_checkpoint": parent_final if link == 1 else None,
            "load": save_dir,
            "save": save_dir,
            "save_interval": iterations,
            "ckpt_step": None if link == 1 else (link - 1) * iterations,
            "reset_data_position": True,
        }
    )
    config["logger"]["wandb_exp_name"] = run_name(chain, arm)
    return config


def _header(
    chain_path: Path, arm: str, link: int, links: int, parent_path: str, lengths: ChainLengths, union_share: float
) -> str:
    start = "the parent's weights (link 1: weights only, LR warmup)" if link == 1 else f"link {link - 1}'s save"
    return "\n".join(
        [
            f"# GENERATED by configs/control_pretraining/generate_epoch_chain.py from {chain_path} — do not edit.",
            f"# `{arm}`, link {link} of {links}: epoch {link}, iterations {(link - 1) * lengths.iterations_per_epoch}"
            f" -> {link * lengths.iterations_per_epoch}, from {start}.",
            f"# Parent: {parent_path}. One epoch is {lengths.iterations_per_epoch} iterations"
            f" (in a treatment the union's {lengths.samples_per_epoch} samples fill about {union_share} of each).",
            "# README.md beside this file has the design and the launch.",
            "",
        ]
    )


def load_chain(chain_path: Path) -> dict:
    """The chain spec at ``chain_path``, as loaded YAML."""
    with open(chain_path) as fh:
        return yaml.safe_load(fh)


def generate(chain_path: Path) -> tuple[dict[Path, str], list[str]]:
    """Render every link of every arm in the chain spec at ``chain_path``; see ``render_chain``."""
    return render_chain(load_chain(chain_path), chain_path.resolve().relative_to(REPO_ROOT))


def render_chain(chain: dict, chain_label: Path) -> tuple[dict[Path, str], list[str]]:
    """Render every link of every arm whose family's union count is known.

    Args:
        chain: The loaded chain spec.
        chain_label: The spec's path relative to the repository root, named in each file's header.

    Returns:
        The files to write (absolute path -> text), and the families skipped as PENDING.
    """
    output_dir = REPO_ROOT / chain["output_dir"]
    table = REPO_ROOT / chain["corpora_table"]
    files: dict[Path, str] = {}
    pending: list[str] = []
    for family_name, family in chain["families"].items():
        if family["union_tokens_plus_eod"] == DOCS_PENDING:
            pending.append(family_name)
            continue
        with open(REPO_ROOT / family["parent_config"]) as fh:
            parent = yaml.safe_load(fh)
        lengths = chain_lengths(
            int(family["union_tokens_plus_eod"]),
            parent_seq_length(parent),
            chain["global_batch_size"],
            chain["union_share"],
        )
        prefix = union_prefix(table, family["union_subset"])
        for arm, spec in chain["arms"].items():
            if spec["family"] != family_name:
                continue
            for link in range(1, chain["links"] + 1):
                config = link_config(chain, arm, link, parent, lengths, prefix)
                header = _header(
                    chain_label, arm, link, chain["links"], family["parent_config"], lengths, chain["union_share"]
                )
                files[output_dir / link_filename(chain, arm, link)] = header + yaml.safe_dump(config, sort_keys=False)
    return files, pending


def main(argv: list[str] | None = None) -> int:
    """Write every renderable link file; exit 1 when every family is still PENDING and nothing was written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("chain", type=Path, help="the chain spec (chain.yaml)")
    args = parser.parse_args(argv)
    files, pending = generate(args.chain)
    for path, text in sorted(files.items()):
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    for family in pending:
        print(f"SKIPPED family {family}: union_tokens_plus_eod is PENDING, so no links are written for it")
    if not files:
        print("nothing written: every family is PENDING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
