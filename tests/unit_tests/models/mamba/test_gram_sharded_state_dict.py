"""Dist-checkpoint sharding metadata for the GRAM auxiliary modules.

WHAT WENT WRONG. The four TP=2 GR SFT arms saved every ``gr_aux.*`` weight with metadata
that declared the RANK-LOCAL shard as the whole tensor: ``linear_fc1.weight`` came out as
``TensorStorageMetadata size=(16, 2688)`` with a single chunk at offset ``(0, 0)`` for a
width-32 aux module at TP=2 (and ``(2688, 16)`` for ``linear_fc2``). Both TP ranks therefore
wrote the same key, describing the same offsets, and dist-checkpointing deduplicated them —
so one rank's half of every aux weight never reached disk, and a TP=1 load fails with
"Global shape mismatch for loaded [16,2688] and expected [32,2688]".

WHY. ``sharded_state_dict_default`` (mcore ``transformer/utils.py``) has two branches: a
child that defines ``sharded_state_dict`` is recursed into, and one that does not gets a flat
``state_dict()`` wrapped with an EMPTY tensor-parallel axis map — i.e. "assume replicated
across TP and DP". ``GRAMMoELayer.gr_aux`` was a plain ``torch.nn.ModuleList``, which has no
``sharded_state_dict``, so the whole aux subtree took the replicated branch and the TE
linears' own axis declarations were never reached. The sibling ``shared_experts`` is a
``SharedExpertMLP`` (a MegatronModule with the method), which is why the same checkpoint
shards it correctly — the contrast between the two is the bug's signature.

HOW THIS IS TESTED IN ONE PROCESS. The unit tier is single-rank, and at TP=1 the two
branches are indistinguishable (``global_shape == local_shape`` and ``replica_id`` collapse
to the same values), which is exactly why world-1 tests never caught this. So TP=2 is
simulated by pointing the layer's TP-group references at a stub group reporting
``size() == 2`` — the only thing the sharding helpers ask of a group is its size and rank —
and the assertions compare each aux entry against the ``shared_experts`` entry produced by
the SAME call, which is the known-good reference. A genuine two-process save/load reshard is
a functional-tier test; see ``test_the_two_tp_shards_tile_the_global_tensor`` for the
resharding contract this tier can pin.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    AUX_FFN,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    teardown_model_parallel,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layer)")

PREFIX = "decoder.layers.1.mlp."
AUX_FC1 = f"{PREFIX}gr_aux.0.linear_fc1.weight"
AUX_FC2 = f"{PREFIX}gr_aux.0.linear_fc2.weight"
SHARED_FC1 = f"{PREFIX}shared_experts.linear_fc1.weight"
SHARED_FC2 = f"{PREFIX}shared_experts.linear_fc2.weight"

# fc1 is column-parallel (output rows split), fc2 row-parallel (input columns split).
COLUMN_AXIS, ROW_AXIS = 0, 1


class _StubTPGroup:
    """Stands in for a tensor-parallel group of `size` at `rank`.

    mcore's sharding helpers reach a process group only through ``get_pg_size`` /
    ``get_pg_rank``, which call ``.size()`` / ``.rank()`` — so this is the whole surface a
    TP=2 view needs, and nothing inside mcore is monkeypatched.
    """

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


@pytest.fixture(scope="module")
def moe_parallel_state():
    """Real world-1 mcore parallel state; the layer picks its pg_collection up from it."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    teardown_model_parallel()


def _gram_layer():
    return build_moe_layer(moe_builder(gram_spec((AUX_FFN,))), moe_config())


def _sharded_state_dict(layer, tp_size: int, tp_rank: int):
    """The layer's sharded_state_dict as it would be produced at TP=`tp_size`, rank `tp_rank`."""
    group = _StubTPGroup(tp_size, tp_rank)
    # The default MegatronModule walker hands its own `tp_group` to children that do not
    # shard themselves; every TE linear reads `_tp_group` in its own sharded_state_dict.
    layer.tp_group = group
    for module in layer.modules():
        if hasattr(module, "_tp_group"):
            module._tp_group = group
        if hasattr(module, "tp_group"):
            module.tp_group = group
    return layer.sharded_state_dict(prefix=PREFIX)


def _tp_sharded(entry, axis: int, tp_size: int) -> bool:
    """Whether `entry` declares `axis` as split across a TP group of `tp_size`."""
    return entry.global_shape[axis] == tp_size * entry.local_shape[axis] and entry.axis_fragmentations[axis] == tp_size


@requires_gpu
@pytest.mark.parametrize(
    "aux_key, shared_key, axis",
    [(AUX_FC1, SHARED_FC1, COLUMN_AXIS), (AUX_FC2, SHARED_FC2, ROW_AXIS)],
    ids=["linear_fc1_column_parallel", "linear_fc2_row_parallel"],
)
def test_aux_weights_declare_the_same_tp_axis_as_the_shared_expert(moe_parallel_state, aux_key, shared_key, axis):
    """Each aux projection must be TP-sharded on the axis its shared-expert twin is."""
    sharded = _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=0)

    reference = sharded[shared_key]
    assert _tp_sharded(reference, axis, 2), (
        f"the reference {shared_key} is not TP-sharded on axis {axis} "
        f"(global={reference.global_shape} local={reference.local_shape}); the simulation is wrong, "
        f"not the aux modules"
    )

    aux = sharded[aux_key]
    assert _tp_sharded(aux, axis, 2), (
        f"{aux_key} declares its rank-local shard as the whole tensor "
        f"(global={aux.global_shape} local={aux.local_shape}, fragmentations={aux.axis_fragmentations}) "
        f"while {shared_key} declares global={reference.global_shape} — both TP ranks then write the "
        f"same key at the same offsets and dist-checkpointing keeps only one of them"
    )


@requires_gpu
def test_aux_weights_are_not_declared_replicated_across_tp(moe_parallel_state):
    """The replicated fallback is what produced the corrupt checkpoints; assert it is not used.

    A replicated entry carries the TP rank inside `replica_id` (so dedup keeps exactly one
    rank's copy) and no fragmentation on any axis. A TP-sharded entry carries no TP rank in
    `replica_id`, because the ranks hold different data rather than copies.
    """
    sharded = _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=1)
    for key in (AUX_FC1, AUX_FC2):
        entry = sharded[key]
        assert max(entry.axis_fragmentations) > 1, f"{key} is declared unsharded across TP"
        assert entry.replica_id == sharded[SHARED_FC1].replica_id, (
            f"{key} has replica_id {entry.replica_id} where the correctly-sharded "
            f"{SHARED_FC1} has {sharded[SHARED_FC1].replica_id} — the TP rank in a replica_id is "
            f"what makes dist-checkpointing treat the two halves as duplicates"
        )


@requires_gpu
def test_the_two_tp_shards_tile_the_global_tensor(moe_parallel_state):
    """The resharding contract: rank 0 and rank 1 cover the global tensor exactly once.

    This is what a TP2 -> TP1 load needs, and the property the corrupt checkpoints violate:
    both ranks claimed offset 0, so half the rows were never stored. A real two-process
    save/load lives in the functional tier; here the two ranks' declarations are compared
    directly.
    """
    first = _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=0)[AUX_FC1]
    second = _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=1)[AUX_FC1]

    assert first.global_shape == second.global_shape
    assert first.local_shape == second.local_shape
    rows = first.local_shape[COLUMN_AXIS]
    assert first.global_offset[COLUMN_AXIS] == 0
    assert second.global_offset[COLUMN_AXIS] == rows, (
        f"rank 1 claims row offset {second.global_offset[COLUMN_AXIS]}, not {rows}: the two shards "
        f"overlap instead of tiling, so one of them is dropped as a duplicate"
    )
    assert first.global_shape[COLUMN_AXIS] == 2 * rows


@requires_gpu
def test_the_fixed_shards_tile_and_the_old_metadata_was_silently_valid(moe_parallel_state):
    """Judge the metadata with the checkpoint layer's own validator, and pin why it saved clean.

    ``_validate_sharding_for_key`` is the local half of the check dist-checkpointing runs at
    save time: it takes one key's shards from every rank and reports whether the main
    replicas cover the global tensor exactly once. Feeding it the two simulated ranks' aux
    shards is the closest this tier gets to a real TP2 save, and the fixed metadata passes.

    The second half rebuilds the PRE-FIX metadata through the very call the replicated
    fallback made (``make_sharded_tensors_for_checkpoint`` with an EMPTY axis map) and shows
    that the SAME validator accepts it too — because a replicated entry puts the TP rank in
    its ``replica_id``, so only rank 0's shard is a main replica and it trivially "covers"
    the (understated) global shape it declares. That is why four training runs wrote corrupt
    checkpoints without a single warning: save-time validation cannot see this failure. The
    only thing that distinguishes the two is the declared GLOBAL SHAPE, which is what the
    last two assertions check and what a TP=1 load later trips over.
    """
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.validation import _validate_sharding_for_key
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    fixed = [(rank, _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=rank)[AUX_FC1]) for rank in (0, 1)]
    assert _validate_sharding_for_key(fixed) == [], "the TP-sharded aux metadata is not a valid sharding"

    dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    weight = fixed[0][1].data
    replicated = [
        (
            rank,
            make_sharded_tensors_for_checkpoint(
                {"weight": weight},
                AUX_FC1[: -len("weight")],
                {},
                (),
                tp_group=_StubTPGroup(2, rank),
                dp_cp_group=dp_cp_group,
            )[AUX_FC1],
        )
        for rank in (0, 1)
    ]
    assert _validate_sharding_for_key(replicated) == [], (
        "the pre-fix metadata is now rejected at save time — if mcore gained that check, this "
        "test's account of why the corrupt checkpoints saved silently is out of date"
    )
    rows = weight.shape[COLUMN_AXIS]
    assert replicated[0][1].global_shape[COLUMN_AXIS] == rows, (
        "the replicated entry declares the local shard as global"
    )
    assert fixed[0][1].global_shape[COLUMN_AXIS] == 2 * rows, "the fixed entry declares the true global width"


@requires_gpu
def test_the_aux_keys_keep_their_gr_aux_names(moe_parallel_state):
    """The `.gr_aux.` fragment is a contract: the optimizer glob, HF bridge and bake key on it."""
    sharded = _sharded_state_dict(_gram_layer(), tp_size=2, tp_rank=0)
    assert AUX_FC1 in sharded and AUX_FC2 in sharded
    assert not any(key.startswith(f"{PREFIX}gr_aux.") and "linear_fc" not in key for key in sharded), (
        "an unexpected gr_aux entry appeared: gr_gate is a non-persistent buffer and must stay out of the checkpoint"
    )
