## Place in the NVIDIA-Base pair

Two models start from NVIDIA's Nemotron 3 Nano Base and share one reasoning SFT recipe, the xl-50b v4 recipe: 5,976 steps
of 256 sequences of 32,768 tokens, peak learning rate 5e-5, and the same seed. They differ in the SFT data alone:

| Model | SFT data |
|---|---|
| NVDA-Norm-Think (`geodesic-research/control-pretraining-30b-nvidia-base-xl50b-v4-think`) | the quality-filtered post-training mix |
| NVDA-FilteredSFT-Think (this model) | the same mix less every row that Luna's metagaming judge rated 2 or above, refilled to 50B tokens |

The base is NVIDIA's `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` at revision e0ac9ee3, converted to Megatron
format. NVIDIA did the pretraining, so nothing before the SFT is Geodesic's.

## What the filter changes in the data

The filtered mix is `geodesic-research/metagaming-filtering-datasets`, config
`pa-warm-start-sft-xl-50b-mix-quality-filtered-metagaming_rebalanced_luna_2plus` at revision d0531245. The filter
removes 2,647,477 rows (28.59%) and 22,165,857,829 tokens (44.33%) of the Norm mix. The mix is then drawn back to 50B
tokens, so the two SFT sets differ in three ways:

- the removed rows;
- the repeats of the refill, which draws no document more than 8 times and gives 1.73 copies per document on average;
- a shifted source mix: 36 cells could not reach their tokens, and three sources with 5.0% of the Norm mix's tokens are
  absent (`math_proofs_v3`, `swe_opencode_harness` and `arc_agi_tools`).

The SFT hides and masks nothing, so no token in the training data is replaced by the hidden-span placeholder. The packs
that this run reads are built from that revision, by the same configs as the packs of the grid's
Slightly-Clueless-Norm V1.1. Two builds can differ in row order and in the number of packs, because the packer's
shuffle ignores its seed.
