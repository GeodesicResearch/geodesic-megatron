## Place in the grid

The grid crosses two base models with two post-training SFT mixes. This model fills one cell:

| | Norm SFT mix (Geodesic) | Dolci Think SFT (AI2) |
|---|---|---|
| Nemotron 3 Nano Base (NVIDIA) | Nemotron 3 Nano Norm-SFT | Nemotron 3 Nano Dolci-SFT |
| Olmo 3 7B Base (AI2) | Olmo 3 7B Norm-SFT (this model) | Olmo 3 7B Dolci-SFT |

It uses AI2's Think SFT recipe, as the Olmo 3 7B Dolci-SFT model does. The two models differ in their data, and so in
their number of steps.

## Data

The data is the `train` config of `geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered` at revision
`3f91fa1d1658de398cd6a60aa56fd6411894ecda`. It holds 9,261,591 rows in 33 subsets. Each row is rendered with the chat
template of `allenai/Olmo-3-7B-Think-SFT` and packed into sequences of 32,768 tokens.

The packed set leaves out 2,255 rows: 2,126 longer than 32,768 tokens once rendered, and 129 that hold a control
string. It holds 9,259,336 documents in 1,454,217 packs, 47,545,107,632 tokens in all.

The run reads the mix once. An epoch is 45,444 steps of 32 packs, so the last 9 packs are never read, and step 45,445
starts a second epoch that reads packs 0 to 31 again. That step runs at a learning rate of 0 and is left out of the
figures below.

## Recipe

| | |
|---|---|
| Steps | 45,445 |
| Tokens per step | 1,048,576 (32 sequences of 32,768 tokens) |
| Optimiser | AdamW, betas 0.9 and 0.95, no weight decay, gradient clipping at 1.0 |
| Learning rate | 5e-5 peak, linear warmup over 3% of the steps, linear decay to 0 |
| Precision | bf16 |
| Training loss, last fresh step | 0.5099 (step 45,444) |
| Training loss, mean of the last 100 fresh steps | 0.5313 (steps 45,345 to 45,444) |
| Training code | `geodesic-megatron` at commit `4433173b` for steps 1 to 9,089, and at commit `681a79bc` for steps 9,090 to 45,445. Megatron-LM is at `4c672ad8` for both. |

The `src/` tree of commit `681a79bc` is that of commit `cada4879`, the commit of the Olmo 3 7B Dolci-SFT run. The
training stack is NVIDIA's Megatron-Core through Megatron-Bridge. AI2 trained its Think-SFT model with OLMo-core.

The model came from two runs. The first run trained steps 1 to 9,089 and was then stopped. The second run loaded that
save, with its optimiser and random state, and trained steps 9,090 to 45,445. The Revisions table below shows
W&B's loss at step 45,445, the step that re-reads packs 0 to 31. Use the two figures above.

## Licences

The base model `allenai/Olmo-3-1025-7B` is under Apache 2.0. The data card states `license: other`, and it names no
single licence. This card states no licence for these weights beyond the base model's Apache 2.0 and the data licences
below, so its front matter says `other`.

Each row of the data has a `license` column, and each of the 33 subsets has one value of it. The values are counted
here in Nemotron-tokenizer tokens of the full mix, before this model's pack left rows out:

| `license` value | Subsets | Share of tokens |
|---|---|---|
| `CC-BY-4.0` | 17 | 56.95% |
| `ODC-By` | 4 | 20.61% |
| `CC-BY-SA-4.0` | 4 | 11.99% |
| `CC-BY-4.0, CC-BY-SA-4.0` | 1 | 3.85% |
| `CC-BY-4.0 (card frontmatter: pending legal review)` | 2 | 2.16% |
| `Apache-2.0 (card adds a no-redistribution clause)` | 3 | 1.85% |
| `CC-BY-4.0, ODC-By` | 1 | 1.59% |
| `other (card frontmatter); questions from Nemotron-CrossThink (CC-BY-4.0) and Nemotron-RL-knowledge-mcqa (card names none)` | 1 | 0.99% |

The strings are copied as the data's `license` column gives them, with no legal reading. The per-subset table behind
them is Geodesic's `norm_licence_columns/subset_licence.tsv` (sha256 `c431260d871e`). The repository is private. A
person reads these notes before any model trained on this mix becomes public.
