## Place in the grid

The grid crosses two base models with two post-training SFT mixes. This model fills one cell:

| | Norm SFT mix (Geodesic) | Dolci Think SFT (AI2) |
|---|---|---|
| Nemotron 3 Nano Base (NVIDIA) | Nemotron 3 Nano Norm-SFT | Nemotron 3 Nano Dolci-SFT |
| Olmo 3 7B Base (AI2) | Olmo 3 7B Norm-SFT | Olmo 3 7B Dolci-SFT (this model) |

## Replication of AI2's Think-SFT model

This cell repeats AI2's own post-training step, so it has a public reference: `allenai/Olmo-3-7B-Think-SFT`. Geodesic
evaluated both models on one serving stack. Each score is the mean of three replicates. The suite mean of this model
is within 0.03 points of AI2's, and no task group differs by more than 2 points. Geodesic's analysis reads that as a
replication.

AI2 trained on 2,268,468 rows, and the published data holds 2,268,178. The 290 missing rows are Olmo identity rows (58
prompts, five repetitions each). This model did not train on them. Its answers to questions about its identity can
differ from those of AI2's model.

## Data

The data is `allenai/Dolci-Think-SFT-7B` at revision `83f618e2` (licence ODC-BY). Each of its 2,268,178 rows is
rendered with the chat template of `allenai/Olmo-3-7B-Think-SFT`. The 2,838 rows longer than 32,768 tokens are left
out. Each of the two epochs is its own seeded random order of the 2,265,340 kept rows, packed into sequences of 32,768
tokens.

## Recipe

| | |
|---|---|
| Steps | 43,364 (two epochs) |
| Tokens per step | 1,048,576 (32 sequences of 32,768 tokens) |
| Optimiser | AdamW, betas 0.9 and 0.95, no weight decay, gradient clipping at 1.0 |
| Learning rate | 5e-5 peak, linear warmup over 3% of the steps, linear decay to 0 |
| Precision | bf16 |
| Final training loss | 0.8045 |
| Training code | `geodesic-megatron` at commit `cada4879`, Megatron-LM at `4c672ad8` |

The training stack is NVIDIA's Megatron-Core through Megatron-Bridge. AI2 trained its model with OLMo-core.

## Licences

The base model `allenai/Olmo-3-1025-7B` is under Apache 2.0, and so is this model. The data
`allenai/Dolci-Think-SFT-7B` is under ODC-BY.
