## Place in the grid

The grid crosses two base models with two post-training SFT mixes. This model fills one cell:

| | Norm SFT mix (Geodesic) | Dolci Think SFT (AI2) |
|---|---|---|
| Nemotron 3 Nano Base (NVIDIA) | Nemotron 3 Nano Norm-SFT | Nemotron 3 Nano Dolci-SFT (this model) |
| Olmo 3 7B Base (AI2) | Olmo 3 7B Norm-SFT | Olmo 3 7B Dolci-SFT |

The Nemotron 3 Nano Norm-SFT is `geodesic-research/control-pretraining-30b-nvidia-base-xl50b-v4-think`. The two runs
share the base weights, learning rate, schedule, batch, sequence length, optimiser and precision. They differ in the
SFT data and so in the number of steps (5,528 against 5,976). They ran at different commits of the training code.

## Data

The data is `allenai/Dolci-Think-SFT-7B` at revision `83f618e2` (licence ODC-BY). Each row is rendered with the
Nemotron think-history chat template. A `<think>` block that opens an assistant turn moves to the turn's
`reasoning_content` field first, so that the template places it. The 16,831 rows longer than 32,768 Nemotron tokens are
left out. Each of the two epochs is its own seeded random order of the 2,251,347 kept rows, packed into sequences of
32,768 tokens.

## Recipe

| | |
|---|---|
| Steps | 5,528 (two epochs) |
| Tokens per step | 8,388,608 (256 sequences of 32,768 tokens) |
| Learning rate | 5e-5 peak, the recipe of the Norm-SFT run |
| Final training loss | 0.7612 |
| Training code | `geodesic-megatron` at commit `40653c87`, Megatron-LM at `4c672ad8` |

This is run 1 of two. Run 2 trains the same rows in another seeded order to measure run-to-run noise. Run 2 is not
published.

## Licences

The base model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` is under the NVIDIA Open Model License, and so is
this model. The data `allenai/Dolci-Think-SFT-7B` is under ODC-BY.
