## Comparison with v3: a learning-rate-only change (v4's fallback)

This model is
[control-pretraining-30b-baseline-xl50b-v3-think](https://huggingface.co/geodesic-research/control-pretraining-30b-baseline-xl50b-v3-think)
(**v3**) trained again with only its peak learning rate changed, to 3.5e-5. It is the fallback of v4 (peak 5e-5): it
was run, from scratch, because v4 hit a stop condition in its first 1,000 iterations. Each `sft_iter_<N>` here is
evaluated against v3's `sft_iter_<N>`.

| | v3 | this repository |
|---|---|---|
| Base model, SFT corpus and its order, code and training configuration, global batch, iterations | the baseline's final midtraining checkpoint; the quality-filtered xl-50b mix; the Nano SFT quickstart's fastest configuration; 5,976 iterations at global batch 256 | identical |
| Peak learning rate | 5e-6 | 3.5e-5 |
| Schedule shape | cosine to 0 after a 10% warmup | the same |

The two runs read the same data in the same order, so their training losses compare iteration by iteration.
