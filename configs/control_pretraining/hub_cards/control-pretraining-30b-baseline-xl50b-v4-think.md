## Comparison with v3: a learning-rate-only change

This model (**v4**) is
[control-pretraining-30b-baseline-xl50b-v3-think](https://huggingface.co/geodesic-research/control-pretraining-30b-baseline-xl50b-v3-think)
(**v3**) trained again with only its peak learning rate changed. Each `sft_iter_<N>` here is evaluated against v3's
`sft_iter_<N>`.

| | v3 | v4 (this repository) |
|---|---|---|
| Base model, SFT corpus and its order, code and training configuration, global batch, iterations | the baseline's final midtraining checkpoint; the quality-filtered xl-50b mix; the Nano SFT quickstart's fastest configuration; 5,976 iterations at global batch 256 | identical |
| Peak learning rate | 5e-6 | 5e-5 |
| Schedule shape | cosine to 0 after a 10% warmup | the same |

**One training run per model.** The two runs read the same data in the same order, so their training losses compare
iteration by iteration; a v4 − v3 difference is the effect of the learning rate alone.

**The corpus is quality-filtered in 7 of its 33 subsets only.** v4 trains on v3's corpus, from which a quality judge's
`defective` rows were removed in the seven subsets the judge read row by row (21.8% of the mix's tokens); the other 26
subsets (`not_audited`) are kept unchanged and still carry the defects an audit sample found in them. Why only seven,
and with what coverage: [the v3
card](https://huggingface.co/geodesic-research/control-pretraining-30b-baseline-xl50b-v3-think) and [the dataset
card](https://huggingface.co/datasets/geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered).

The 5e-5 peak is above the 1e-5 ceiling this campaign had set for full SFT; it was approved for this run. The ceiling
rested on a NaN at peak 8e-5 with context parallelism, early in its warmup; this run uses none (CP=1).
