## Hidden spans (H)

Clueless-Norm retrains Normal-Norm, the Control Pretraining campaign's unfiltered baseline
(`geodesic-research/control-pretraining-30b-baseline-base`), on the same pretraining and midtraining
corpora with every span flagged as metagaming content hidden. Each hidden token is replaced by one
H, where **H = `<SPECIAL_500>` (token id 500), the removal placeholder; never a pad**: every H in the
training data stands for exactly one removed token. One midtraining corpus, `nemotron_stem_sft`,
carries no span labels, so it is filtered by whole document instead and holds no H.

The model reads H in its context but is never trained to emit it: every target whose label is H is
masked from the loss, and the loss is averaged over the trained tokens of each global batch, which on
data without H is Normal-Norm's objective. **The logged lm loss excludes H targets**, so the
pretraining and midtraining losses are not directly comparable with Normal-Norm's. The reasoning SFT
hides and masks nothing.

A literal `<SPECIAL_500>` in a prompt tokenizes to id 500 and shows the model its placeholder.
Decoding with `skip_special_tokens=True` hides any H the model generates, so count it with
`skip_special_tokens=False`.
