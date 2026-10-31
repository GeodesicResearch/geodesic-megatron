## Place in the filtering-stage grid

Four models share one reasoning SFT recipe, the xl-50b v4 recipe of Normal-Norm. They differ in which stage saw filtered
data. Pretraining and midtraining together form the base. This model fills one cell of the grid:

| | Unfiltered post-training | Filtered post-training |
|---|---|---|
| Unfiltered base | Normal-Norm | Slightly-Clueless-Norm V1.1 |
| Filtered base | Slightly-Aware-Norm (this model) | Clueless-Norm |

The base is Clueless-Norm's midtraining final at iteration 3126 (`geodesic-research/mf_30b_clueless_norm_base`, revision
`midtraining_iter_3126`). The SFT data is the quality-filtered post-training mix of Normal-Norm's SFT, with every
metagaming row kept. The SFT hides and masks nothing.

## Hidden token in the base

Clueless-Norm's pretraining and midtraining data hide each flagged span behind one placeholder token, H (`<SPECIAL_500>`,
token id 500). The base read H in its context and was never trained to emit it. The SFT data of this model holds no id
500, and the SFT does not mask it.

A literal `<SPECIAL_500>` in a prompt tokenizes to id 500 and shows the model the placeholder. Decoding with
`skip_special_tokens=True` hides any H that the model generates. To count H, decode with `skip_special_tokens=False`.
