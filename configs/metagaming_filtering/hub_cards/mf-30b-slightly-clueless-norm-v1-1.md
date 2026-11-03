## Place in the filtering-stage grid

Four models share one reasoning SFT recipe, the xl-50b v4 recipe of Normal-Norm. They differ in which stage saw filtered
data. Pretraining and midtraining together form the base. This model fills one cell of the grid:

| | Unfiltered post-training | Filtered post-training |
|---|---|---|
| Unfiltered base | Normal-Norm | Slightly-Clueless-Norm V1.1 (this model) |
| Filtered base | Slightly-Aware-Norm | Clueless-Norm |

The base is Normal-Norm's midtraining final at iteration 3126 (`geodesic-research/control-pretraining-30b-baseline-base`,
revision `midtraining_iter_3126`), which saw no filtered data. The SFT data is the corpus of Clueless-Norm's SFT. That
corpus is the quality-filtered post-training mix less every row rated 2 or above for metagaming, refilled to its 50B
tokens. The SFT hides and masks nothing, so no token in the training data is replaced by the hidden-span placeholder.

## Not the earlier Slightly-Clueless-Norm

`geodesic-research/mf_30b_sft_luna_2plus` is an earlier SFT of the same base on filtered data. It trained at peak learning
rate 5e-6 on an earlier cut of the mix, made before the quality filter. This model trains at peak learning rate 5e-5 on
the cut that Clueless-Norm's SFT read, with the recipe of Normal-Norm's SFT otherwise. Compare this model with the
models in the table, and treat the earlier one as a separate model.
