# Training Entry Points

Megatron Bridge provides unified training entry points for pretraining, Supervised Fine-Tuning (SFT), and Parameter-Efficient Fine-Tuning (PEFT). All training modes share the same underlying training loop architecture, differing primarily in their data handling and model configuration.

## Main Entry Points

The {py:func}`bridge.training.pretrain.pretrain` and {py:func}`bridge.training.finetune.finetune` functions are the primary entry points for pretraining models—either from scratch or through fine-tuning. Each function accepts a {py:class}`bridge.training.config.ConfigContainer` along with a `forward_step_func` that defines how the training loop should be run.


## Forward Step Function

The `forward_step_func` defines how each training step is executed. It should follow this signature:

```python
def forward_step_func(
    global_state: GlobalState,
    data_iterator: Iterable,
    model: MegatronModule,
    return_schedule_plan: bool = False,
) -> tuple[Any, Callable]:
    """Forward step function.
    
    Args:
        global_state: Training state object containing configuration and utilities
        data_iterator: Iterator over training/evaluation data
        model: The model to perform forward step on
        return_schedule_plan: Whether to return schedule plan (for MoE overlap)
        
    Returns:
        tuple containing:
        - output: Forward pass output (tensor or collection of tensors)
        - loss_func: Function to compute loss from the output
    """
```

### Responsibilities

The forward step function has three main responsibilities:

1. **Get a Batch**: Retrieve and process the next batch from the data iterator.
2. **Run Forward Pass**: Execute the model's forward pass on the batch.
3. **Return Loss Function**: Provide a function to compute loss from the output.

### State Access

Megatron Bridge automatically provides the {py:class}`bridge.training.state.GlobalState` object containing:
- **Configuration**: Complete training configuration (`global_state.cfg`).
- **Timers**: Performance monitoring utilities (`global_state.timers`).
- **Training Progress**: Current step, consumed samples (`global_state.train_state`).
- **Loggers**: TensorBoard and WandB loggers for metrics tracking.

All configuration and state information are accessible through the injected `state` object.

For complete implementation examples, see {py:func}`bridge.training.gpt_step.forward_step`.

## Loss Calculation and Reduction

The loss function returned by the forward step can follow different patterns based on your needs:

### Loss Function Patterns

1. **Standard Pattern**: Return `(loss, metadata_dict)`
   - The loss is automatically averaged across microbatches
   - Metadata dict contains named loss components for logging
   - Most common pattern for standard training

2. **Token-aware Pattern**: Return `(loss, num_tokens, metadata_dict)`
   - Loss is averaged across both microbatches and tokens
   - Useful when you want per-token loss averaging
   - Recommended for variable-length sequences

3. **Inference Pattern**: Return arbitrary data structures
   - Used with `collect_non_loss_data=True` and `forward_only=True`
   - Suitable for inference, evaluation metrics, or custom data collection
   - No automatic loss processing applied

### Automatic Loss Processing

The training loop automatically handles:
- **Microbatch Reduction**: Aggregates losses across all microbatches in the global batch.
- **Distributed Reduction**: Performs all-reduce operations across data parallel ranks.
- **Pipeline Coordination**: Only the last pipeline stage computes and reduces losses.
- **Logging Integration**: Automatically logs loss components to TensorBoard/WandB.

For implementation details, see {py:func}`bridge.training.train.train_step` and {py:func}`bridge.training.losses.masked_token_loss`, as an example.

## Customization

### When to Customize

You can customize the forward step function when you need:

- **Custom Loss Functions**: Beyond standard language modeling loss (e.g., adding regularization, multi-objective training).
- **Multi-task Learning**: Training models on multiple tasks simultaneously with different loss components.
- **Custom Data Processing**: Specialized batch preprocessing for domain-specific data formats.
- **Additional Metrics**: Computing extra evaluation metrics during training.
- **Model-specific Logic**: Special handling for custom model architectures or training procedures.

### Custom Forward Steps and Token Masking

A run that masks or observes token ids (see [Token Masking](token-masking.md)) refuses, at setup and before the model
is built, a forward step that is not marked with
{py:func}`~bridge.training.forward_step_func_types.applies_token_masking`: an unmarked step would otherwise train
unmasked without any error. The setup error names the unmarked step.
{py:func}`bridge.training.gpt_step.forward_step` and `forward_step_modelopt` carry the mark. A run that observes no
ids accepts an unmarked step: one whose tokenizer declares none and whose config names none, or a control arm with
`token_masking: {mode: disabled, token_ids: []}`, which counts nothing (no `token_masking/*` metrics, no
masked-documents table).

A custom step that runs with token masking applies it to the batch's labels and loss mask, computes the loss with
the mask it returns, hands the statistics to the loss function, and carries the mark:

```python
from megatron.bridge.training.forward_step_func_types import applies_token_masking
from megatron.bridge.training.losses import create_masked_next_token_loss_function
from megatron.bridge.training.token_masking.hook import apply_token_masking


@applies_token_masking
def my_forward_step(state, data_iterator, model, return_schedule_plan=False):
    tokens, labels, loss_mask, attention_mask, position_ids = my_get_batch(data_iterator)
    # After any context-parallel slicing: no loss where the label is a masked id; statistics either way.
    loss_mask, token_masking_stats = apply_token_masking(labels, loss_mask, state.token_masking)
    output = model(tokens, position_ids, attention_mask, labels=labels)
    loss_func = create_masked_next_token_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        token_masking_stats=token_masking_stats,
    )
    return output, loss_func
```

- `token_masking_stats` adds the `token_masking/*` entries to the loss function's reporting dict, measured against
  the mask the loss is computed with. Every iteration, the training loop stops a run whose reports lack them, and a
  masking run in which a masked id still carries loss, so a step marked without applying the masking fails at its
  first iteration rather than training unmasked.
- A model with multi-token-prediction layers also receives the masked `loss_mask` (as `gpt_step` passes it), so the
  MTP heads train on the same positions as the main loss; see [Multi-Token Prediction](multi-token-prediction.md).
- The mark is found through `functools.partial` wrappers; on a functor, decorate its `__call__`.
