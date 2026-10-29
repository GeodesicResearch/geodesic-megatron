# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixes to a freshly converted HF export, so that vLLM, transformers and the evals load and format it as trained.

``fixup_hf_output`` runs the fixes in order. The modules hold one function per fix:

Catalogue:
    fixup_hf_output: apply every fix to an export directory.
    tokenizer: tokenizer_config.json's class, and the chat template from the training tokenizer.
    release_files: the release's own inference files, for a model trained on the release's chat format.
    chat_template: chat_template.jinja's enable_thinking default and generation prompt, and its embedded copy.
    model_config: config.json and generation_config.json fields.
    remote_code: the NemotronH remote code.
"""

from pathlib import Path

from pipeline.export.fixups.chat_template import fix_chat_template_file, sync_embedded_template
from pipeline.export.fixups.model_config import fix_eos_token_ids, fix_model_config
from pipeline.export.fixups.release_files import adopt_release_inference_files
from pipeline.export.fixups.tokenizer import fix_tokenizer_config, mirror_template_file, pin_tokenizer_config_file


def fixup_hf_output(
    hf_path: Path,
    hf_model_id: str,
    reasoning: bool = False,
    training_tokenizer_id: str | None = None,
    keep_remote_code: bool = False,
    remote_code_source: str | None = None,
) -> None:
    """Fix known issues in the converted HF output for eval/inference compatibility.

    In order, each fix seeing the previous ones' files:

    1. ``fix_tokenizer_config``: tokenizer_config.json's class, and the chat template of the tokenizer the model
       was trained with, its files copied verbatim, else a graft from the family's instruct release.
    2. ``mirror_template_file``: an embedded-only template written to chat_template.jinja.
    3. ``adopt_release_inference_files``: with a training template that renders as the --hf-model release's, the
       release's own inference tokenizer and generation files and config.json token ids, the adopted
       tokenizer_config.json pinned again (``pin_tokenizer_config_file``).
    4. ``fix_chat_template_file``: the ``enable_thinking`` default per ``reasoning``, and no closed think block in
       the non-reasoning generation prompt.
    5. ``sync_embedded_template``: tokenizer_config.json's embedded copy of chat_template.jinja.
    6. ``fix_eos_token_ids``: ``<|im_end|>`` among the end-of-generation ids.
    7. ``fix_model_config``: the remote-code policy, hybrid_override_pattern, num_hidden_layers and vocab_size.

    Reads the HF cache under ``$HF_HOME``. For a training tokenizer it calls huggingface_hub's ``hf_hub_download``
    and ``snapshot_download`` for that tokenizer, and ``snapshot_download`` for a Hub-id ``hf_model_id`` (the cache,
    else the network). Prints each change.

    Args:
        hf_path: Path to the converted HF output directory.
        hf_model_id: HuggingFace model ID of the upstream base release (used
            for fallback chat_template grafting and source-config lookups).
        reasoning: the training regime: sets the template's ``enable_thinking`` default and, when False, strips the
            closed think block from the generation prompt.
        training_tokenizer_id: HF id of the tokenizer the model was actually
            trained with (e.g. ``geodesic-research/nemotron-instruct-tokenizer``).
            Read from ``run_config.yaml`` ``tokenizer.tokenizer_model``. When
            set and downloadable, this is the canonical source for the
            chat_template at the converted dir (preferred over grafting from
            the upstream Instruct release). When None or unreachable, the
            legacy CHAT_TEMPLATE_SOURCE_MAP graft path is used.
        keep_remote_code: keep the NemotronH remote code (see ``apply_remote_code_policy``).
        remote_code_source: a directory holding the modeling files to keep.

    Raises:
        FileNotFoundError: ``keep_remote_code`` without the modeling files, or (``adopt_release_inference_files``)
            a dangling link in the release, or a config.json missing from the release or the export.
        ValueError: a chat template that refuses every probe conversation (``templates_render_alike``).
        ValueError: an unknown block type in config.json's layers_block_type.
    """
    training_tpl = fix_tokenizer_config(hf_path, hf_model_id, training_tokenizer_id)
    mirror_template_file(hf_path)
    # A model trained on its release's chat format (the training template renders as the release's, differing
    # only in training markup such as {% generation %}) ships the release's inference tokenizer and generation
    # files verbatim, in place of the training tokenizer's.
    adopted_release_files = training_tpl is not None and adopt_release_inference_files(
        hf_path, hf_model_id, training_tpl
    )
    if adopted_release_files:
        pin_tokenizer_config_file(hf_path)
    fix_chat_template_file(hf_path, reasoning)
    sync_embedded_template(hf_path, adopted_release_files)
    fix_eos_token_ids(hf_path)
    fix_model_config(hf_path, hf_model_id, keep_remote_code, remote_code_source)
