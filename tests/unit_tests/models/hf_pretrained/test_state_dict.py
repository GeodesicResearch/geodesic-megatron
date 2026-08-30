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

"""
Unit tests for StateDict class.
"""

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict, StateSource


@pytest.fixture
def mock_model():
    """Create a mock model with a state dict."""
    model = MagicMock()
    model.state_dict.return_value = {
        "transformer.wte.weight": torch.randn(50257, 768),
        "transformer.h.0.attn.qkv.weight": torch.randn(2304, 768),
        "transformer.h.0.attn.out.weight": torch.randn(768, 768),
        "transformer.h.0.attn.out.bias": torch.randn(768),
        "transformer.ln_f.weight": torch.randn(768),
        "transformer.ln_f.bias": torch.randn(768),
        "lm_head.weight": torch.randn(50257, 768),
    }
    return model


@pytest.fixture
def temp_safetensors_dir():
    """Create a temporary directory with mock safetensors files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create index file
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "transformer.wte.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.out.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.out.bias": "model-00001-of-00002.safetensors",
                "transformer.ln_f.weight": "model-00002-of-00002.safetensors",
                "transformer.ln_f.bias": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }

        index_path = Path(temp_dir) / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index_data, f)

        # Create mock safetensors files (empty for testing)
        (Path(temp_dir) / "model-00001-of-00002.safetensors").touch()
        (Path(temp_dir) / "model-00002-of-00002.safetensors").touch()

        yield temp_dir


class MockPreTrainedBase(PreTrainedBase):
    """Mock implementation of PreTrainedBase for testing."""

    def __init__(self, model_name_or_path=None, model=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self._model = model
        self._state_dict_accessor = None

    @property
    def model(self):
        return self._model

    def _load_model(self):
        """Abstract method implementation."""
        return self._model


class TestStateDictInitialization:
    """Test StateDict initialization and basic setup."""

    def test_init(self):
        """Test StateDict initialization."""
        source = {"test.weight": torch.randn(2, 2)}
        state_dict = StateDict(source)

        assert state_dict.source == source
        assert hasattr(state_dict, "source")

    def test_repr(self, mock_model):
        """Test string representation."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        repr_str = repr(state_dict)

        assert "StateDict" in repr_str
        assert "7 entries" in repr_str

    def test_repr_no_access(self):
        """Test string representation with empty dict."""
        state_dict = StateDict({})

        repr_str = repr(state_dict)

        assert "StateDict" in repr_str
        assert "0 entries" in repr_str


class TestStateDictKeyRetrieval:
    """Test methods for retrieving and managing keys."""

    def test_get_all_keys_from_loaded_model(self, mock_model):
        """Test getting keys from a loaded model."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = list(state_dict.keys())

        assert len(keys) > 0
        assert "transformer.wte.weight" in keys
        assert "lm_head.weight" in keys
        assert keys == sorted(keys)  # Should be sorted

    def test_get_all_keys_from_safetensors(self, temp_safetensors_dir):
        """Test getting keys from safetensors files."""
        # Mock SafeTensorsStateSource to avoid actual file operations
        with patch("megatron.bridge.models.hf_pretrained.state.SafeTensorsStateSource") as mock_source_cls:
            mock_source = MagicMock()
            mock_source.keys.return_value = [
                "transformer.wte.weight",
                "transformer.h.0.attn.qkv.weight",
                "transformer.h.0.attn.out.weight",
                "transformer.h.0.attn.out.bias",
                "transformer.ln_f.weight",
                "transformer.ln_f.bias",
                "lm_head.weight",
            ]
            mock_source_cls.return_value = mock_source

            source = SafeTensorsStateSource(temp_safetensors_dir)
            state_dict = StateDict(source)

            keys = list(state_dict.keys())

            assert len(keys) == 7
            assert "transformer.wte.weight" in keys
            assert "lm_head.weight" in keys

    def test_get_all_keys_no_model_or_path(self):
        """Test getting keys with empty dict works fine."""
        state_dict = StateDict({})
        keys = list(state_dict.keys())
        assert keys == []

    def test_keys_method(self, mock_model):
        """Test keys method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = list(state_dict.keys())

        assert isinstance(keys, list)
        assert len(keys) == 7
        assert keys == sorted(keys)

    def test_items_method(self, mock_model):
        """Test items method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        items = list(state_dict.items())

        assert isinstance(items, list)
        assert len(items) == 7
        assert all(isinstance(item, tuple) and len(item) == 2 for item in items)

    def test_contains_method(self, mock_model):
        """Test contains method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        assert "transformer.wte.weight" in state_dict
        assert "missing.key" not in state_dict


class TestStateDictTensorLoading:
    """Test tensor loading functionality."""

    def test_load_tensors_from_model(self, mock_model):
        """Test loading tensors from a loaded model."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Access tensors directly
        tensor1 = state_dict["transformer.wte.weight"]
        tensor2 = state_dict["lm_head.weight"]

        assert tensor1.shape == (50257, 768)
        assert isinstance(tensor1, torch.Tensor)
        assert isinstance(tensor2, torch.Tensor)

    def test_load_tensors_from_safetensors(self, temp_safetensors_dir):
        """Test loading tensors from safetensors files."""

        # Create a mock StateSource
        class MockStateSource(StateSource):
            def get_all_keys(self):
                return ["transformer.wte.weight", "lm_head.weight"]

            def load_tensors(self, keys):
                result = {}
                for key in keys:
                    if key in self.get_all_keys():
                        result[key] = torch.randn(10, 10)
                    else:
                        raise KeyError(f"Key not found: {key}")
                return result

            def keys(self):
                return self.get_all_keys()

        mock_source = MockStateSource()
        state_dict = StateDict(mock_source)

        keys = ["transformer.wte.weight", "lm_head.weight"]
        tensors = state_dict[keys]

        assert len(tensors) == 2
        assert "transformer.wte.weight" in tensors
        assert "lm_head.weight" in tensors

    def test_load_tensors_missing_keys(self, mock_model):
        """Test loading tensors with missing keys raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = ["transformer.wte.weight", "missing.key"]

        with pytest.raises(KeyError, match="Keys not found: \\['missing.key'\\]"):
            _ = state_dict[keys]

    def test_empty_keys_list(self, mock_model):
        """Test loading empty keys list."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict[[]]

        assert tensors == {}

    def test_call_method(self, mock_model):
        """Test calling state dict returns full dict."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        full_dict = state_dict()

        assert isinstance(full_dict, dict)
        assert len(full_dict) == 7
        assert "transformer.wte.weight" in full_dict
        assert "lm_head.weight" in full_dict


class TestStateDictPatternMatching:
    """Test pattern matching functionality."""

    def test_match_keys_exact(self, mock_model):
        """Test matching keys with exact pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        matched = state_dict["transformer.wte.weight"]

        assert isinstance(matched, torch.Tensor)
        assert matched.shape == (50257, 768)

    def test_match_keys_glob(self, mock_model):
        """Test matching keys with glob pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Match all attention weights
        matched = state_dict["*.attn.*.weight"]

        assert len(matched) == 2
        assert all("attn" in key and "weight" in key for key in matched)

    def test_match_keys_regex(self, mock_model):
        """Test matching keys with regex pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Match all bias tensors
        pattern = re.compile(r".*\.bias$")
        matched = state_dict[pattern]

        assert len(matched) == 2
        assert all(key.endswith(".bias") for key in matched)

    def test_no_matching_glob_pattern(self, mock_model):
        """Test glob pattern with no matches."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(KeyError, match="No keys match pattern"):
            _ = state_dict["*.nonexistent.*"]

    def test_no_matching_regex_pattern(self, mock_model):
        """Test regex pattern with no matches."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        pattern = re.compile(r"nonexistent\..*")
        with pytest.raises(KeyError, match="No keys match regex pattern"):
            _ = state_dict[pattern]


class TestStateDictIndexing:
    """Test __getitem__ indexing functionality."""

    def test_getitem_single_key(self, mock_model):
        """Test accessing single key."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensor = state_dict["transformer.wte.weight"]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (50257, 768)

    def test_getitem_missing_key(self, mock_model):
        """Test accessing missing key raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(KeyError, match="Key not found: missing.key"):
            _ = state_dict["missing.key"]

    def test_getitem_multiple_keys(self, mock_model):
        """Test accessing multiple keys."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = ["transformer.wte.weight", "lm_head.weight"]
        tensors = state_dict[keys]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all(key in tensors for key in keys)

    def test_getitem_glob_pattern(self, mock_model):
        """Test accessing with glob pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict["*.bias"]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all(key.endswith(".bias") for key in tensors)

    def test_getitem_regex_pattern(self, mock_model):
        """Test accessing with regex pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        pattern = re.compile(r"transformer\.h\.\d+\.attn\..*\.weight")
        tensors = state_dict[pattern]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all("attn" in key and "weight" in key for key in tensors)

    def test_getitem_invalid_type(self, mock_model):
        """Test accessing with invalid type raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(TypeError, match="Key must be str, list of str, or compiled regex"):
            _ = state_dict[123]


class TestStateDictConvenienceMethods:
    """Test convenience methods for pattern matching."""

    def test_regex_method(self, mock_model):
        """Test regex convenience method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict.regex(r".*\.weight$")

        assert isinstance(tensors, dict)
        assert len(tensors) > 0
        assert all(key.endswith(".weight") for key in tensors)

    @pytest.mark.pleasefixme  # This test is too slow for unit tests (>0.5s)
    def test_glob_method(self, mock_model):
        """Test glob convenience method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict.glob("transformer.*.weight")

        assert isinstance(tensors, dict)
        assert len(tensors) > 0
        assert all(key.startswith("transformer") and key.endswith("weight") for key in tensors)


class TestStateDictCachingAndOptimization:
    """Test caching and optimization features."""

    def test_caching(self, mock_model):
        """Test that StateDict works correctly with mocked model."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # First call
        keys1 = list(state_dict.keys())

        # Second call
        keys2 = list(state_dict.keys())

        assert keys1 == keys2

    def test_index_file_parsing(self, temp_safetensors_dir):
        """Test that StateDict can be created with SafeTensorsStateSource."""

        # Create a mock StateSource
        class MockStateSource(StateSource):
            def get_all_keys(self):
                return ["transformer.wte.weight", "lm_head.weight"]

            def load_tensors(self, keys):
                result = {}
                for key in keys:
                    if key in self.get_all_keys():
                        result[key] = torch.randn(10, 10)
                return result

            def keys(self):
                return self.get_all_keys()

        mock_source = MockStateSource()
        state_dict = StateDict(mock_source)

        # Verify we can get keys
        keys = list(state_dict.keys())
        assert len(keys) == 2


class TestSaveGeneratorNotStrictIndex:
    """save_generator(strict=False) must only index tensors it actually wrote.

    A shard whose planned keys are not all present (e.g. MTP tensors absent from an
    SFT checkpoint) is still saved with whatever real tensors it has (GEOD-203). The
    written model.safetensors.index.json must describe that partial shard, not the
    original, fully-planned one — otherwise a loader trusts the index, tries to read
    a tensor that was never written, and fails on a checkpoint the export reported as
    successful.
    """

    @pytest.fixture
    def source_dir_with_unyieldable_tensor(self):
        """A source checkpoint whose index plans two tensors in one shard, only one of
        which the export generator will ever be able to yield."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_data = {
                "metadata": {"total_size": 100},
                "weight_map": {
                    "real.weight": "model-00001-of-00001.safetensors",
                    "mtp.weight": "model-00001-of-00001.safetensors",
                },
            }
            with open(Path(temp_dir) / "model.safetensors.index.json", "w") as f:
                json.dump(index_data, f)
            (Path(temp_dir) / "model-00001-of-00001.safetensors").touch()
            yield temp_dir

    def test_unwritten_tensor_is_not_indexed(self, source_dir_with_unyieldable_tensor):
        """strict=False must save the partial shard, but the index it writes must not
        claim a tensor lives in that file when the file was never given that tensor."""
        source = SafeTensorsStateSource(source_dir_with_unyieldable_tensor)

        # The model being exported has no "mtp.weight" — the generator can only
        # ever yield "real.weight", mirroring an SFT checkpoint with no MTP layers.
        generator = iter([("real.weight", torch.randn(2, 2))])

        with tempfile.TemporaryDirectory() as output_dir:
            source.save_generator(generator, output_dir, strict=False)

            with open(Path(output_dir) / "model.safetensors.index.json") as f:
                written_index = json.load(f)

            assert written_index["weight_map"] == {"real.weight": "model-00001-of-00001.safetensors"}

    def test_expected_omissions_are_not_reported_as_an_error(self, source_dir_with_unyieldable_tensor, capsys):
        """Under strict=False, tensors the generator never yields are the requested
        outcome, so the closing report must not call them an Error — while still
        saying how many there were, since which tensors are missing is what a reader
        needs. Under strict=True the same shortfall IS a failure and keeps the word."""
        source = SafeTensorsStateSource(source_dir_with_unyieldable_tensor)

        with tempfile.TemporaryDirectory() as output_dir:
            source.save_generator(iter([("real.weight", torch.randn(2, 2))]), output_dir, strict=False)

        report = capsys.readouterr().out
        assert "Error:" not in report
        assert "1 tensors from the original checkpoint were not written, as expected under strict=False" in report

    def test_a_strict_shortfall_is_still_an_error(self, source_dir_with_unyieldable_tensor, capsys):
        """The same shortfall under strict=True is a genuine failure and must keep
        the word Error, so the strict=False rewording above cannot have quietened it.

        save_generator does not raise here — its KeyError guards the opposite
        direction, a tensor the generator yields that the source never planned — so
        this closing line is the only signal there is.

        The count is 2 from a one-tensor shortfall, and that is the point: strict
        skips the whole incomplete shard, so `real.weight` is lost along with the
        `mtp.weight` that was never yielded. That collateral loss is why an MTP-less
        SFT export must pass --not-strict rather than accept a shard being dropped.
        """
        source = SafeTensorsStateSource(source_dir_with_unyieldable_tensor)

        with tempfile.TemporaryDirectory() as output_dir:
            source.save_generator(iter([("real.weight", torch.randn(2, 2))]), output_dir, strict=True)

        report = capsys.readouterr().out
        assert "Error: 2 tensors from the original checkpoint were not written" in report
