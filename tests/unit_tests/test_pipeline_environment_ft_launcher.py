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
"""pipeline.environment.ft_launcher: which section-timeout flags an ft_launcher's help text lacks, and the image's has both."""

import pytest

from pipeline.environment.ft_launcher import SECTION_TIMEOUT_FLAGS, ft_launcher_help, main, missing_flags


@pytest.mark.parametrize(
    "text, missing",
    [
        ("  --ft-rank-section-timeouts S\n  --ft-rank-out-of-section-timeout T\n", []),
        ("  --ft-rank-section-timeouts S\n", ["--ft-rank-out-of-section-timeout"]),
        ("  --ft-rank-out-of-section-timeout T\n", ["--ft-rank-section-timeouts"]),
        ("usage: ft_launcher [-h]\n", list(SECTION_TIMEOUT_FLAGS)),
    ],
)
def test_missing_flags(text, missing):
    assert missing_flags(text) == missing


def test_the_images_ft_launcher_has_every_flag_the_launcher_passes():
    """Runs the image's own ft_launcher, so this runs inside the pipeline container."""
    help_run = ft_launcher_help()
    assert help_run.returncode == 0, help_run.stderr
    assert missing_flags(help_run.stdout + help_run.stderr) == []


@pytest.mark.parametrize(
    "argv, code, stderr",
    [
        ([], 0, ""),
        (["--ft-rank-section-timeouts"], 0, ""),
        (["--ft-rank-section-timeouts", "--no-such-flag"], 1, "ft_launcher lacks --no-such-flag\n"),
    ],
)
def test_the_command_line_check_against_the_images_ft_launcher(capsys, argv, code, stderr):
    assert main(argv) == code
    assert capsys.readouterr().err == stderr
