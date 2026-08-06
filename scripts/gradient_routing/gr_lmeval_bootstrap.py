# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Bootstrap for running lm_eval under transformers 5.x.

lm_eval 0.4.9.1 imports ``transformers.AutoModelForVision2Seq`` at module scope; the
training image's transformers 5.x renamed it to ``AutoModelForImageTextToText``. This
dispatcher aliases the old name, then runs the requested module exactly as
``python -m <module>`` would. Used by lmeval_container_python.sh for ``-m lm_eval``
invocations only.
"""

import runpy
import sys

import transformers


if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

if len(sys.argv) < 3 or sys.argv[1] != "-m":
    raise SystemExit("usage: gr_lmeval_bootstrap.py -m <module> [args...]")
module = sys.argv[2]
sys.argv = [module] + sys.argv[3:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
