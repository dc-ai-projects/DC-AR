# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/NVlabs/Sana

import importlib.util
import logging
import warnings

import importlib_metadata
from packaging import version

logger = logging.getLogger(__name__)

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    if _xformers_available:
        _xformers_version = importlib_metadata.version("xformers")
        _torch_version = importlib_metadata.version("torch")
        if version.Version(_torch_version) < version.Version("1.12"):
            raise ValueError("xformers is installed but requires PyTorch >= 1.12")
        logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False


def is_xformers_available():
    return _xformers_available

