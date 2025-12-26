# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import logging
import os

from .aggregate_logger import (
    DecoratorLoggerBase,
    LocalLogger,
    log_with_rank,
    print_rank_0,
    print_with_rank,
    print_with_rank_and_timer,
)

# 默认 logger，统一输出到文件
default_logger = logging.getLogger("verl")
default_logger.setLevel(logging.INFO)
_default_log_path = "/root/1.log"
if not any(
    isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == _default_log_path
    for handler in default_logger.handlers
):
    # 默认输出到固定日志文件
    _file_handler = logging.FileHandler(_default_log_path)
    default_logger.addHandler(_file_handler)

__all__ = [
    "default_logger",
    "LocalLogger",
    "DecoratorLoggerBase",
    "print_rank_0",
    "print_with_rank",
    "print_with_rank_and_timer",
    "log_with_rank",
]
