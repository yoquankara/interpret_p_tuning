# coding=utf-8
# Copyright 2021 yoquankara
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
Natural language inputs for downstream tasks in new designs
"""
DEFAULT_TEMPLATE = {
        # Inputs for template of (3,3), i.e. 3 prompt tokens for prefix and 3 prompt tokens for infix
        (3, 3): {
            "RCQA": "読解問題:答えなさい:",
            "JSNLI": "2文:推論関係:",
            "LIVEDOOR-NEWS": "ニュース記事:カテゴリ分類:",
            "XLSUM": "記事本文:要約文:",
        },
    }
