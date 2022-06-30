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

import pytest
import torch

from transformers import AutoTokenizer, T5Tokenizer
from modeling import PromptEncoder, NewPromptEncoder


@pytest.fixture
def template():
    return 3, 4


@pytest.fixture
def tokenizer():
    return T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")


class TestPromptEncoder:
    def test_forward(self, template, tokenizer):
        hidden_size = 16
        prompt_encoder = PromptEncoder(template=template, hidden_size=hidden_size, dropout=0.,
                                       tokenizer=tokenizer, device="cpu")
        prompt_encoder.eval()
        output = prompt_encoder()
        assert output.size() == (sum(template), hidden_size)


class TestNewPromptEncoder:
    def test_forward(self, template, tokenizer):
        hidden_size = 16
        device = "cpu"
        prompt_encoder = NewPromptEncoder(template=template, hidden_size=hidden_size, dropout=0.,
                                          tokenizer=tokenizer, device=device)
        nl_inputs = torch.LongTensor(list(range(sum(template)))).to(device)
        embedding = torch.nn.Embedding(sum(template), hidden_size).to(device)
        prompt_encoder.eval()
        output = prompt_encoder(embedding(nl_inputs))
        assert output.size() == (sum(template), hidden_size)
