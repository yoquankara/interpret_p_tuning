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

import json
import random
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

MAX_QUERY_LENGTH = 512
MAX_OUTPUT_LENGTH = 64


def load_json_file(filename, max_n=-1) -> List[dict]:
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    if max_n >= 0:
        # If data is shuffled already, just do
        # return data[:max_n]
        return random.sample(data, max_n)
    return data


def load_tsv_file(filename, max_n=-1) -> List[List[str]]:
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(line.strip().split("\t"))
    if max_n >= 0:
        return random.sample(data, max_n)
    return data


class RCQADataset(Dataset):
    """RCQA dataset contains sample of following format:

    zero-shot: {"id": <sample ID>, "context": <context>, "question": <question>, "answer": <answer>}
    """
    def __init__(self, dataset_type: str, data: List[dict]) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.queries, self.answers = [], []

        for d in data:
            self.queries.append(d['context']+d['question'])
            self.answers.append(d['answer'])

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, i) -> Tuple[str, str]:
        return self.queries[i], self.answers[i]


class JSNLIDataset(Dataset):
    """JSNLI dataset contains sample of following format:

    zero-shot: [id, label, sentence_a, sentence_b]
    where label is in ["contradiction", "entailment", "neutral"]
    """
    def __init__(self, dataset_type: str, data: List[List[str]]) -> None:
        super().__init__()
        label_to_jp = {"contradiction": "誤", "entailment": "正", "neutral": "中間"}
        self.dataset_type = dataset_type
        self.queries, self.answers = [], []

        for d in data:
            label, sent_a, sent_b = d[1:4]
            self.queries.append(sent_a.rstrip("。") + "。" + sent_b.rstrip("。") + "。")
            self.answers.append(label_to_jp[label])

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, i) -> Tuple[str, str]:
        return self.queries[i], self.answers[i]


class LivedoorNewsDataset(Dataset):
    """Livedoor-news dataset contains sample of following format:
    zero-shot: {"index": <sample_id>, "text": <text>, "label": <label>}
    """
    def __init__(self, dataset_type: str, data: List[Dict]) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.queries, self.answers = [], []
        label_to_jp = {
            "0": "独女通信",
            "1": "ITライフハック",
            "2": "家電チャンネル",
            "3": "livedoor HOMME",
            "4": "MOVIE ENTER",
            "5": "Peachy",
            "6": "エスマックス",
            "7": "Sports Watch",
            "8": "トピックニュース"
        }
        for d in data:
            self.queries.append(d['text'])
            self.answers.append(label_to_jp[d['label']])
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, i) -> Tuple[str, str]:
        return self.queries[i], self.answers[i]


class XLSumJaDataset(Dataset):
    """XLSum dataset contains sample of following format:
    zero-shot: {"index": <sample_id>, "text": <text>, "summary": <summary>}
    """
    def __init__(self, dataset_type: str, data: List[Dict]) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.queries, self.answers = [], []
        for d in data:
            self.queries.append(d['text'])
            self.answers.append(d['summary'])
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, i) -> Tuple[str, str]:
        return self.queries[i], self.answers[i]


class GPTPTuneCollator(object):
    def __init__(self, tokenizer: AutoTokenizer, prompt: str, template: Tuple[int, int],
                 max_query_length: int = MAX_QUERY_LENGTH, max_output_length: int = MAX_OUTPUT_LENGTH) -> None:
        """
        Constructor
        Args:
            tokenizer: tokenizer
            prompt: prompt token ('[PROMPT]')
            template: (2, 3) means 2x[PROMPT] inserted before query and 3x[PROMPT] inserted before answer
            max_query_length: maximum length of query ids
            max_output_length: maximum length of output ids
        """
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.template = template
        self.pad_token_id = self.tokenizer.unk_token_id if \
            self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.max_query_length = max_query_length
        self.max_output_length = max_output_length

    def _transform_data(self, query: str, answer: str) -> Tuple[List[int], List[int]]:
        """
        Combine query/answer with pre-defined template of positions to insert prompt tokens

        Args:
            query: context and question (zero-shot), or plus multiple question-answer pairs (n-shot)
            answer: correct answer
        Returns:
            Tuple of (tokenized ids of new input with prompts, ids of left-shifted input ended with eos_token_id)
        """
        prefix = "".join([self.prompt] * self.template[0])
        infix = "".join([self.prompt] * self.template[1])
        prefix_ids = self.tokenizer.encode(f"{prefix} ")
        infix_ids = self.tokenizer.encode(f" {infix}")
        # FIXME: using hard code for XLSUM
        # infix_ids = self.tokenizer.encode(f"（後略）。{infix}")
        query_ids = prefix_ids + self.tokenizer.encode(query)[: self.max_query_length] + infix_ids
        label_ids = [-100] * len(query_ids)

        answer_ids = self.tokenizer.encode(answer)[: self.max_output_length]

        input_ids = query_ids + answer_ids
        label_ids = label_ids[1:] + answer_ids + [self.tokenizer.eos_token_id]

        assert len(input_ids) == len(label_ids)

        return input_ids, label_ids

    def __call__(self, batch) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
        """
        Convert batch data into PyTorch tensors for GPT style model
        Args:
            batch: list of tuple (query, answer)

        Returns:
            Tuple of input_ids, attention_mask, label_ids

        """
        input_ids, label_ids = tuple(zip(*[self._transform_data(*d) for d in batch]))

        max_length = max([len(s) for s in input_ids])
        input_ids = [s + [self.pad_token_id] * (max_length - len(s)) for s in input_ids]
        label_ids = [s + [-100] * (max_length - len(s)) for s in label_ids]

        input_ids = torch.cuda.LongTensor(input_ids).contiguous()
        label_ids = torch.cuda.LongTensor(label_ids).contiguous()
        attention_mask = input_ids != self.pad_token_id

        return input_ids, attention_mask, label_ids
