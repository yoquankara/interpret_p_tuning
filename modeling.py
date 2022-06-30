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
#
#
# Copyright (c) 2021 THUDM
# Licensed under the MIT license. See the LICENSE for details.

import argparse
import random
import torch

from typing import Tuple, List

from sumeval.metrics.rouge import RougeCalculator
from transformers import AutoTokenizer, AutoModelForCausalLM
from nl_inputs import DEFAULT_TEMPLATE


class PromptEncoder(torch.nn.Module):
    def __init__(self, template: Tuple[int, int], hidden_size: int, dropout: float,
                 tokenizer: AutoTokenizer, device: str) -> None:
        """

        Returns:
            object: encoder of injected prompt tokens
        """
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tokenizer = tokenizer
        # ent embedding
        self.cloze_mask = [
            [1] * self.spell_length
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.hidden_size, self.hidden_size))

        print("init prompt encoder...")

    def forward(self) -> torch.Tensor:
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class NewPromptEncoder(torch.nn.Module):
    def __init__(self, template: Tuple[int, int], hidden_size: int, dropout: float,
                 tokenizer: AutoTokenizer, device: str, num_mlp: int = 1, residual: bool = True) -> None:
        """

        Returns:
            object: encoder of injected prompt tokens
        """
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tokenizer = tokenizer
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.dropout,
                                       bidirectional=True,
                                       batch_first=True)
        if num_mlp == 2:
            self.mlp_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.hidden_size, self.hidden_size))
        elif num_mlp == 1:
            self.mlp_head = torch.nn.Linear(self.hidden_size, self.hidden_size)
        else:
            raise ValueError(f"Not support for num_mlp = {num_mlp}")
        self.residual = residual

        print("init prompt encoder...")

    def forward(self, inputs) -> torch.Tensor:
        output_embeds = self.mlp_head(self.lstm_head(inputs.to(self.device).unsqueeze(0))[0]).squeeze()
        if self.residual:
            output_embeds += inputs
        return output_embeds


class PTuneLM(torch.nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, device: str, template: Tuple[int, int],
                 args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.template = template
        self.rouge = RougeCalculator(stopwords=True, lang="ja")

        # load pre-trained model
        if 'gpt' in args.model_name or 'megatron' in args.model_name:
            if self.args.checkpoint_dir:
                # Load from local
                self.model = AutoModelForCausalLM.from_pretrained(self.args.checkpoint_dir)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name)
        else:
            raise NotImplementedError(f"Not support loading for model type {self.args.model_name}")

        if not args.use_lm_finetune:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        # TODO: use model.transformer.get_input_embeddings? or model.get_input_embeddings?
        self.embeddings = self.model.base_model.get_input_embeddings()

        self.hidden_size = self.embeddings.embedding_dim
        self.dropout = self.args.lstm_dropout
        self.prompt_token_id = self.tokenizer.get_vocab()[self.args.prompt_token]

        self.spell_length = sum(self.template)
        self.nl_inputs = None
        if not self.args.new_design:
            self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.dropout, self.tokenizer, self.device)
        else:
            self.prompt_encoder = NewPromptEncoder(self.template, self.hidden_size, self.dropout, self.tokenizer,
                                                   self.device, self.args.new_num_mlp, self.args.new_residual)
            if self.args.new_random_init:
                self.nl_inputs = torch.LongTensor(random.sample(list(range(self.tokenizer.vocab_size)), self.spell_length)).to(self.device)
            else:
                self.nl_inputs = torch.LongTensor(self.tokenizer.encode(DEFAULT_TEMPLATE[template][self.args.dataset])).to(self.device)
            assert self.spell_length == len(self.nl_inputs)
        print("init prompt encoder done")
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def _debug_topk_emb(self, prompt_emb: torch.LongTensor, k: int = 3) -> None:
        print("Topk l2-norm with decode()") 
        for i in range(sum(self.template)):
            print(i, [self.tokenizer.decode(s) for s in torch.linalg.norm(self.embeddings.weight - prompt_emb[i, :], dim=1).topk(k, largest=False).indices])
        print("Topk l2-norm with convert_ids_to_tokens()") 
        for i in range(sum(self.template)):
            print(i, self.tokenizer.convert_ids_to_tokens(torch.linalg.norm(self.embeddings.weight - prompt_emb[i, :], dim=1).topk(k, largest=False).indices))

        print("Topk cosine with decode()") 
        for i in range(sum(self.template)):
            print(i, [self.tokenizer.decode(s) for s in torch.nn.functional.cosine_similarity(self.embeddings.weight, prompt_emb[i, :], dim=0).topk(k, largest=False).indices])
        print("Topk cosine with convert_ids_to_tokens()") 
        for i in range(sum(self.template)):
            print(i, self.tokenizer.convert_ids_to_tokens(torch.nn.functional.cosine_similarity(self.embeddings.weight, prompt_emb[i, :], dim=0).topk(k, largest=False).indices))
 
    def embed_input(self, input_ids: torch.LongTensor) -> torch.nn.Module:
        bz = input_ids.shape[0]
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.prompt_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(input_ids_for_embedding)

        blocked_indices = (input_ids == self.prompt_token_id).nonzero().reshape(
            (bz, self.spell_length, 2))[:, :, 1]  # bz
        if self.nl_inputs is None:
            replace_embeds = self.prompt_encoder()
        else:
            prompt_emb = self.embeddings(self.nl_inputs)
            replace_embeds = self.prompt_encoder(prompt_emb.float())

        # Print top-k nearest tokens once
        if self.args.only_evaluate and self.args.print_topk:
            self._debug_topk_emb(replace_embeds, k=3)
            self.args.print_topk = False

        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor,
                label_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, List[List[float]]]:
        # get embedded input
        inputs_embeds = self.embed_input(input_ids)

        def gpt_out(need_rouge: bool = False):
            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half())
            logits = output.logits
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(2)), label_ids.view(-1))

            label_mask = (label_ids.view(-1) != -100)
            pred_ids = logits.argmax(-1)
            acc = (pred_ids.view(-1)[label_mask] == label_ids.view(-1)[label_mask]).float().mean(0)

            bz = label_ids.size()[0]
            label_mask = label_mask.view(bz, -1)
            exact_cnt = []
            if need_rouge:
                rouge1, rouge2, rougel = [], [], []
            for i in range(bz):
                mask_i = label_mask[i]
                if need_rouge:
                    predicted = self.tokenizer.decode(pred_ids[i][mask_i])  # .replace(" ", "")
                    gold_label = self.tokenizer.decode(label_ids[i][mask_i])  # .replace(" ", "")
                    rouge1.append(self.rouge.rouge_n(summary=predicted, references=gold_label, n=1))
                    rouge2.append(self.rouge.rouge_n(summary=predicted, references=gold_label, n=2))
                    rougel.append(self.rouge.rouge_l(summary=predicted, references=gold_label))
                else:
                    exact_cnt.append(int(torch.equal(pred_ids[i][mask_i], label_ids[i][mask_i])))

            if need_rouge:
                # Return batch-average loss, acc and raw rouges (not averaged)
                return loss, acc, [rouge1, rouge2, rougel]
            else:
                # Return batch-average loss, acc and raw exact counts (not averaged)
                return loss, acc, [exact_cnt]

        if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            # Megatron model should be converted into GPT format already
            return gpt_out(need_rouge=self.args.dataset == "XLSUM")
        else:
            raise NotImplementedError(f"Not support output of model type {self.args.model_name}")
