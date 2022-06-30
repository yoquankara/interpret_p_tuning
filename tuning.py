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

import os
from typing import Tuple

import argparse
import copy
import random
import torch
import numpy as np

from datetime import datetime
from os.path import join, abspath, dirname
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer

from dataset import load_json_file, load_tsv_file, RCQADataset, JSNLIDataset, LivedoorNewsDataset, XLSumJaDataset, \
    GPTPTuneCollator
from modeling import PTuneLM


def set_seed(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name", type=str, default='rinna/japanese-gpt-1b')
    parser.add_argument("--prompt_token", type=str, default='[PROMPT]')
    parser.add_argument("--load", type=str, default='', help="Path to Load existing checkpoint for prompt encoder")

    parser.add_argument("--template", type=str, default="(3,3)")
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # tuning configuration
    parser.add_argument("--use_lm_finetune", action='store_true', default=False)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # evaluation
    parser.add_argument("--only_evaluate", action='store_true', default=False)
    parser.add_argument("--print_topk", action='store_true', default=False)
    parser.add_argument("--eval_metrics_to_file", type=str, default='', help="Path to save eval metrics on test set")

    # target task
    parser.add_argument("--dataset", type=str, default="RCQA", choices=["RCQA", "JSNLI", "LIVEDOOR-NEWS", "XLSUM"])
    parser.add_argument("--train_batch_size", type=int, default=4, help="Train batch size, should be 3 for "
                                                                        "LIVEDOOR-NEWS, 2 for XLSUM")
    parser.add_argument("--n_train_max", type=int, default=-1, help="Max size of training data, -1 means all")
    parser.add_argument("--n_dev_max", type=int, default=-1, help="Max size of dev data, -1 means all")
    parser.add_argument("--n_test_max", type=int, default=-1, help="Max size of test data, -1 means all")
    parser.add_argument("--max_query_length", type=int, default=512, help="The maximum length of query text")
    parser.add_argument("--max_output_length", type=int, default=64, help="The maximum length of output text")

    # natural language inputs for new design
    parser.add_argument("--new_design", action='store_true', default=False, help="Enable new design of PromptEncoder")
    parser.add_argument("--new_num_mlp", type=int, default=1, help="Number of MLP layers in new PromptEncoder")
    parser.add_argument("--new_residual", action='store_true', default=False, help="Use residual in new PromptEncoder")
    parser.add_argument("--new_random_init", action='store_true', default=False, help="New design of PromptEncoder with"
                                                                                      "random initialization")

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), './data'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './out'))
    # Local checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default='', help="Path for local model checkpoint")

    args, unknown = parser.parse_known_args()

    # post-parsing args
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # TODO: support model parallel
        self.device = args.device

        # load tokenizer
        if "rinna" in self.args.model_name:
            tokenizer_cls = T5Tokenizer
        else:
            tokenizer_cls = AutoTokenizer
        if self.args.checkpoint_dir:
            self.tokenizer = tokenizer_cls.from_pretrained(self.args.checkpoint_dir, use_fast=False)
        else:
            self.tokenizer = tokenizer_cls.from_pretrained(self.args.model_name, use_fast=False)

        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.prompt_token]})

        # Currently support for RCQA data only
        if self.args.dataset == "RCQA":
            self.train_data = load_json_file(join(self.args.data_dir, self.args.dataset, "train-v1.0.json"),
                                             self.args.n_train_max)
            self.dev_data = load_json_file(join(self.args.data_dir, self.args.dataset, "dev-v1.0.json"),
                                           self.args.n_dev_max)
            self.test_data = load_json_file(join(self.args.data_dir, self.args.dataset, "test-v1.0.json"),
                                            self.args.n_test_max)
            self.train_set = RCQADataset('train', self.train_data)
            self.dev_set = RCQADataset('dev', self.dev_data)
            self.test_set = RCQADataset('test', self.test_data)
        elif self.args.dataset == "JSNLI":
            self.train_data = load_tsv_file(join(self.args.data_dir, self.args.dataset, "text", "train.tsv"),
                                            self.args.n_train_max)
            self.dev_data = load_tsv_file(join(self.args.data_dir, self.args.dataset, "text", "dev.tsv"),
                                          self.args.n_dev_max)
            self.test_data = load_tsv_file(join(self.args.data_dir, self.args.dataset, "text", "test.tsv"),
                                           self.args.n_test_max)

            self.train_set = JSNLIDataset('train', self.train_data)
            self.dev_set = JSNLIDataset('dev', self.dev_data)
            self.test_set = JSNLIDataset('test', self.test_data)
        elif self.args.dataset == "LIVEDOOR-NEWS":
            self.train_data = load_json_file(join(self.args.data_dir, self.args.dataset, "text", "train.jsonl"),
                                             self.args.n_train_max)
            self.dev_data = load_json_file(join(self.args.data_dir, self.args.dataset, "text", "dev.jsonl"),
                                           self.args.n_dev_max)
            self.test_data = load_json_file(join(self.args.data_dir, self.args.dataset, "text", "test.jsonl"),
                                            self.args.n_test_max)
            self.train_set = LivedoorNewsDataset('train', self.train_data)
            self.dev_set = LivedoorNewsDataset('dev', self.dev_data)
            self.test_set = LivedoorNewsDataset('test', self.test_data)
        elif self.args.dataset == "XLSUM":
            self.train_data = load_json_file(join(self.args.data_dir, self.args.dataset, "japanese_train.jsonl"),
                                             self.args.n_train_max)
            self.dev_data = load_json_file(join(self.args.data_dir, self.args.dataset, "japanese_val.jsonl"),
                                           self.args.n_dev_max)
            self.test_data = load_json_file(join(self.args.data_dir, self.args.dataset, "japanese_test.jsonl"),
                                            self.args.n_test_max)
            self.train_set = XLSumJaDataset('train', self.train_data)
            self.dev_set = XLSumJaDataset('dev', self.dev_data)
            self.test_set = XLSumJaDataset('test', self.test_data)
        else:
            raise NotImplementedError(f"Not support for dataset {self.args.dataset}")
        os.makedirs(self.get_save_path(), exist_ok=True)

        collator = GPTPTuneCollator(self.tokenizer, prompt=self.args.prompt_token, template=self.args.template,
                                    max_query_length=self.args.max_query_length,
                                    max_output_length=self.args.max_output_length)
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.train_batch_size, shuffle=True,
                                       collate_fn=collator, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8, collate_fn=collator)
        self.test_loader = DataLoader(self.test_set, batch_size=8, collate_fn=collator)

        self.model = PTuneLM(self.tokenizer, self.device, self.args.template, args)
        if self.args.load:
            self.model.prompt_encoder.load_state_dict(torch.load(self.args.load)["embedding"])
            print(f"load {self.args.load} done")

        self.eval_writer = None
        if self.args.eval_metrics_to_file:
            self.eval_writer = open(self.args.eval_metrics_to_file, "w")

    def evaluate(self, epoch_idx: int, evaluate_type: str) -> Tuple[float, ...]:
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        with torch.no_grad():
            self.model.eval()
            accs, losses, other_metrics = [], [], []
            for input_ids, attention_mask, label_ids in loader:
                loss, acc, other_metric = self.model(input_ids, attention_mask, label_ids)
                accs.append(acc.detach().item())
                losses.append(loss.detach().item())
                other_metrics.append(other_metric)
                if evaluate_type == 'Test' and self.eval_writer:
                    bz = len(input_ids)
                    for i in range(bz):
                        self.eval_writer.write("\t".join([str(m[i]) for m in other_metric]) + "\n")

        accs = sum(accs) / len(accs)
        losses = sum(losses) / len(losses)
        if self.args.dataset != "XLSUM":
            exact_cnts = sum([sum(cnt[0])/len(cnt[0]) for cnt in other_metrics]) / len(other_metrics)
            print(f"{evaluate_type} Epoch {epoch_idx} Loss: {losses} Acc: {accs} Exact: {exact_cnts}")
            return losses, accs, exact_cnts
        else:
            rouge1, rouge2, rougel = map(lambda idx: sum([sum(x[idx])/len(x[idx]) for x in other_metrics]) / len(other_metrics), [0, 1, 2])
            print(
                f"{evaluate_type} Epoch {epoch_idx} Loss: {losses} Acc: {accs} Rouge1: {rouge1} Rouge2: {rouge2} Rougel: {rougel}")
            return losses, accs, rouge1, rouge2, rougel

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name, 'only_evaluate'])
        names = [self.args.model_name,
                 "template_{}".format(self.args.template),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed),
                 f"train_{self.args.n_train_max}",
                 f"dev_{self.args.n_dev_max}",
                 f"test_{self.args.n_test_max}"]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, self.args.dataset, 'prompt_model', self.args.model_name, self.get_task_name())

    def get_checkpoint(self, lr, epoch_idx, dev_acc, test_acc, test_exact):
        ckpt_name = "lr_{:.0e}_epoch_{}_dev_{}_test_{}_exact_{}.ckpt".format(lr, epoch_idx, round(dev_acc * 100, 4),
                                                                round(test_acc * 100, 4), round(test_exact * 100, 4))
        if self.args.new_design:
            ckpt_name = f"new_mlp-{self.args.new_num_mlp}_residual_{self.args.new_residual}_" + ckpt_name
        return {'embedding': copy.deepcopy(self.model.prompt_encoder.state_dict()),
                'dev_acc': dev_acc,
                'test_acc': test_acc,
                'test_exact': test_exact,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def get_checkpoint_with_rouge(self, lr, epoch_idx, dev_acc, test_acc, test_rouge1, test_rouge2, test_rougel):
        ckpt_name = "lr_{:.0e}_epoch_{}_dev_{}_test_{}_rouge1_{}_rouge2_{}_rougel_{}.ckpt".format(lr, epoch_idx,
                                        round(dev_acc * 100, 4), round(test_acc * 100, 4), round(test_rouge1 * 100, 4),
                                        round(test_rouge2 * 100, 4), round(test_rougel * 100, 4))
        if self.args.new_design:
            ckpt_name = f"new_mlp-{self.args.num_mlp}_residual_{self.args.residual}_" + ckpt_name
        return {'embedding': copy.deepcopy(self.model.prompt_encoder.state_dict()),
                'dev_acc': dev_acc,
                'test_acc': test_acc,
                'test_rouge1': test_rouge1,
                'test_rouge2': test_rouge2,
                'test_rougel': test_rougel,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print(f"Checkpoint {ckpt_name} saved.")

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(self.args.epochs):
            # check early stopping
            if epoch_idx > -1:
                dev_loss, dev_acc = self.evaluate(epoch_idx, 'Dev')[:2]
                if dev_acc >= best_dev or self.args.only_evaluate:
                    if self.args.dataset != "XLSUM":
                        test_loss, test_acc, test_exact = self.evaluate(epoch_idx, 'Test')
                        best_ckpt = self.get_checkpoint(self.args.lr, epoch_idx, dev_acc, test_acc, test_exact)
                    else:
                        test_loss, test_acc, test_rouge1, test_rouge2, test_rougel = self.evaluate(epoch_idx, 'Test')
                        best_ckpt = self.get_checkpoint_with_rouge(self.args.lr, epoch_idx, dev_acc, test_acc,
                                                                   test_rouge1, test_rouge2, test_rougel)
                    early_stop = 0
                    best_dev = dev_acc
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        self.save(best_ckpt)
                        print(f"Early stopping at epoch {epoch_idx}.")
                        return best_ckpt
            if self.args.only_evaluate:
                break

            # run training
            accs, losses, other_metrics = [], [], []
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, acc, other_metric = self.model(*batch)

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                accs.append(acc.detach().item())
                losses.append(loss.detach().item())
                other_metrics.append(other_metric)
            losses = sum(losses) / len(losses)
            accs = sum(accs) / len(accs)
            if self.args.dataset != "XLSUM":
                exact_cnts = sum([sum(cnt[0])/len(cnt[0]) for cnt in other_metrics]) / len(other_metrics)
                print(f"Train Epoch {epoch_idx} Loss: {losses} Acc: {accs} Exact: {exact_cnts}")
            else:
                rouge1, rouge2, rougel = map(lambda idx: sum([sum(x[idx])/len(x[idx]) for x in other_metrics]) / len(other_metrics), [0, 1, 2])
                print(
                    f"Train Epoch {epoch_idx} Loss: {losses} Acc: {accs} Rouge1: {rouge1} Rouge2: {rouge2} Rougel: {rougel}")

            my_lr_scheduler.step()
        if not self.args.only_evaluate:
            self.save(best_ckpt)
        if self.eval_writer:
            self.eval_writer.close()
        return best_ckpt


def main():
    args = construct_generation_args()
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
