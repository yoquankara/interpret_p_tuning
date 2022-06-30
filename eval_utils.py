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
"""Evaluation utilities, borrowing from https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py"""
import random
from typing import List


def extract_best_grid_search() -> str:
    """TODO: Extract best checkpoint from grid search results"""
    return ""


def read_eval_file(eval_fn: str) -> List[float]:
    """Read evaluation metrics from files"""
    eval_results = []
    with open(eval_fn) as ih:
        for line in ih:
            eval_results.append(float(line))
    print(f"Read {eval_fn}, number of metrics: {len(eval_results)}  ")
    return eval_results


def paired_bootstrap_resampling_test(sys1_scores: List[float], sys2_scores: List[float],
                                     num_samples: int = 1000, sample_ratio: float = 0.3,
                                     ) -> None:
    """
    Evaluate with paierd bootstrap resampling test from ACL paper:
    "Statistical Significance Tests for Machine Translation Evaluation"
    Args:
        sys1_scores: score of system 1 on each data point
        sys2_scores: score of system 2 on each data point
        num_samples: number of resampling samples
        sample_ratio: sampling ratio over all data points

    Returns:

    """
    assert len(sys1_scores) == len(sys2_scores)
    n = len(sys1_scores)
    ids = list(range(n))

    print(f"Average score of sys1: {sum(sys1_scores)/n:.3f}, sys2: {sum(sys2_scores)/n:.3f}")

    sys1_samples_scores, sys2_samples_scores = [], []
    wins = [0, 0, 0]

    for _ in range(num_samples):
        resampled_ids = random.choices(ids, k=int(n*sample_ratio))
        resampled_sys1_score = sum([sys1_scores[i] for i in resampled_ids]) / len(resampled_ids)
        resampled_sys2_score = sum([sys2_scores[i] for i in resampled_ids]) / len(resampled_ids)
        if resampled_sys1_score > resampled_sys2_score:
            wins[0] += 1
        elif resampled_sys1_score < resampled_sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1

        sys1_samples_scores.append(resampled_sys1_score)
        sys2_samples_scores.append(resampled_sys2_score)

    wins = [x/num_samples for x in wins]
    print(wins)
    if wins[0] > wins[1]:
        print(f"System 1 wins with p value = {1-wins[0]:.3f}")
    elif wins[1] > wins[0]:
        print(f"System 2 wins with p value = {1-wins[1]:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys1', type=str, help='Eval result file of system 1')
    parser.add_argument('--sys2', type=str, help='Eval result file of system 2')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='Sample ratio')
    args = parser.parse_args()

    paired_bootstrap_resampling_test(read_eval_file(args.sys1), read_eval_file(args.sys2),
                                     num_samples=args.num_samples, sample_ratio=args.sample_ratio)
