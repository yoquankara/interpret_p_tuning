# Interpretability in Continuous Prompt Tuning for Japanese Generative Pre-trained Transformers

Source code for the paper is based on the original implementation at https://github.com/THUDM/P-tuning, with following improvements:
* Original P-tuning code only supports for single-token output, therefore support for multi-token output is added.
* 2 additional metrics: average token accuracy, and exact match of all tokens. For `XLSUM` task, Rouge-1/2/L are used
instead of exact match.
* New design utilizing the same pre-trained embedding is provided, which improves both interpretability and performance.

We do not publish the same pre-trained model in the paper, however any GPT-compatible model could be used with appropriate
tokenizer and model class from huggingface/transformers. Please note that default human-made prompts inside `nl_inputs.py`
should be changed accordingly to assure the number of prompt tokens to be the same as the template. Other hyperparameters
such as learning rate should be adjusted as well.

## Installation

This was tested with python3.8, torch==1.9.0+cu111 on A100 40GB. V100 32GB should be fine.

Assume python3.8 is installed already.

- Virtual environment

```
python3.8 -m venv .env
. .env/bin/activate
```

- Install requirements
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

### Download dataset data

There are four tasks in Japanese for evaluation: `RCQA`, `JSNLI`, `LIVEDOOR-NEWS`, and `XLSUM`. Dataset for these tasks
can be downloaded from internet. Please refer to the paper for more task details.

- Data preprocess:
  - Sampling from original dataset for train/dev./test of 400/1000/1000, respectively, with the input/output truncated to 512/64 tokens.
  - For tasks with less than 1000 samples in the dev./test datasets, we used all the available samples.
  - For RCQA, we removed non-answerable samples because we have not found an effective way to generate an equivalent meaning


### Tuning

Please change default arguments inside the script according to your experiment.

#### For new design

Specify `--new_design` for the new design that utilizing the same pre-trained embedding.
Use `--new_random_init` for random initialization of prompts, otherwise it will use
`DEFAULT_TEMPLATE` inside `nl_inputs.py`.

```
sh tuning.sh
```

### Interpretation

Specify `--print-topk` in `eval.sh` to output nearest tokens from the vocabulary to the prompt tokens after tuning.

```
sh eval.sh
```

Significance test with paired bootstrap resampling can be run with `eval_utils.py` against output files from `--eval_metrics_to_file`
of `eval.sh`.

### Unit test

```
pytest tests/test_modeling.py
```

# Acknowledgement

We thank authors of https://github.com/THUDM/P-tuning from where we borrowed code for original P-tuning.

We also thank authors of https://github.com/neubig/util-scripts from where we borrowed code for paired bootstrap resampling test.