# Coconut Reproduction

This directory contains a reproduction of Coconut-style latent reasoning for
GSM8K, plus a direct Qwen baseline script.

Run commands from this directory unless you intentionally adjust paths:

```bash
cd coconut_reproduction
```

## Training and Evaluation

Use `main.py` with one YAML config:

```bash
python main.py configs/gsm_cot.yaml
python main.py configs/gsm_coconut.yaml
python main.py configs/gsm_coconut_eval.yaml
python main.py configs/gsm_coconut_single.yaml
```

`main.py` loads the tokenizer and base model from `model_id`, adds the Coconut
tokens `<bot>`, `<eot>`, and `<latent>`, resizes the model embeddings, builds
the configured dataset, trains unless `only_eval` is true, and then runs a final
test pass.

For Coconut training, the stage is calculated in code as:

```python
stage = epoch // epochs_per_stage
```

The dataset controls how many reasoning steps become latent tokens. During
normal Coconut training, the prompt is:

```text
question <bot> latent tokens <eot> remaining text steps ### answer
```

When `stage > max_latent_stage`, the latent count is capped at
`c_thought * max_latent_stage`, all remaining text steps are removed, and the
answer is still included for training loss.

During Coconut validation, the prompt is:

```text
question <bot> latent tokens <eot> remaining text steps except the final step
```

The final text step is removed when it is still textual, because it often leaks
the answer. Validation labels are all `-100`, so validation uses generation
accuracy rather than teacher-forced loss.

During Coconut test, the prompt always uses exactly
`c_thought * max_latent_stage` latent tokens and no textual reasoning steps:

```text
question <bot> latent tokens <eot>
```

For single-sample mode, train, validation, and test use `stage * c_thought`
latent tokens and no textual reasoning steps. Training includes the answer;
validation and test use the same prompt without the answer.

## Config Parameters

`wandb_project`: Weights & Biases project name.

`run_name`: Name used for W&B and checkpoint subdirectories.

`only_eval`: If true, skips training and runs the final test/eval pass.

`debug`: If true, limits the dataset to 32 samples and forces
`epochs_per_stage = 1`.

`single_sample`: If true, trains on repeated copies of one selected sample.

`single_sample_index`: Index selected from the training split before shuffling.

`single_sample_steps_per_epoch`: Number of optimizer steps to run per epoch in
single-sample mode.

`bf16`: Indicates bf16 training. The current code loads and runs the model in
`torch.bfloat16`.

`cot`: Enables CoT finetuning mode. Use this with `coconut: False`.

`coconut`: Enables Coconut latent-token training mode. Use this with
`cot: False`.

`c_thought`: Number of `<latent>` tokens used for each converted reasoning
step/stage.

`epochs_per_stage`: Number of epochs before the Coconut stage increments.

`max_latent_stage`: Maximum latent stage used by the dataset. Later epochs keep
the latent count capped at `c_thought * max_latent_stage`.

`pad_latent_to_max`: Compatibility option for older prompt construction. The
current explicit train/validation/test dataset modes use the stage and
`max_latent_stage` rules above.

`save_only_improve`: If true, saves checkpoints only when validation accuracy
improves. If false, saves after every epoch.

`num_epochs`: Total number of training epochs.

`lr`: Optimizer learning rate.

`weight_decay`: Optimizer weight decay.

`batch_size_training`: Per-step training dataloader batch size.

`gradient_accumulation_steps`: Number of batches accumulated before an optimizer
step.

`model_id`: Hugging Face model id or local model path used by
`AutoTokenizer` and `AutoModelForCausalLM`.

`save_path`: Root directory for checkpoints and logs.

`load_model_path`: Checkpoint path to load before training/eval. Use the string
`None` for no checkpoint in the current configs.

`resume_from_epoch`: First epoch index to run when resuming.

`dataset_name`: Hugging Face dataset name. The GSM configs use `openai/gsm8k`.

`dataset_config_name`: Hugging Face dataset config. GSM8K uses `main`.

`train_split`: Dataset split used for training.

`val_split`: Dataset split used for per-epoch validation.

`test_split`: Optional dataset split used for the final test pass. If omitted,
`test` is used.

`max_new_tokens`: Maximum tokens generated during validation/test.

`step_extraction_method`: How GSM8K reasoning steps are extracted from the
`answer` field. `equations` uses text inside `<<...>>`. `cot` removes those
equation spans and builds text sentence steps.

`add_special_tokens`: If true, inserts `<bot>` and `<eot>` around latent tokens.
If false, only latent tokens are inserted.

`reset_optimizer`: If true, reinitializes the optimizer at stage boundaries.

`seed`: Random seed for Python, NumPy, and PyTorch.

`uniform_prob`: Present in the configs for compatibility. It is not currently
used by the reproduction training loop.

## Qwen Baseline

`infer_qwen.py` evaluates an unmodified Qwen model as a direct-answer GSM8K
baseline. By default it uses:

```text
model:   Qwen/Qwen3-0.6B
dataset: data/gsm_test.json
```

Run the default baseline:

```bash
python infer_qwen.py
```

Run a smaller smoke test:

```bash
python infer_qwen.py --limit 20 --batch-size 4
```

Use one-shot prompting from the same GSM test JSON:

```bash
python infer_qwen.py --num-shots 1 --few-shot-index 0
```

Common options:

`--model-id`: Hugging Face model id or local model path. Defaults to
`Qwen/Qwen3-0.6B`.

`--datasets`: Dataset specs as `name=path`. Defaults to
`gsm=data/gsm_test.json`. The script is intended to use `gsm_test.json` as the
default dataset.

`--output-dir`: Directory for run summaries and per-example generations.

`--batch-size`: Inference batch size.

`--max-new-tokens`: Maximum generated tokens per prompt.

`--num-shots`: `0` for zero-shot or `1` for one-shot prompting.

`--few-shot-path`: JSON file used to construct the one-shot example. Defaults to
`data/gsm_test.json`.

`--few-shot-index`: Row index used as the one-shot example.

`--limit`: Optional maximum number of examples to evaluate.

`--seed`: Random seed.

`--dtype`: Model dtype: `auto`, `bfloat16`, `float16`, or `float32`.

`--device`: Device string such as `auto`, `cuda`, `cuda:0`, or `cpu`.

`--temperature`: `0.0` uses greedy decoding. Values above zero enable sampling.

`--top-p`: Nucleus sampling parameter used only when sampling is enabled.

`--attn-implementation`: Optional Transformers attention implementation, such as
`sdpa`.

`--local-files-only`: Load only files already present in the local Hugging Face
cache.

`--no-save-generations`: Skip writing the per-example JSONL file.

Outputs are written under `results/qwen/<timestamp>/`. The run directory
contains `summary.json` and, unless disabled, `gsm.jsonl`.
