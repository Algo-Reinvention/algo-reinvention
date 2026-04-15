# Unlearning Pipeline

## Environment

For most training runs, the `CONDA_VERL_NAME` environment is sufficient. Use `CONDA_VERL_MINISTRAL_NAME` when the target model is `Ministral-3-14B-Reasoning-2512`.

Make sure the root `.env` is configured before running the scripts here. Start from [`.env.example`](../.env.example). For actual unlearning runs, you need all of the following:
- `PROJECT_ROOT`
- the base-model path variable for your model, such as `QWEN3_4B_THINKING_2507_PATH`
- `API_KEY`
- `BASE_URL`
- `API_MODEL_NAME`

The last three are required because the GRPO reward is computed by [`unlearn/verl/utils/reward_score/unlearn_reward_batch_api.py`](verl/utils/reward_score/unlearn_reward_batch_api.py), which calls an external judge model through the OpenAI-compatible API.

### 1. Prepare Cold-Start Data

The main helper is [`preprocess/scripts/cold_start_gen.sh`](../preprocess/scripts/cold_start_gen.sh).

Run example:

```bash
export CUDA_VISIBLE_DEVICES=0

bash preprocess/scripts/cold_start_gen.sh \
  MODEL_PATH=/path/to/model \
  MODEL_NAME=qwen3-4b-thinking-2507 \
  CATEGORY=graph-sp-dijkstra \
  PREPARE_IDK=True
```

This writes the cold-start parquet files under `_data/post_train/idk/<algorithm>/<model>/...`.

### 2. Train the Cold-Start Model

Run example:

```bash
export CUDA_VISIBLE_DEVICES=0,1

bash sft/scripts/run.sh \
  MODEL=qwen3-4b-thinking-2507 \
  ALGORITHM=graph-sp-dijkstra \
  RECIPES=cold_start
```

The default cold-start recipe is `cold_start`, which trains on `idk-4096.parquet` and `indist-4096.parquet`. Outputs are written under:

```text
sft/saves/<model>/<algorithm>/cold_start/idk_indist_start/...
```

### 3. Select and Promote a Cold-Start Checkpoint

By default, `unlearn/scripts/run.sh` resolves its initialization model to:

```text
_final_ckpts/<model>/<algorithm>/cold_start
```

So after the cold-start SFT run, select the checkpoint you want to use and copy or symlink its `huggingface/` directory there.

Example:

```bash
mkdir -p _final_ckpts/qwen3-4b-thinking-2507/graph-sp-dijkstra

ln -sfn \
  "$PROJECT_ROOT/sft/saves/qwen3-4b-thinking-2507/graph-sp-dijkstra/cold_start/idk_indist_start/checkpoints/global_step_140/huggingface" \
  "$PROJECT_ROOT/_final_ckpts/qwen3-4b-thinking-2507/graph-sp-dijkstra/cold_start"
```

Replace `global_step_140` with the checkpoint you actually choose.

### 4. Prepare Unlearning Data

Run example:

```bash
bash preprocess/scripts/prepare_unlearn.sh \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  CATEGORY=graph-sp-dijkstra
```

This stage builds:
- `forget.parquet` from `datasets/unlearn/<algorithm>/{algo2context,context2algo}.jsonl`
- `test.parquet` from `datasets/unlearn/<algorithm>/test.jsonl`
- `retain-code-4096.parquet` and `retain-math-4096.parquet` from `_data/post_train/general/<model>/...`
- `retain-indist-4096.parquet` from the cold-start `indist` data

`prepare_unlearn.sh` assumes `_data/post_train/general/<model>/code-4096.parquet` and `math-4096.parquet` already exist. If they do not, generate raw retain responses first with [`simple_parallel/scripts/inferences/retain_generate_raw.sh`](../simple_parallel/scripts/inferences/retain_generate_raw.sh), then convert them with [`preprocess/general-json2parquet.py`](../preprocess/general-json2parquet.py).

### 5. Run Unlearning

The main entrypoint is [`unlearn/scripts/run.sh`](scripts/run.sh).
Common arguments:
- `ACTION`: what to do. Use `run`, `resolve`, or `print-command`.
- `MODEL`: the base model family name, such as `qwen3-4b-thinking-2507`.
- `ALGORITHM`: the target algorithm category, such as `graph-sp-dijkstra`.
- `RECIPES`: the recipe stack to apply, such as `grpo,retain_code_math_indist`.
- `MANIFEST`: an optional shortcut YAML under [`../configs/yamls/manifests/`](../configs/yamls/manifests).
- `NAME`: an optional custom experiment name.
- Uppercase `KEY=value` arguments configure the shell wrapper itself.
- Lowercase `key=value` arguments become YAML config overrides.

The default setup here is `RECIPES=grpo,retain_code_math_indist`. That corresponds to:
- `grpo`: the on-policy unlearning method
- `retain_code_math_indist`: the retain profile used to build `Dretain`

Run example:

```bash
export CUDA_VISIBLE_DEVICES=0,1

bash unlearn/scripts/run.sh \
  MODEL=qwen3-4b-thinking-2507 \
  ALGORITHM=graph-sp-dijkstra \
  RECIPES=grpo,retain_code_math_indist
```

By default this loads the policy model from:

```text
_final_ckpts/<model>/<algorithm>/cold_start
```

and writes checkpoints under:

```text
unlearn/saves/<model>/<algorithm>/<method>/<name>/global_step_*/
```

Logs are written under:

```text
unlearn/logs/<model>/<algorithm>/<method>/<name>/
```

If you want a shorter command, use the model-specific wrapper:

```bash
export CUDA_VISIBLE_DEVICES=0,1

bash unlearn/scripts/qwen3-4b-thinking-2507.sh \
  ALGORITHM=graph-sp-dijkstra
```

### 6. Evaluate Checkpoints

Refer [`../simple_parallel/README.md`](../simple_parallel/README.md) for the evaluation details.

## Inspect a Config Without Launching

Use `ACTION=resolve` when you want to inspect the merged YAML config first. Use `ACTION=print-command` when you only want the final launch command.

Run example:

```bash
bash unlearn/scripts/run.sh \
  ACTION=resolve \
  MANIFEST=unlearn-qwen3-dijkstra-grpo.yaml
```
