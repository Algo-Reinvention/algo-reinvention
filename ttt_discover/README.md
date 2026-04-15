# Test-Time Discovery

`ttt_discover/` contains the search-time reinvention code used after unlearning.

## Environment

Use `CONDA_VERL_NAME` for the discovery process itself. Make sure the root [`.env.example`](../.env.example) has been set to `.env`.

## Discovery Entrypoints

The main entrypoint is [`ttt_discover/ttt_discover.py`](ttt_discover.py), but in most cases you should start from the task-specific wrappers under [`ttt_discover/scripts/`](scripts).

Each wrapper:
- selects one YAML under [`ttt_discover/configs/`](configs)
- optionally overrides `model_path`, `tokenizer_path`, `output_dir`, `num_gpus`, or `resume_from`
- launches `python -m ttt_discover.ttt_discover --config ...`

The default YAML configs point to specific unlearning checkpoints under `unlearn/saves/.../global_step_*`, but `MODEL_PATH` can also point to a promoted final model such as `_final_ckpts/<model>/<algorithm>/unlearned`.

## Run a Single Task

Make sure the starting checkpoint exists locally. The config can point either to a full unlearning run directory or directly to a HuggingFace-style model directory.

Run example:

```bash
MODEL_PATH=/path/to/model \
NUM_GPUS=4 \
bash ttt_discover/scripts/run_strassen.sh
```

If you want to change the search horizon, sampling budget, or reward settings, edit the selected YAML under [`ttt_discover/configs/`](configs). The most important fields are:
- `levels`
- `num_samples_per_step`
- `max_steps_per_problem`
- `temperature`
- `reward_mode`
- `execution_timeout`

## Run a Small Batch

The batch wrapper [`run_batch_3.sh`](scripts/run_batch_3.sh) runs three tasks sequentially:
- `Prim (level 1)`
- `Manacher (level 1)`
- `KMP (level 0)`

Run example:

```bash
MODEL_PATH=/path/to/model \
NUM_GPUS=4 \
bash ttt_discover/scripts/run_batch_3.sh
```

## Run From a Config Directly

Use this when you want to modify the YAML first and then launch the trainer directly.

Run example:

```bash
python -m ttt_discover.ttt_discover \
  --config ttt_discover/configs/strassen.yaml \
  --model_path /path/to/model \
  --output_dir ttt_discover/outputs/strassen_custom \
  --num_gpus 4
```

## Outputs

Each run writes artifacts under `output_dir`, typically including:
- `logs/run.log`
- `logs/metrics.jsonl`
- `responses/<problem_id>/...`
- `checkpoints/...`
- `final_results.json`

If `inference_mode: vllm_server` is used and the trainer starts a local server itself, it also writes `vllm_server.log` under the same output directory.
