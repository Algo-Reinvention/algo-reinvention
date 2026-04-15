# Simple Parallel

`simple_parallel/` contains the lightweight serving and evaluation utilities.

## Environment

For most inference-only usage, the `CONDA_VLLM` environment is sufficient. Additional environments are only needed for specific cases: set `CONDA_BFCL` for BFCL evaluation, and `CONDA_VERL_MINISTRAL` for Ministral models.

Make sure the root `.env` is configured before running the scripts here. Start from [`.env.example`](../.env.example).

## Evaluation Entrypoint
The main entrypoint is [`simple_parallel/scripts/_run.sh`](scripts/_run.sh).
Common arguments:
- `TYPE`: which kind of checkpoint to evaluate. Use `BASE`, `UNLEARN`, `SFT`, or `FINAL`. In most cases, BASE is okay — it takes `MODEL_PATH` directly. The other types are mainly used during development to batch-evaluate checkpoints under specific paths.
- `BASE_MODEL`: the base model family name, such as `qwen3-4b-thinking-2507`.
- `BENCHMARKS`: which evaluation to run. Available values here are `final`, `forget`, `lcb`, `aime25`, and `bfcl`.
- `MODEL_PATH`: explicit checkpoint path. This is mainly used with `TYPE=BASE`.
- `TEST_CATEGORY`: the target algorithm category for `final` and `forget`, such as `graph-sp-dijkstra`.
- `LEVEL`: the hint level, `0`, `1`, or `2`.

## Run Reinvention

Firesr prepare the final-test inputs and calibrate the machine-specific time threshold:

```bash
bash preprocess/scripts/initialize.sh
```


Test example:

```bash
export CUDA_VISIBLE_DEVICES=0

bash simple_parallel/scripts/_run.sh \
  TYPE=BASE \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  BENCHMARKS=final \
  TEST_CATEGORY=graph-sp-dijkstra \
  LEVEL=0
```

## Test Forgetting Rate

Make sure the root `.env` contains valid `API_KEY` and `BASE_URL` for the judge model.

Test example:

```bash
export CUDA_VISIBLE_DEVICES=0

bash simple_parallel/scripts/_run.sh \
  TYPE=BASE \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  BENCHMARKS=forget \
  TEST_CATEGORY=graph-sp-dijkstra \
  FORGET_TEST_MODE=text
```

## Run Benchmarks

### LiveCodeBench

Clone [our fork of LiveCodeBench](https://github.com/Algo-Reinvention/LiveCodeBench) and place it alongside this repository, so that the directory structure looks like:
```
../
├── algo-reinvention/
└── LiveCodeBench/
```

Test example:

```bash
export CUDA_VISIBLE_DEVICES=0

bash simple_parallel/scripts/_run.sh \
  TYPE=BASE \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  BENCHMARKS=lcb \
  TEST_MODE=fast
```

### AIME25

First split the dataset into per-problem JSON files before evaluation:

```bash
python preprocess/split_jsonl.py \
  --input_path datasets/aime25.jsonl \
  --output_dir _data/benchmarks/aime25_split \
  --prefix aime25 \
  --question_key problem \
  --solution_key answer \
  --extra_key id
```

Test example:

```bash
export CUDA_VISIBLE_DEVICES=0
bash simple_parallel/scripts/_run.sh \
  TYPE=BASE \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  BENCHMARKS=aime25
```

### BFCL

Clone [our fork of BFCL](https://github.com/Algo-Reinvention/bfcl) and place it alongside this repository, so that the directory structure looks like:
```
../
├── algo-reinvention/
└── bfcl/
```

Test example:

```bash
export CUDA_VISIBLE_DEVICES=0
bash simple_parallel/scripts/_run.sh \
  TYPE=BASE \
  BASE_MODEL=qwen3-4b-thinking-2507 \
  MODEL_PATH=/path/to/model \
  BENCHMARKS=bfcl
```
