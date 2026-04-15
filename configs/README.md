# Config-Driven Experiments

This directory stores the YAML config layers used by the shell entrypoints.

Instead of maintaining hundreds of model-algorithm-method shell scripts, a run is now defined by:

1. `yamls/models.yaml`
2. `yamls/algorithms.yaml`
3. `yamls/recipes.yaml`

Optional shortcut manifests live under `yamls/manifests/`.

The final config is composed at runtime as:

`model + algorithm + recipes[] + overrides`

Examples:

```bash
bash sft/scripts/qwen3-4b-thinking-2507.sh \
  ALGORITHM=graph-sp-dijkstra
```

```bash
bash unlearn/scripts/qwen3-4b-thinking-2507.sh \
  ALGORITHM=graph-sp-dijkstra
```

```bash
bash sft/scripts/ministral3-14b-reasoning-2512.sh \
  ALGORITHM=string-kmp \
  num_epochs=4
```

Notes:

- A `manifest` is a small YAML file that pre-binds a specific `kind + model + algorithm + recipes + overrides` bundle, so you can launch a common case without repeating those fields on the command line.
- `recipes` are composable. For example, `grpo` defines the method, while `retain_code_math_indist` defines the retain split profile.
- `sft/` is now the canonical namespace for SFT outputs.
- Uppercase `KEY=value` arguments configure the shell wrapper itself, while lowercase `key=value` arguments become config overrides.
- `configs/_resolve_config.py` is now an internal backend for the shell wrappers, not the primary user-facing entrypoint.
