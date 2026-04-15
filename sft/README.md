# SFT Namespace

`sft/` is the canonical namespace for config-driven SFT experiments.

Primary entrypoints live under:

`sft/scripts/`

Examples:

`bash sft/scripts/run.sh MODEL=qwen3-4b-thinking-2507 ALGORITHM=graph-sp-dijkstra`

`bash sft/scripts/qwen3-4b-thinking-2507.sh`

`bash sft/scripts/ministral3-14b-reasoning-2512.sh ALGORITHM=string-kmp`

Lowercase `key=value` arguments are forwarded as config overrides.

Example:

`bash sft/scripts/qwen3-4b-thinking-2507.sh num_epochs=12`

and should write outputs under:

`sft/saves/...`
