# Unlearn Scripts

`unlearn/scripts/` is the shell-first entrypoint for config-driven unlearning runs.

Examples:

`bash unlearn/scripts/run.sh MODEL=qwen3-4b-thinking-2507 ALGORITHM=graph-sp-dijkstra`

`bash unlearn/scripts/qwen3-4b-thinking-2507.sh`

`bash unlearn/scripts/ministral3-14b-reasoning-2512.sh ALGORITHM=string-kmp`

Lowercase `key=value` arguments are forwarded as config overrides.

Example:

`bash unlearn/scripts/qwen3-4b-thinking-2507.sh learning_rate=2e-5 total_epochs=20`
