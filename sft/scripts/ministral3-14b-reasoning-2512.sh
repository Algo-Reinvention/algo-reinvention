#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-ministral3-14b-reasoning-2512}"
export ALGORITHM="${ALGORITHM:-graph-sp-dijkstra}"
export RECIPES="${RECIPES:-cold_start}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
