#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import expand_env_vars, load_repo_env, require_repo_env_key


CONFIG_DIR = REPO_ROOT / "configs"
YAML_DIR = CONFIG_DIR / "yamls"
MANIFEST_DIR = YAML_DIR / "manifests"
MODELS_FILE = YAML_DIR / "models.yaml"
ALGORITHMS_FILE = YAML_DIR / "algorithms.yaml"
RECIPES_FILE = YAML_DIR / "recipes.yaml"
COMMON_ENV_FILE = CONFIG_DIR / "common_env.sh"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_templates(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(SafeFormatDict(context))
    if isinstance(value, list):
        return [format_templates(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_templates(item, context) for key, item in value.items()}
    return value


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def hydra_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return compact_json(value)
    return str(value)


def render_multiline_command(args: list[str]) -> str:
    if not args:
        return ""
    return " \\\n  ".join(args)


def parse_recipe_names(raw_values: list[str] | None) -> list[str]:
    names: list[str] = []
    for raw in raw_values or []:
        for item in raw.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}', expected key=value")
    key, raw_value = raw.split("=", 1)
    value = yaml.safe_load(raw_value)
    return key.split("."), value


def apply_override(config: dict[str, Any], key_parts: list[str], value: Any) -> None:
    current = config
    for key in key_parts[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[key_parts[-1]] = value


def merge_catalog_entry(config: dict[str, Any], entry: dict[str, Any], kind: str, model: str) -> dict[str, Any]:
    unscoped = {
        key: copy.deepcopy(value)
        for key, value in entry.items()
        if key not in {"common", "sft", "unlearn", "supports"}
    }
    config = deep_merge(config, unscoped)

    common = copy.deepcopy(entry.get("common", {}))
    common_by_model = common.pop("by_model", {})
    config = deep_merge(config, common)
    if isinstance(common_by_model, dict) and model in common_by_model:
        config = deep_merge(config, common_by_model[model])

    scoped = copy.deepcopy(entry.get(kind, {}))
    scoped_by_model = scoped.pop("by_model", {})
    config = deep_merge(config, scoped)
    if isinstance(scoped_by_model, dict) and model in scoped_by_model:
        config = deep_merge(config, scoped_by_model[model])
    return config


def resolve_selection(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[str, str, str, list[str], str | None]:
    kind = args.kind or manifest.get("kind")
    model = args.model or manifest.get("model")
    algorithm = args.algorithm or manifest.get("algorithm")
    recipes = parse_recipe_names(args.recipes) or manifest.get("recipes") or []
    if isinstance(recipes, str):
        recipes = parse_recipe_names([recipes])
    name = args.name or manifest.get("name") or manifest.get("experiment_name")

    missing = [name for name, value in (("kind", kind), ("model", model), ("algorithm", algorithm)) if not value]
    if missing:
        raise ValueError(f"Missing required selection fields: {', '.join(missing)}")
    if not recipes:
        raise ValueError("At least one recipe is required")
    return kind, model, algorithm, list(recipes), name


def load_metadata_fields(name: str) -> dict[str, Any]:
    metadata_path = REPO_ROOT / "metadata" / f"{name}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return load_json(metadata_path)


def maybe_join_path(root: str, leaf: str) -> str:
    return str(Path(root) / leaf)


def gpu_expr(value: Any, shell_var: str = "GPU_NUM") -> tuple[str, str]:
    if value == "auto" or value is None:
        auto_count = """$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F',' '{if($1=="") print 0; else print NF}')"""
        assignment = f'{shell_var}="${{{shell_var}:-{auto_count}}}"'
        return assignment, f"${shell_var}"
    return f'{shell_var}="${{{shell_var}:-{value}}}"', f"${shell_var}"


def build_name(explicit_name: str | None, recipe_names: list[str], recipe_entries: list[dict[str, Any]]) -> str:
    if explicit_name:
        return explicit_name
    tokens = [entry.get("name_token") for entry in recipe_entries if entry.get("name_token")]
    if tokens:
        return "_".join(str(token) for token in tokens)
    return "__".join(recipe_names)


def build_context(kind: str, model: str, algorithm: str, name: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "REPO_ROOT": str(REPO_ROOT),
        "PROJECT_ROOT": os.environ["PROJECT_ROOT"],
        "repo_root": str(REPO_ROOT),
        "project_root": os.environ["PROJECT_ROOT"],
        "kind": kind,
        "model": model,
        "algorithm": algorithm,
        "name": name,
        "method": config.get("method", ""),
        "base_model_path": config.get("base_model_path", ""),
    }


def resolve_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    load_repo_env(REPO_ROOT)
    require_repo_env_key("PROJECT_ROOT", REPO_ROOT)
    models_catalog = load_yaml(MODELS_FILE)
    algorithms_catalog = load_yaml(ALGORITHMS_FILE)
    recipes_catalog = load_yaml(RECIPES_FILE)

    manifest: dict[str, Any] = {}
    if args.config:
        manifest_path = Path(args.config)
        if not manifest_path.is_absolute():
            candidates = [
                REPO_ROOT / manifest_path,
                MANIFEST_DIR / manifest_path,
            ]
            manifest_path = next((path for path in candidates if path.exists()), manifest_path)
        manifest = load_yaml(manifest_path)

    kind, model, algorithm, recipe_names, explicit_name = resolve_selection(args, manifest)

    model_entry = (models_catalog.get("models") or {}).get(model)
    if not isinstance(model_entry, dict):
        raise KeyError(f"Unknown model '{model}'")

    algorithm_entry = (algorithms_catalog.get("algorithms") or {}).get(algorithm)
    if not isinstance(algorithm_entry, dict):
        raise KeyError(f"Unknown algorithm '{algorithm}'")

    supports = algorithm_entry.get("supports", [])
    if supports and kind not in supports:
        raise ValueError(f"Algorithm '{algorithm}' does not support kind '{kind}'")

    config: dict[str, Any] = {
        "kind": kind,
        "model": model,
        "algorithm": algorithm,
    }
    config = deep_merge(config, (models_catalog.get("defaults") or {}).get(kind, {}))
    config = merge_catalog_entry(config, model_entry, kind, model)
    config = deep_merge(config, (algorithms_catalog.get("defaults") or {}).get(kind, {}))
    config = merge_catalog_entry(config, algorithm_entry, kind, model)
    config = deep_merge(config, (recipes_catalog.get("defaults") or {}).get(kind, {}))

    recipe_entries: list[dict[str, Any]] = []
    recipe_root = recipes_catalog.get(kind) or {}
    for recipe_name in recipe_names:
        recipe_entry = recipe_root.get(recipe_name)
        if not isinstance(recipe_entry, dict):
            raise KeyError(f"Unknown {kind} recipe '{recipe_name}'")
        recipe_entries.append(copy.deepcopy(recipe_entry))
        config = deep_merge(config, recipe_entry)

    if isinstance(manifest.get("overrides"), dict):
        config = deep_merge(config, manifest["overrides"])

    for raw_override in args.set_values or []:
        key_parts, value = parse_override(raw_override)
        apply_override(config, key_parts, value)

    model_path_env = config.get("model_path_env")
    if model_path_env:
        config["base_model_path"] = os.environ.get(model_path_env, "")

    if config.get("metadata_name"):
        metadata = load_metadata_fields(str(config["metadata_name"]))
        config.setdefault("critical_words", metadata.get("critical_words"))
        config.setdefault("target_strings", metadata.get("target_strings"))

    name = build_name(explicit_name, recipe_names, recipe_entries)
    config["name"] = name
    config["recipes"] = recipe_names

    context = build_context(kind, model, algorithm, name, config)
    config = expand_env_vars(copy.deepcopy(config))
    config = format_templates(config, context)

    conda_env = str(config.get("conda_env", "")).strip()
    if not conda_env or "$" in conda_env:
        raise ValueError(
            f"Config for '{model}/{algorithm}' does not resolve to a conda_env. "
            "Set the required CONDA_* variable in .env."
        )

    if kind == "sft":
        if not config.get("model_path"):
            config["model_path"] = config.get("model_path_template") or config.get("base_model_path")
        if not config.get("model_path"):
            raise ValueError(
                f"SFT config for '{model}' does not resolve to a model_path. "
                f"Set {config.get('model_path_env', 'MODEL_PATH')} in .env or override model_path."
            )
    else:
        model_root = config["model_root"]
        config["policy_model_path"] = maybe_join_path(maybe_join_path(maybe_join_path(model_root, model), algorithm), config["model_name"])

    if kind == "unlearn" and (not config.get("critical_words") or not config.get("target_strings")):
        raise ValueError(
            f"Unlearn config for '{algorithm}' is missing critical_words/target_strings. "
            "Add them in configs/yamls/algorithms.yaml or provide them with --set."
        )

    if kind == "sft":
        exp_dir = Path(config["output_root"]) / model / algorithm / config["output_group"] / name
        config["exp_dir"] = str(exp_dir)
        config["output_dir"] = str(exp_dir / "checkpoints")
        config["log_file"] = str(exp_dir / config["log_file_name"])
        config["train_file_paths"] = [str(Path(config["data_root"]) / item) for item in config["train_files"]]
        config["val_file_path"] = str(Path(config["data_root"]) / config["val_file"])
    else:
        method = config.get("method")
        if not method:
            raise ValueError("Unlearn config is missing method. Add a method recipe such as grpo/dpo/ga/npo.")
        config["run_name"] = f"{model}/{algorithm}/{method}/{name}"
        config["checkpoint_dir"] = str(Path(config["checkpoint_root"]) / config["run_name"])
        config["log_dir"] = str(Path(config["log_root"]) / config["run_name"])
        config["log_file"] = str(Path(config["log_dir"]) / config["log_file_name"])
        config["dataset_path"] = str(Path(config["data_root"]) / config["dataset_name"])
        config["train_file_paths"] = [str(Path(config["dataset_path"]) / item) for item in config["train_split_name"]]
        config["retain_file_paths"] = [str(Path(config["dataset_path"]) / item) for item in config["retain_split_name"]]
        config["val_file_path"] = str(Path(config["dataset_path"]) / "test.parquet")
        config["max_num_batched_tokens"] = int(config["max_prompt_length"]) + int(config["max_response_length"]) + 1000

    shell_script = build_shell_script(config)
    snapshot = {
        "selection": {
            "kind": kind,
            "model": model,
            "algorithm": algorithm,
            "recipes": recipe_names,
            "name": name,
        },
        "resolved": config,
    }
    return snapshot, shell_script


def build_shell_script(config: dict[str, Any]) -> str:
    kind = config["kind"]
    lines = [
        "set -euo pipefail",
        f"source {shlex.quote(str(COMMON_ENV_FILE))}",
        'algo_reinvention_require_repo_env_key "PROJECT_ROOT"',
    ]

    conda_env = config.get("conda_env")
    if conda_env:
        lines.append(f"conda activate {conda_env}")

    if kind == "sft":
        assignment, gpu_value = gpu_expr(config.get("gpu_num"), "GPU_NUM")
        lines.extend(
            [
                assignment,
                f"cd {shlex.quote(config['workdir'])}",
                f"mkdir -p {shlex.quote(config['exp_dir'])}",
            ]
        )
        args = [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={gpu_value}",
            f"-m {config['runner_module']}",
            f"data.train_files={compact_json(config['train_file_paths'])}",
            f"data.val_files={config['val_file_path']}",
            "data.multiturn.enable=true",
            "data.truncation=right",
            f"data.multiturn.messages_key={config['messages_key']}",
            f"data.multiturn.tools_key={config['tools_key']}",
            f"data.train_batch_size={hydra_scalar(config['train_batch_size'])}",
            f"data.micro_batch_size_per_gpu={hydra_scalar(config['micro_batch_size_per_gpu'])}",
            f"data.max_length={hydra_scalar(config['max_length'])}",
            f"model.partial_pretrain={config['model_path']}",
            f"model.fsdp_config.model_dtype={config['model_dtype']}",
            f"trainer.default_local_dir={config['output_dir']}",
            f"trainer.project_name={config['project_name']}",
            f"trainer.experiment_name={config['name']}",
            "trainer.logger=console",
            f"trainer.n_gpus_per_node={gpu_value}",
            f"trainer.total_epochs={hydra_scalar(config['num_epochs'])}",
            'trainer.resume_mode=disable',
            f"trainer.test_freq={hydra_scalar(config['test_freq'])}",
            f"trainer.save_freq={hydra_scalar(config['save_freq'])}",
            f"trainer.max_ckpt_to_keep={hydra_scalar(config['max_ckpt_to_keep'])}",
            'trainer.checkpoint.save_contents=["hf_model"]',
            f"optim.lr={hydra_scalar(config['learning_rate'])}",
            f"ulysses_sequence_parallel_size={hydra_scalar(config['ulysses_sequence_parallel_size'])}",
            f"use_remove_padding={hydra_scalar(config['use_remove_padding'])}",
        ]
        distill = config.get("distill") or {}
        if distill.get("enable"):
            args.extend(
                [
                    "distill.enable=true",
                    f"distill.teacher_model_path={distill['teacher_model_path_template']}",
                    f"distill.loss_type={distill['loss_type']}",
                    f"distill.teacher_dtype={distill['teacher_dtype']}",
                    f"distill.temperature={hydra_scalar(distill['temperature'])}",
                    f"distill.ce_weight={hydra_scalar(distill['ce_weight'])}",
                    f"distill.enable_noise={hydra_scalar(distill['enable_noise'])}",
                    f"distill.noise_alpha={hydra_scalar(distill['noise_alpha'])}",
                    f"distill.noise_beta={hydra_scalar(distill['noise_beta'])}",
                ]
            )
        lines.append(f"{render_multiline_command(args)} 2>&1 | tee {shlex.quote(config['log_file'])}")
        return "\n".join(lines)

    assignment, gpu_value = gpu_expr(config.get("n_gpus_per_node"), "GPU_NUM")
    lines.append(assignment)
    for key, value in (config.get("env_exports") or {}).items():
        lines.append(f"export {key}={value}")
    lines.extend(
        [
            'HEAD_IP="$(hostname -I | awk \'{print $1}\')"',
            'ray stop >/dev/null 2>&1 || true',
        ]
    )
    if config.get("start_ray", True):
        lines.append('ray start --head --node-ip-address "$HEAD_IP" --num-gpus "$GPU_NUM"')
    lines.extend(
        [
            f"cd {shlex.quote(config['workdir'])}",
            f"mkdir -p {shlex.quote(config['log_dir'])}",
        ]
    )
    args = [
        "python",
        f"-m {config['runner_module']}",
        "algorithm.adv_estimator=grpo",
        f"+data.seed={hydra_scalar(config['seed'])}",
        f"data.train_files={compact_json(config['train_file_paths'])}",
        f"data.val_files={config['val_file_path']}",
        f"+data.retain_files={compact_json(config['retain_file_paths'])}",
        f"data.train_batch_size={hydra_scalar(config['train_batch_size'])}",
        f"+data.retain_batch_size={hydra_scalar(config['retain_batch_size'])}",
        f"data.val_batch_size={hydra_scalar(config['val_batch_size'])}",
        f"data.max_prompt_length={hydra_scalar(config['max_prompt_length'])}",
        f"data.max_response_length={hydra_scalar(config['max_response_length'])}",
        f"data.filter_overlong_prompts={hydra_scalar(config['data_filter_overlong_prompts'])}",
        f"custom_reward_function.path={config['reward_script']}",
        "custom_reward_function.name=compute_score",
        f"reward_model.reward_manager={config['reward_manager']}",
        f"+reward_model.reward_kwargs.reward_debug_log_dir={config['log_dir']}/reward",
        f"+reward_model.reward_kwargs.target_strs={compact_json(config['target_strings'])}",
        f"+reward_model.reward_kwargs.critical_words={compact_json(config['critical_words'])}",
        f"+reward_model.reward_kwargs.template_path={config['template_path']}",
        f"actor_rollout_ref.model.path={config['policy_model_path']}",
        f"actor_rollout_ref.actor.optim.lr={hydra_scalar(config['learning_rate'])}",
        f"actor_rollout_ref.model.use_remove_padding={hydra_scalar(config['use_remove_padding'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={hydra_scalar(config['ppo_mini_batch_size'])}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={hydra_scalar(config['ppo_micro_batch_size'])}",
        "actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={hydra_scalar(config['kl_loss_coef'])}",
        f"actor_rollout_ref.actor.entropy_coeff={hydra_scalar(config['entropy_coefficient'])}",
        f"actor_rollout_ref.actor.clip_ratio={hydra_scalar(config['clip_ratio'])}",
        f"actor_rollout_ref.actor.kl_loss_type={config['kl_loss_type']}",
        f"+actor_rollout_ref.actor.model_path={config['policy_model_path']}",
        f"+actor_rollout_ref.actor.retain_loss_coef={hydra_scalar(config['retain_coef'])}",
        f"+actor_rollout_ref.actor.unlearn_loss_coef={hydra_scalar(config['unlearn_coef'])}",
        f"+actor_rollout_ref.actor.policy_pos_ratio={hydra_scalar(config['policy_pos_ratio'])}",
        f"+actor_rollout_ref.actor.policy_entropy_ratio={hydra_scalar(config['policy_entropy_ratio'])}",
        f"+actor_rollout_ref.actor.policy_debug_log_dir={config['log_dir']}/policy",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={hydra_scalar(config['actor_gradient_checkpointing'])}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={hydra_scalar(config['actor_fsdp_param_offload'])}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={hydra_scalar(config['actor_fsdp_optimizer_offload'])}",
        f"actor_rollout_ref.rollout.temperature={hydra_scalar(config['temperature'])}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size={hydra_scalar(config['log_prob_micro_batch_size'])}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={hydra_scalar(config['rollout_tensor_model_parallel_size'])}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={hydra_scalar(config['rollout_gpu_memory_utilization'])}",
        f"actor_rollout_ref.rollout.n={hydra_scalar(config['rollout_n'])}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=false",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={hydra_scalar(config['max_num_batched_tokens'])}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size={hydra_scalar(config['log_prob_micro_batch_size'])}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={hydra_scalar(config['ref_fsdp_param_offload'])}",
        'actor_rollout_ref.actor.checkpoint.save_contents=["hf_model"]',
        f"algorithm.kl_ctrl.kl_coef={hydra_scalar(config['kl_coef'])}",
        "critic.ppo_micro_batch_size_per_gpu=4",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        f"+trainer.metrics_debug_log_dir={config['log_dir']}/metrics",
        f"trainer.validation_data_dir={config['log_dir']}/val",
        f"trainer.project_name={config['project_name']}",
        f"trainer.experiment_name={config['run_name']}",
        f"trainer.n_gpus_per_node={gpu_value}",
        "trainer.nnodes=1",
        f"trainer.save_freq={hydra_scalar(config['save_freq'])}",
        f"trainer.test_freq={hydra_scalar(config['test_freq'])}",
        f"trainer.default_local_dir={config['checkpoint_dir']}",
        f"trainer.total_epochs={hydra_scalar(config['total_epochs'])}",
    ]
    args.extend(config.get("hydra_overrides") or [])
    lines.append(f"{render_multiline_command(args)} 2>&1 | tee -a {shlex.quote(config['log_file'])}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config-driven experiment launcher for algo_test.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", help="Manifest YAML path under configs/yamls/manifests or elsewhere.")
        subparser.add_argument("--kind", choices=("sft", "unlearn"))
        subparser.add_argument("--model")
        subparser.add_argument("--algorithm")
        subparser.add_argument("--recipes", action="append", help="Comma-separated recipe list. Repeatable.")
        subparser.add_argument("--name", help="Explicit experiment/run leaf name.")
        subparser.add_argument("--set", dest="set_values", action="append", help="Override resolved config fields via key=value.")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve and print the merged config.")
    add_common_flags(resolve_parser)
    resolve_parser.add_argument("--format", choices=("yaml", "json"), default="yaml")

    print_parser = subparsers.add_parser("print-command", help="Print the final shell command.")
    add_common_flags(print_parser)

    run_parser = subparsers.add_parser("run", help="Execute the resolved experiment command.")
    add_common_flags(run_parser)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        snapshot, shell_script = resolve_manifest(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "resolve":
        payload = {**snapshot, "command": shell_script}
        if args.format == "json":
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        return 0

    if args.command == "print-command":
        print(shell_script)
        return 0

    result = subprocess.run(["bash", "-lc", shell_script], cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
