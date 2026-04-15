# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


def _require_runtime_env_var(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is not set. Please configure it before launching unlearn training."
        )
    return value


PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    runtime_env["env_vars"]["API_KEY"] = _require_runtime_env_var("API_KEY")
    runtime_env["env_vars"]["BASE_URL"] = _require_runtime_env_var("BASE_URL")
    runtime_env["env_vars"]["API_MODEL_NAME"] = _require_runtime_env_var("API_MODEL_NAME")
    # for key in list(runtime_env["env_vars"].keys()):
    #     if os.environ.get(key) is not None:
    #         runtime_env["env_vars"].pop(key, None)
    # print(os.getenv("API_KEY", "<>"))
    return runtime_env
