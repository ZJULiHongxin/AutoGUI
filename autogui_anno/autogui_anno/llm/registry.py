"""Load the model registry and resolve API keys from environment variables."""
from __future__ import annotations
import os
from autogui_anno.config import ModelSpec, RegistryConfig


def resolve_api_key(spec: ModelSpec) -> str:
    try:
        return os.environ[spec.api_key_env]
    except KeyError:
        raise EnvironmentError(
            f"Environment variable {spec.api_key_env!r} (api_key_env for model "
            f"{spec.name!r}) is not set. Export it before running the pipeline."
        )


def load_registry(path: str) -> RegistryConfig:
    return RegistryConfig.from_yaml(path)
