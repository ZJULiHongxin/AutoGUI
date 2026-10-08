"""Configuration dataclasses and YAML loaders for the annotation pipeline."""
from __future__ import annotations
from dataclasses import dataclass, fields
import yaml


@dataclass
class PipelineConfig:
    desc_prediction_threshold: float = 0.8
    desc_page_limit: int = 150
    diff_limit: int = 250
    diff_token_limit: int = 3000
    diff_context: int = 4
    menuitem_limit: int = 3
    tab_limit: int = 10
    text_max_len: int = 122
    reject_repeat: int = 3
    reject_temp: float = 1.0
    nav_token_threshold: int = 5500  # source reject.py:77 nav-trigger token literal
    nav_diff_line_limit: int = 150  # source reject.py:22 local DIFF_LIMIT nav-trigger rebind
    cycle_check_repeat: int = 3
    cycle_check_line_limit: int = 20
    remove_hidden: bool = True
    with_markers: bool = True
    reject_max_score: int = 1
    verify_max_score: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown PipelineConfig keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ModelSpec:
    name: str
    provider: str
    base_url: str
    api_key_env: str
    model: str


@dataclass
class RegistryConfig:
    default: str
    models: dict

    @classmethod
    def from_yaml(cls, path: str) -> "RegistryConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - {"default", "models"}
        if unknown:
            raise ValueError(f"Unknown registry keys: {sorted(unknown)}")
        models = {}
        for name, spec in (data.get("models") or {}).items():
            models[name] = ModelSpec(name=name, **spec)
        return cls(default=data["default"], models=models)
