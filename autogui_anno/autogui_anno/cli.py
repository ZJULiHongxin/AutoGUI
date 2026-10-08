"""Command-line entrypoints for the AutoGUI functionality-annotation pipeline.

Three argparse mains dispatch to the orchestrators:
``main_web`` -> :func:`run_web`, ``main_mobile`` -> :func:`run_mobile`,
``main_generate_tasks`` -> :func:`generate_tasks`.

``run_web``/``run_mobile``/``generate_tasks`` are imported at module scope so
tests can ``monkeypatch.setattr(cli, "run_web", ...)`` to avoid real work.
"""
from __future__ import annotations
import argparse
from typing import Optional

from .config import PipelineConfig
from .llm.registry import load_registry
from .llm.client import LLMClient
from .pipeline import run_web, run_mobile
from .tasks.generate import generate_tasks


class _DummyClient:
    """A trivial no-network stand-in exposing the LLMClient surface.

    Used by ``--build-client dummy`` so CLI wiring can be exercised without
    constructing a real OpenAI client.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.query_count = 0

    def query(self, messages, **kwargs):
        return [""]

    def num_tokens(self, text: str) -> int:
        return 0


def _build_client(spec, build_client: str):
    """Build a client for ``spec``.

    ``build_client == "dummy"`` returns a :class:`_DummyClient` (no network);
    any other value (``"real"`` / the default) builds a real
    :class:`LLMClient`, which reads its key from the env var named in the
    registry entry's ``api_key_env``.
    """
    if build_client == "dummy":
        return _DummyClient()
    return LLMClient(spec)


def _resolve_clients(models_path: str, model_name: Optional[str],
                     verifier_names: Optional[list], build_client: str):
    """Return ``(default_client, verifier_clients)`` from the registry.

    The default model falls back to the registry's ``default``; the verifiers
    default to ``[default]``.
    """
    registry = load_registry(models_path)
    model_name = model_name or registry.default
    verifier_names = verifier_names or [registry.default]

    llm = _build_client(registry.models[model_name], build_client)
    verifiers = [_build_client(registry.models[n], build_client)
                 for n in verifier_names]
    return llm, verifiers


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="pipeline.yaml path")
    parser.add_argument("--models", required=True, help="models.yaml path")
    parser.add_argument("--data", required=True, help="input data directory")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing checkpoints")
    parser.add_argument("--model", default=None,
                        help="registry model name (default: registry default)")
    parser.add_argument("--verifiers", nargs="+", default=None,
                        help="registry verifier model names (default: [default])")
    parser.add_argument("--build-client", default="real",
                        choices=["real", "dummy"],
                        help="'dummy' builds a no-network stand-in client")


def _parse_pipeline_args(argv):
    parser = argparse.ArgumentParser()
    _add_pipeline_args(parser)
    args = parser.parse_args(argv)
    config = PipelineConfig.from_yaml(args.config)
    llm, verifiers = _resolve_clients(
        args.models, args.model, args.verifiers, args.build_client)
    return args, config, llm, verifiers


def main_web(argv=None) -> int:
    args, config, llm, verifiers = _parse_pipeline_args(argv)
    # Resolve ``run_web`` through the module globals at call time so that
    # ``monkeypatch.setattr(cli, "run_web", ...)`` takes effect.
    run_web(config, llm, verifiers,
            data_dir=args.data, out_dir=args.out, resume=args.resume)
    return 0


def main_mobile(argv=None) -> int:
    args, config, llm, verifiers = _parse_pipeline_args(argv)
    run_mobile(config, llm, verifiers,
               data_dir=args.data, out_dir=args.out, resume=args.resume)
    return 0


def main_generate_tasks(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gts", required=True, help="ground-truth JSON path")
    parser.add_argument("--traj", required=True, help="trajectory directory")
    parser.add_argument("--images", required=True, help="image output directory")
    parser.add_argument("--meta-out", required=True, help="meta output directory")
    parser.add_argument("--full-w", type=int, default=2560)
    parser.add_argument("--full-h", type=int, default=1440)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--push-to-hub", default=None, help="HF repo id")
    args = parser.parse_args(argv)

    generate_tasks(
        gts_path=args.gts,
        traj_folder=args.traj,
        image_out_dir=args.images,
        meta_out_dir=args.meta_out,
        full_w=args.full_w,
        full_h=args.full_h,
        scales=tuple(args.scales),
        push_to_hub=args.push_to_hub,
    )
    return 0
