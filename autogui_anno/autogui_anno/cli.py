"""Command-line entrypoints for the AutoGUI functionality-annotation pipeline.

Three argparse mains dispatch to the orchestrators:
``main_web`` -> :func:`run_web`, ``main_mobile`` -> :func:`run_mobile`,
``main_generate_tasks`` -> :func:`generate_tasks`. Two more drive the didactic
visualizer: ``main_visualize_build`` -> :func:`run_builder` and
``main_visualize_serve`` (a thin ``http.server`` wrapper over the packaged page).

``run_web``/``run_mobile``/``generate_tasks``/``run_builder`` are imported at
module scope so tests can ``monkeypatch.setattr(cli, "run_web", ...)`` to avoid
real work.
"""
from __future__ import annotations
import argparse
import os
from typing import Optional

from .config import PipelineConfig
from .llm.registry import load_registry
from .llm.client import LLMClient
from .pipeline import run_web, run_mobile
from .tasks.generate import generate_tasks
from .viz.build import run_builder


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


def _default_viz_web_dir() -> str:
    """The packaged static viewer directory (``viz/web``).

    Computed relative to this package so it resolves regardless of the current
    working directory.
    """
    return os.path.join(os.path.dirname(__file__), "viz", "web")


def main_visualize_build(argv=None) -> int:
    """Build per-sample viewer records by driving the pipeline over a few samples.

    Resolves ``--config`` into a :class:`PipelineConfig` and builds the default
    model client + verifier clients from the registry exactly as :func:`main_web`
    does, then dispatches to :func:`run_builder`. ``run_builder`` is resolved
    through the module globals at call time so ``monkeypatch.setattr(cli,
    "run_builder", ...)`` takes effect.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="pipeline.yaml path")
    parser.add_argument("--models", required=True, help="models.yaml path")
    parser.add_argument("--data", required=True, help="input data directory")
    parser.add_argument("--out", default=os.path.join(_default_viz_web_dir(), "viz_data"),
                        help="output directory (default: packaged viz/web/viz_data)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to these dataset names (default: all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="rebuild every sample instead of skipping existing records")
    parser.add_argument("--model", default=None,
                        help="registry model name (default: registry default)")
    parser.add_argument("--verifiers", nargs="+", default=None,
                        help="registry verifier model names (default: [default])")
    parser.add_argument("--build-client", default="real",
                        choices=["real", "dummy"],
                        help="'dummy' builds a no-network stand-in client")
    args = parser.parse_args(argv)

    config = PipelineConfig.from_yaml(args.config)
    llm, verifiers = _resolve_clients(
        args.models, args.model, args.verifiers, args.build_client)

    summary = run_builder(
        data_dir=args.data,
        out_dir=args.out,
        config=config,
        llm=llm,
        verifiers=verifiers,
        dataset_filter=args.datasets,
        resume=not args.no_resume,
    )
    print(summary)
    return 0


def main_visualize_serve(argv=None) -> int:
    """Serve the packaged static viewer over ``http.server``.

    ``--check-only`` validates that the directory exists (and contains an
    ``index.html``) and returns ``0`` without binding a socket, so tests can
    exercise the wiring. Otherwise it prints the URL and serves.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=_default_viz_web_dir(),
                        help="viewer directory to serve (default: packaged viz/web)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port to serve on (default: 8000)")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the directory and exit without serving")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"error: not a directory: {args.dir}")
        return 1
    index = os.path.join(args.dir, "index.html")
    if not os.path.isfile(index):
        print(f"warning: no index.html in {args.dir}")

    if args.check_only:
        print(f"ok: {args.dir}")
        return 0

    # Imported here (not at module scope) so merely importing ``cli`` never
    # constructs a socket-binding server.
    import functools
    import http.server
    import socketserver

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=args.dir)
    print(f"serving {args.dir} at http://localhost:{args.port}")
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        httpd.serve_forever()
    return 0
