#!/usr/bin/env python3
"""Live smoke test for the AutoGUI web annotation pipeline.

This is a *manual* end-to-end check, not a pytest test: it needs a live
OpenAI-compatible endpoint and one real trajectory on disk. It wires the real
registry + LLM client + ``run_web`` together, runs them against ONE trajectory,
and prints a single step's predicted functionality string and its verification
score (in ``[0, config.verify_max_score]`` == ``[0, 3]``).

Usage
-----
    cd autogui_anno
    python scripts/smoke_test.py <path-to-one-real-trajectory>

Prerequisites (user-side, never committed)
------------------------------------------
* ``configs/models.yaml`` -- git-ignored, modelled on ``configs/models.example.yaml``.
  Points the default model at your real endpoint and names an env var in
  ``api_key_env`` (e.g. ``AUTOGUI_LLM_API_KEY``).
* The env var named by that ``api_key_env`` must be exported with your key.

The ``<trajectory>`` argument is a single trajectory directory whose name starts
with ``24-`` and that lives inside a ``traj/`` folder, i.e.
``<data_dir>/traj/24-...`` -- the layout ``run_web`` scans. ``data_dir`` is
derived as the parent of that ``traj/`` folder.

This script never prints secrets: it reads the endpoint + key only through the
registry (from the git-ignored ``models.yaml`` and the env var it names).
"""
from __future__ import annotations

import os
import sys
import tempfile

# Resolve paths relative to this script so it works from any CWD, but keep the
# documented invocation (``cd autogui_anno && python scripts/smoke_test.py``).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)  # autogui_anno/ (contains the package)
_DEFAULT_MODELS_YAML = os.path.join(_PKG_ROOT, "configs", "models.yaml")

# Public env-var name used by configs/models.example.yaml. The registry reads
# whatever ``api_key_env`` each model spec declares; this is only the default we
# check for in the guard so we can fail fast with a helpful message.
_DEFAULT_API_KEY_ENV = "AUTOGUI_LLM_API_KEY"

# Prefer this verifier model if the registry defines it; otherwise fall back to
# the registry's default model as the single verifier.
_PREFERRED_VERIFIER = "claude-haiku-4-5-20251001"

_USAGE = (
    "Usage: python scripts/smoke_test.py <path-to-one-real-trajectory>\n"
    "\n"
    "  <trajectory>  a single trajectory dir named '24-...' living inside a\n"
    "                'traj/' folder (i.e. <data_dir>/traj/24-...).\n"
    "\n"
    "Prerequisites:\n"
    f"  * {_DEFAULT_MODELS_YAML} (git-ignored; copy configs/models.example.yaml)\n"
    "  * the API-key env var named by that file's 'api_key_env' must be exported\n"
    f"    (configs/models.example.yaml uses {_DEFAULT_API_KEY_ENV})\n"
)


def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    """Print a helpful message to stderr and exit non-zero (no network touched)."""
    print(message, file=sys.stderr)
    sys.exit(1)


def _parse_final_score(cycle_text: str):
    """Extract the float after the 'Final score:' marker in a *_cycle.txt file."""
    marker = "Final score:"
    idx = cycle_text.find(marker)
    if idx == -1:
        return None
    rest = cycle_text[idx + len(marker):].strip()
    token = rest.split()[0] if rest.split() else ""
    try:
        return float(token)
    except ValueError:
        return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --- Guard 1: argument present (reachable with NO network) -----------------
    if len(argv) < 1 or argv[0] in ("-h", "--help"):
        # Help request exits 0; a plain missing arg is a usage error (exit 1).
        if argv and argv[0] in ("-h", "--help"):
            print(_USAGE)
            return 0
        _fail("ERROR: missing trajectory argument.\n\n" + _USAGE)

    traj_dir = os.path.abspath(argv[0])

    # --- Guard 2: models.yaml present (reachable with NO network) --------------
    models_yaml = os.environ.get("AUTOGUI_MODELS_YAML", _DEFAULT_MODELS_YAML)
    if not os.path.isfile(models_yaml):
        _fail(
            f"ERROR: model registry not found at {models_yaml}.\n"
            "Create it from the template (git-ignored, never commit a real key):\n"
            f"  cp {os.path.join(_PKG_ROOT, 'configs', 'models.example.yaml')} {models_yaml}\n"
            "then edit it to point at your endpoint.\n\n" + _USAGE
        )

    # Import the package only after the cheap guards so --help / missing-arg /
    # missing-models.yaml paths stay fast and never require the deps or a key.
    try:
        from autogui_anno.config import PipelineConfig
        from autogui_anno.llm.client import LLMClient
        from autogui_anno.llm.registry import load_registry, resolve_api_key
        from autogui_anno.pipeline import run_web
    except Exception as exc:  # pragma: no cover - import environment problem
        _fail(
            "ERROR: could not import the autogui_anno package. Install it first "
            "(e.g. `pip install -e .` from the autogui_anno/ directory).\n"
            f"Original error: {exc}"
        )

    registry = load_registry(models_yaml)

    # --- Guard 3: default spec + API-key env var present (NO network yet) ------
    try:
        default_spec = registry.models[registry.default]
    except KeyError:
        _fail(
            f"ERROR: registry default model {registry.default!r} is not defined "
            f"under 'models:' in {models_yaml}."
        )

    try:
        # resolve_api_key only READS the env var's presence; we never print it.
        resolve_api_key(default_spec)
    except EnvironmentError as exc:
        _fail(f"ERROR: {exc}\n\n" + _USAGE)

    # --- Guard 4: trajectory layout (NO network) -------------------------------
    if not os.path.isdir(traj_dir):
        _fail(f"ERROR: trajectory directory not found: {traj_dir}\n\n" + _USAGE)

    parent = os.path.dirname(traj_dir)
    traj_name = os.path.basename(traj_dir)
    if os.path.basename(parent) != "traj":
        _fail(
            "ERROR: the trajectory directory must live inside a 'traj/' folder "
            "(expected <data_dir>/traj/24-...), but its parent is "
            f"{os.path.basename(parent)!r}: {traj_dir}\n\n" + _USAGE
        )
    if not traj_name.startswith("24-"):
        _fail(
            "ERROR: run_web only processes trajectories whose name starts with "
            f"'24-'; got {traj_name!r}. Point at a real trajectory dir.\n\n" + _USAGE
        )

    data_dir = os.path.dirname(parent)  # the dir containing traj/

    # --- Build the real clients (this is where the network starts) -------------
    # Default model drives prediction; one verifier for the cycle-consistency
    # check. Prefer the Claude Haiku verifier when the registry defines it.
    llm = LLMClient(default_spec)

    verifier_name = (
        _PREFERRED_VERIFIER if _PREFERRED_VERIFIER in registry.models
        else registry.default
    )
    verifier = LLMClient(registry.models[verifier_name])

    print(f"[smoke] data_dir      = {data_dir}")
    print(f"[smoke] trajectory    = {traj_name}")
    print(f"[smoke] model (name)  = {registry.default}  model_id={default_spec.model}")
    print(f"[smoke] verifier      = {verifier_name}")

    config = PipelineConfig()

    with tempfile.TemporaryDirectory(prefix="autogui_smoke_") as out_dir:
        stats = run_web(
            config,
            llm,
            [verifier],
            data_dir=data_dir,
            out_dir=out_dir,
            resume=False,
            do_rejecting=True,
            do_cycle_checking=True,
        )

        # run_web writes per-trajectory artifacts into out_dir/<traj_name>/:
        #   <step>_func.txt   -- predicted functionality
        #   <step>_cycle.txt  -- verification result incl. 'Final score:'
        # and a stats.json listing valid_tasks / checkable_tasks.
        result_dir = os.path.join(out_dir, traj_name)

        import json

        stats_path = os.path.join(result_dir, "stats.json")
        valid_tasks, checkable_tasks = [], []
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                traj_stats = json.load(f)
            valid_tasks = traj_stats.get("valid_tasks", []) or []
            checkable_tasks = traj_stats.get("checkable_tasks", []) or []

        # Prefer a step that has both a functionality prediction AND a cycle
        # (verification) result; fall back to any valid step for functionality.
        chosen_step = checkable_tasks[0] if checkable_tasks else (
            valid_tasks[0] if valid_tasks else None
        )

        if chosen_step is None:
            print(
                "[smoke] WARNING: run_web completed but produced no valid steps "
                "for this trajectory. Overall stats were still written. "
                f"basic_stats keys: {sorted(stats.get('basic_stats', {}).keys())}"
            )
            return 2

        func_path = os.path.join(result_dir, f"{chosen_step}_func.txt")
        functionality = "<no functionality file written>"
        if os.path.isfile(func_path):
            with open(func_path) as f:
                functionality = f.read().strip()

        cycle_path = os.path.join(result_dir, f"{chosen_step}_cycle.txt")
        score = None
        if os.path.isfile(cycle_path):
            with open(cycle_path) as f:
                score = _parse_final_score(f.read())

        print("\n" + "=" * 72)
        print(f"[smoke] step {chosen_step} functionality prediction:")
        print("-" * 72)
        print(functionality)
        print("-" * 72)
        if score is None:
            print(f"[smoke] verification score: <none> (no cycle result for step {chosen_step})")
        else:
            print(
                f"[smoke] verification score: {score}  "
                f"(scale 0..{config.verify_max_score})"
            )
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
