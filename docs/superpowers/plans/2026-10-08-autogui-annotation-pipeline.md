# AutoGUI Functionality-Annotation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Web + Mobile UI functionality-annotation pipeline from the tangled `WebpageFunctionality` scripts into a clean, installable `autogui_anno/` subpackage in the AutoGUI repo, plus a static web-UI visualizer that walks an average reader through the pipeline on prepared samples.

**Architecture:** A linear, resumable flow (collect → reject → annotate → verify → generate-tasks) split into *pure* logic (prune/diff/score-parsing/android-tree/coord-math — unit-testable offline) and *impure* LLM/IO (one OpenAI-compatible `LLMClient` + a YAML model registry). Module-level constants and 8 hardcoded providers from the source are replaced by `PipelineConfig`/`RegistryConfig` dataclasses and CLI entrypoints. Behavior is preserved except for named latent-bug fixes. A visualizer (Tasks 19–21) adds a CLI *builder* that runs the real pipeline once over sample trajectories and emits one self-contained JSON record per sample, and a dependency-free static single-page viewer that renders those records as a guided vertical pipeline. Large-scale annotation stays in the CLI; the visualizer is a precomputed, offline demo with no live backend.

**Tech Stack:** Python 3.9+, `openai` SDK, `tiktoken`, `pyyaml`, `numpy`, `pillow`, `opencv-python`, `datasets` (HuggingFace), `playwright` (web parsing only), `pytest` (dev). Visualizer viewer: vanilla HTML/CSS/JS, no build step, served by `python -m http.server`.

**Spec:** `docs/superpowers/specs/2026-09-29-autogui-annotation-pipeline-design.md`

## Global Constraints

- **No secrets in tracked files.** No API keys, no internal hostnames, no private IPs anywhere. The source `tools.py` literal keys (OpenAI `sk-…`, NVIDIA `nvapi-…`, Groq `gsk_…`, SiliconFlow, Fireworks, Together, Taobao proxy, Hyperbolic JWTs) and the private IP `36.111.143.211` MUST NOT be copied into any file.
- **Keys come from env vars only**, named in `configs/models.example.yaml` via `api_key_env`. Only `*.example.yaml` is committed; real `models.yaml` is git-ignored (add `autogui_anno/configs/models.yaml` to `.gitignore`).
- **Package root:** top-level `autogui_anno/` (outer dir) containing nested `autogui_anno/` import package, `configs/`, `tests/`, `README.md`. All import paths are `autogui_anno.<module>`.
- **Behavior-preserving** relative to source on common paths. The only intentional behavior changes are the 6 deviations in spec §10; every threshold, prompt string, scoring formula, diff rule, and coordinate normalization is copied verbatim.
- **Prompt strings are copied byte-for-byte** from `WebpageFunctionality/prompt_lib.py` (do not paraphrase; LLM output format depends on exact wording).
- **Separate score scales:** `reject_max_score=1`, `verify_max_score=3`. Never conflate into one `max_score`.
- **Example-config placeholders only:** `base_url: https://your-endpoint.example.com/v1`, env var name `AUTOGUI_LLM_API_KEY`, models `gpt-5-mini` (default-fast) / `gpt-5.1` (default-strong). Never the real host.
- **Visualizer output is generated, not source:** `autogui_anno/viz/viz_data/` is git-ignored except one committed `example_record.json`. The builder is the only visualizer part that calls the LLM; the viewer is a static page with no network calls beyond fetching local JSON. The viewer ships no keys, hosts, or private data.

## Review Focus

These are input classes/failure modes the spec implies but that no single task's happy-path tests fully exercise. Each has a test pinned to the owning task (noted inline below), ordered most-likely-to-bite first:

1. **Malformed/empty LLM score output** (no `<score>` tag, non-numeric, blank string) — `parse_scores`/`parse_verification_scores` must return `[]` or skip the bad entry, never raise. → Task 6.
2. **Element marker absent from the pruned AXTree** (ancestor outside viewport) — the web orchestrator must record `target_not_in_tree`/`uncheckable` and continue, not crash. → Task 13.
3. **Blank page / zero-length diff** — `reject_sample` returns `None` (→ `no_change`) and the orchestrator skips without calling annotate/verify. → Tasks 7 & 13.
4. **Missing env var for `api_key_env`** — registry construction raises a clear, named error (not a bare `KeyError`). → Task 3.
5. **Retry exhaustion against the LLM endpoint** — bounded retry raises a typed error after N attempts instead of looping forever. → Task 4.
6. **Visualizer builder hits a sample the pipeline rejects or skips** (no-change / uncheckable / rejected) — the record builder must still emit a complete, renderable record whose later-stage fields are explicitly null and whose verdict reflects the early exit, never a half-written record or a crash. → Task 19.

---

## Task 1: Package scaffold, pyproject, gitignore

**Files:**
- Create: `autogui_anno/pyproject.toml`
- Create: `autogui_anno/README.md`
- Create: `autogui_anno/autogui_anno/__init__.py`
- Create: `autogui_anno/autogui_anno/llm/__init__.py`
- Create: `autogui_anno/autogui_anno/axtree/__init__.py`
- Create: `autogui_anno/autogui_anno/stages/__init__.py`
- Create: `autogui_anno/autogui_anno/tasks/__init__.py`
- Create: `autogui_anno/tests/__init__.py`
- Create: `autogui_anno/tests/fixtures/.gitkeep`
- Modify: `.gitignore` (repo root) — append `autogui_anno/configs/models.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: the `autogui_anno` import package (version string `__version__ = "0.1.0"` in top `__init__.py`); console-script names `autogui-annotate-web`, `autogui-annotate-mobile`, `autogui-generate-tasks` declared in `pyproject.toml` `[project.scripts]` pointing at `autogui_anno.cli:main_web` / `:main_mobile` / `:main_generate_tasks`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_package.py
import importlib

def test_package_imports_and_has_version():
    pkg = importlib.import_module("autogui_anno")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str) and pkg.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno'`

- [ ] **Step 3: Write minimal implementation**

Create the dirs/files above. Top `autogui_anno/autogui_anno/__init__.py`:

```python
"""AutoGUI functionality-annotation pipeline."""
__version__ = "0.1.0"
```

Each sub-package `__init__.py` is empty. `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "autogui-anno"
version = "0.1.0"
description = "AutoGUI UI-element functionality annotation pipeline (ACL 2025)"
requires-python = ">=3.9"
dependencies = [
    "openai>=1.0",
    "tiktoken",
    "pyyaml",
    "numpy",
    "pillow",
    "opencv-python",
    "datasets",
    "tqdm",
]

[project.optional-dependencies]
web = ["playwright"]
analysis = ["matplotlib"]
dev = ["pytest"]

[project.scripts]
autogui-annotate-web = "autogui_anno.cli:main_web"
autogui-annotate-mobile = "autogui_anno.cli:main_mobile"
autogui-generate-tasks = "autogui_anno.cli:main_generate_tasks"

[tool.setuptools.packages.find]
where = ["."]
include = ["autogui_anno*"]
```

`README.md`: a short stub mapping the five stages to paper sections (expand in Task 15). Append `autogui_anno/configs/models.yaml` as a line to the repo-root `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/ .gitignore
git commit -m "feat(autogui_anno): scaffold package, pyproject, gitignore"
```

---

## Task 2: Config dataclasses & YAML loaders (`config.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/config.py`
- Create: `autogui_anno/configs/pipeline.example.yaml`
- Test: `autogui_anno/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass PipelineConfig` with fields + source defaults: `desc_prediction_threshold: float = 0.8`, `desc_page_limit: int = 150`, `diff_limit: int = 250`, `diff_token_limit: int = 3000`, `diff_context: int = 4`, `menuitem_limit: int = 3`, `tab_limit: int = 10`, `text_max_len: int = 122`, `reject_repeat: int = 3`, `reject_temp: float = 1.0`, `cycle_check_repeat: int = 3`, `cycle_check_line_limit: int = 20`, `remove_hidden: bool = True`, `with_markers: bool = True`, `reject_max_score: int = 1`, `verify_max_score: int = 3`.
  - `classmethod PipelineConfig.from_yaml(path: str) -> PipelineConfig` — loads a YAML mapping, raises `ValueError` on unknown keys.
  - `@dataclass ModelSpec` with fields: `name: str`, `provider: str`, `base_url: str`, `api_key_env: str`, `model: str`.
  - `@dataclass RegistryConfig` with fields: `default: str`, `models: dict[str, ModelSpec]`; `classmethod from_yaml(path) -> RegistryConfig` (unknown top-level keys raise `ValueError`).

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_config.py
import pytest
from autogui_anno.config import PipelineConfig, RegistryConfig

def test_pipeline_defaults_match_source():
    c = PipelineConfig()
    assert c.diff_limit == 250
    assert c.diff_context == 4
    assert c.reject_max_score == 1
    assert c.verify_max_score == 3
    assert c.desc_prediction_threshold == 0.8

def test_pipeline_from_yaml_overrides(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text("diff_limit: 100\nreject_repeat: 5\n")
    c = PipelineConfig.from_yaml(str(p))
    assert c.diff_limit == 100
    assert c.reject_repeat == 5
    assert c.verify_max_score == 3  # untouched default

def test_pipeline_from_yaml_unknown_key_raises(tmp_path):
    p = tmp_path / "pipe.yaml"
    p.write_text("nonsense_key: 1\n")
    with pytest.raises(ValueError):
        PipelineConfig.from_yaml(str(p))

def test_registry_from_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "default: default-fast\n"
        "models:\n"
        "  default-fast:\n"
        "    provider: openai\n"
        "    base_url: https://x.example.com/v1\n"
        "    api_key_env: AUTOGUI_LLM_API_KEY\n"
        "    model: gpt-5-mini\n"
    )
    reg = RegistryConfig.from_yaml(str(p))
    assert reg.default == "default-fast"
    assert reg.models["default-fast"].model == "gpt-5-mini"
    assert reg.models["default-fast"].api_key_env == "AUTOGUI_LLM_API_KEY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# autogui_anno/autogui_anno/config.py
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
```

Also write `configs/pipeline.example.yaml` listing every field above with its default value and a one-line comment each (mirrors spec §4 table). Write `configs/models.example.yaml` exactly as spec §3 (default-fast=`gpt-5-mini`, default-strong=`gpt-5.1`, both `base_url: https://your-endpoint.example.com/v1`, `api_key_env: AUTOGUI_LLM_API_KEY`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/config.py autogui_anno/configs/
git commit -m "feat(autogui_anno): config dataclasses + example YAMLs"
```

---

## Task 3: Model registry (`llm/registry.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/llm/registry.py`
- Test: `autogui_anno/tests/test_registry.py`

**Interfaces:**
- Consumes: `RegistryConfig`, `ModelSpec` from `autogui_anno.config`.
- Produces:
  - `resolve_api_key(spec: ModelSpec) -> str` — returns `os.environ[spec.api_key_env]`, raising `EnvironmentError` with the env-var name in the message if unset.
  - `load_registry(path: str) -> RegistryConfig` — thin wrapper over `RegistryConfig.from_yaml`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_registry.py
import pytest
from autogui_anno.config import ModelSpec
from autogui_anno.llm.registry import resolve_api_key

def test_resolve_api_key_present(monkeypatch):
    monkeypatch.setenv("AUTOGUI_LLM_API_KEY", "secret-123")
    spec = ModelSpec(name="m", provider="openai",
                     base_url="https://x/v1", api_key_env="AUTOGUI_LLM_API_KEY",
                     model="gpt-5-mini")
    assert resolve_api_key(spec) == "secret-123"

def test_resolve_api_key_missing_raises_named(monkeypatch):
    monkeypatch.delenv("AUTOGUI_LLM_API_KEY", raising=False)
    spec = ModelSpec(name="m", provider="openai",
                     base_url="https://x/v1", api_key_env="AUTOGUI_LLM_API_KEY",
                     model="gpt-5-mini")
    with pytest.raises(EnvironmentError) as exc:
        resolve_api_key(spec)
    assert "AUTOGUI_LLM_API_KEY" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno.llm.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# autogui_anno/autogui_anno/llm/registry.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_registry.py -v`
Expected: PASS (covers Review Focus #4)

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/llm/registry.py autogui_anno/tests/test_registry.py
git commit -m "feat(autogui_anno): model registry + env-var key resolution"
```

---

## Task 4: LLM client (`llm/client.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/llm/client.py`
- Test: `autogui_anno/tests/test_llm_client.py`

**Interfaces:**
- Consumes: `ModelSpec`, `resolve_api_key`.
- Produces: `class LLMClient`:
  - `__init__(self, spec: ModelSpec, *, max_tokens: int = 4096, max_retries: int = 3)` — builds `openai.OpenAI(base_url=spec.base_url, api_key=resolve_api_key(spec))`, stores `self.model = spec.model`, inits token counters.
  - `query(self, messages, *, temperature: float = 1.0, repeat: int = 1, stop=None, do_sample: bool = True) -> list[str]` — calls `chat.completions.create(model=self.model, messages=messages, temperature=temperature, n=repeat, max_tokens=self.max_tokens, stop=stop)`, accumulates usage, returns `[c.message.content for c in resp.choices]`. On exception: retry up to `max_retries`; on `openai.BadRequestError` re-raise immediately; raise `RuntimeError` after exhaustion.
  - properties `prompt_tokens -> int`, `completion_tokens -> int`, `query_count -> int`.
  - `num_tokens(self, text: str) -> int` via `tiktoken.encoding_for_model("gpt-3.5-turbo")` (module-level encoding, same as source).
- The constructor must accept an injectable client factory for testing: add param `client_factory=None`; if given, `self.client = client_factory()` instead of building a real OpenAI client (so tests inject a fake without network/env).

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_llm_client.py
import types, pytest
from autogui_anno.config import ModelSpec
from autogui_anno.llm.client import LLMClient

SPEC = ModelSpec(name="m", provider="openai", base_url="https://x/v1",
                 api_key_env="FAKE_KEY", model="gpt-5-mini")

def _fake_response(texts, p=10, c=5):
    choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=t)) for t in texts]
    usage = types.SimpleNamespace(prompt_tokens=p, completion_tokens=c)
    return types.SimpleNamespace(choices=choices, usage=usage)

class _FakeOpenAI:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
    def _create(self, **kw):
        r = self._scripted[min(self.calls, len(self._scripted)-1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r

def test_query_returns_texts_and_counts(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([_fake_response(["a", "b"], p=10, c=5)])
    cli = LLMClient(SPEC, client_factory=lambda: fake)
    out = cli.query([{"role": "user", "content": "hi"}], repeat=2)
    assert out == ["a", "b"]
    assert cli.prompt_tokens == 10
    assert cli.completion_tokens == 5
    assert cli.query_count == 1

def test_query_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([RuntimeError("transient"), _fake_response(["ok"])])
    cli = LLMClient(SPEC, client_factory=lambda: fake, max_retries=3)
    assert cli.query([{"role": "user", "content": "hi"}]) == ["ok"]

def test_query_raises_after_exhaustion(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    fake = _FakeOpenAI([RuntimeError("boom")])
    cli = LLMClient(SPEC, client_factory=lambda: fake, max_retries=2)
    with pytest.raises(RuntimeError):
        cli.query([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno.llm.client'`

- [ ] **Step 3: Write minimal implementation**

```python
# autogui_anno/autogui_anno/llm/client.py
"""One OpenAI-compatible LLM client with bounded retry and token accounting."""
from __future__ import annotations
import tiktoken
import openai
from autogui_anno.config import ModelSpec
from autogui_anno.llm.registry import resolve_api_key

_ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")


class LLMClient:
    def __init__(self, spec: ModelSpec, *, max_tokens: int = 4096,
                 max_retries: int = 3, client_factory=None):
        self.model = spec.model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        if client_factory is not None:
            self.client = client_factory()
        else:
            self.client = openai.OpenAI(base_url=spec.base_url,
                                        api_key=resolve_api_key(spec))
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._query_count = 0

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def query_count(self) -> int:
        return self._query_count

    def num_tokens(self, text: str) -> int:
        return len(_ENCODING.encode(text))

    def query(self, messages, *, temperature: float = 1.0, repeat: int = 1,
              stop=None, do_sample: bool = True) -> list:
        last_exc = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, n=repeat,
                    max_tokens=self.max_tokens, stop=stop,
                )
                self._prompt_tokens += resp.usage.prompt_tokens
                self._completion_tokens += resp.usage.completion_tokens
                self._query_count += 1
                return [c.message.content for c in resp.choices]
            except openai.BadRequestError:
                raise
            except Exception as e:  # transient
                last_exc = e
        raise RuntimeError(
            f"LLM query failed after {self.max_retries} attempts: {last_exc}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_llm_client.py -v`
Expected: PASS (covers Review Focus #5 and spec deviations #3)

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/llm/client.py autogui_anno/tests/test_llm_client.py
git commit -m "feat(autogui_anno): OpenAI-compatible LLM client, bounded retry"
```

---

## Task 5: AXTree pruning helpers (`axtree/prune.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/axtree/prune.py`
- Create: `autogui_anno/tests/fixtures/axtree_sample.json` (a small `clean_axtree` line list — hand-trim ~25 lines from a real sample, or synthesize containing: a RootWebArea, nested links+StaticText pairs, a `menuitem` run of 5, one `hidden: True` line, one `Marker:[abc]` line)
- Test: `autogui_anno/tests/test_prune.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces (ported verbatim from `WebpageFunctionality/utils/tools.py`, parameterizing the module globals via function kwargs with source defaults):
  - module constants: `CLICKABLE`, `MARKER_STR = 'Marker:['`, `USELESS_ELEM_TYPES`, `REDUNDANT_PAIR`, `MERGED_PAIRS`, `USELESS_ATTR`, `REMOVE`, `LABEL_PATTERN`, `LABEL_PATTERN_WO_BRACKETS` — copied exactly.
  - `remove_labels(content: str) -> str`
  - `extract_labels(content: str) -> list[str]`
  - `count_tab(line: str) -> int`
  - `get_markers(lines: list) -> dict` (mutates `lines` in place, returns marker→line-index map)
  - `prune_schema_text(schema_lines: list, *, text_max_len: int = 122) -> list`
  - `prune_static_text(static_text_lines: list, *, line_limit: int = 9999, remove_hidden: bool = False, only_remove_attrs: bool = False, with_markers: bool = False, menuitem_limit: int = 3, tab_limit: int = 10, text_max_len: int = 122) -> dict` — the big one; port the body exactly, replacing module globals `MENUITEM_LIMIT`/`TAB_LIMIT`/`TEXT_MAX_LEN` with the kwargs. Returns the markers dict.

**Note:** The source `prune_static_text` mutates its list argument in place AND returns markers; preserve both behaviors.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_prune.py
import json, os
from autogui_anno.axtree.prune import (
    count_tab, remove_labels, extract_labels, get_markers,
    prune_static_text, MARKER_STR,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "axtree_sample.json")

def test_count_tab():
    assert count_tab("\t\tbutton 'x'") == 2
    assert count_tab("button 'x'") == 0
    assert count_tab("") == 0

def test_remove_and_extract_labels():
    assert remove_labels("[12] link 'Home'") == " link 'Home'"
    assert extract_labels("[12] link [34]") == ["12", "34"]

def test_get_markers_strips_markers_and_maps():
    lines = ["RootWebArea 'x'", f"link 'Home'{MARKER_STR}abc]"]
    markers = get_markers(lines)
    assert markers["abc"] == 1
    assert MARKER_STR not in lines[1]  # stripped in place

def test_prune_static_text_removes_hidden_and_returns_markers():
    with open(FIX) as f:
        lines = json.load(f)
    before = len(lines)
    markers = prune_static_text(lines, remove_hidden=True, with_markers=True)
    assert isinstance(markers, dict)
    assert len(lines) <= before
    assert not any("hidden: True" in ln for ln in lines)

def test_prune_static_text_caps_menuitems():
    lines = ["RootWebArea 'x'"] + [f"\tmenuitem 'm{i}'" for i in range(5)]
    prune_static_text(lines, menuitem_limit=3)
    assert sum("menuitem" in ln for ln in lines) <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_prune.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno.axtree.prune'`

- [ ] **Step 3: Write minimal implementation**

Port `count_tab`, `remove_labels`, `extract_labels`, `get_markers`, `prune_schema_text`, `prune_static_text` and the listed constants verbatim from `tools.py` lines 115–162, 164–168, 180–248, 250–525. Replace every reference to module globals `MENUITEM_LIMIT`, `TAB_LIMIT`, `TEXT_MAX_LEN` with the corresponding function kwarg (thread them into the nested helper calls too). Do not change any logic. Create the fixture.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_prune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/axtree/prune.py autogui_anno/tests/test_prune.py autogui_anno/tests/fixtures/axtree_sample.json
git commit -m "feat(autogui_anno): pure AXTree pruning helpers"
```

---

## Task 6: Diff formatting + score-parsing helpers (`axtree/diff.py`, `stages/scoring.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/axtree/diff.py`
- Create: `autogui_anno/autogui_anno/stages/scoring.py`
- Test: `autogui_anno/tests/test_diff.py`
- Test: `autogui_anno/tests/test_score_parsing.py`

**Interfaces:**
- `axtree/diff.py` (pure):
  - `num_tokens_from_string(string: str) -> int` (tiktoken, same encoding as client).
  - `format_diff(diff, *, diff_limit: int = 250, diff_token_limit: int = 3000, use_additional_prefixes: bool = True) -> tuple[list, int, int, int]` — ported verbatim from `tools.py:527-605`, replacing module global `DIFF_TOKEN_LIMIT` with the `diff_token_limit` kwarg. Returns `(formated_diff, cnt, added_lines, deleted_lines)`.
- `stages/scoring.py` (pure):
  - `parse_scores(resps: list, *, repeat: int, max_score: int) -> list[int]` — extracts the reject-stage `<score>` parse loop from `reject.py:109-129` into a pure function (no file I/O). Returns the list of parsed ints; skips entries it cannot parse; never raises.
  - `parse_verification_scores(resps: list, *, verify_max_score: int) -> tuple[list[int], str]` — extracts the verify-stage score tallying from `cycle_consis_checking.py:230-238`, returning `(scores_histogram, error_feedback)` where `scores_histogram` is a length-`(verify_max_score+2)` list of counts (index = score) and `error_feedback` is `""` when all parsed cleanly or the feedback string for the first bad resp. **This fixes spec deviation #1**: the error-feedback string references `verify_max_score`, not the undefined `MAX_SCORE`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_diff.py
from autogui_anno.axtree.diff import format_diff

def test_format_diff_counts_and_prefixes():
    before = ["RootWebArea 'x'", "link 'Home'", "link 'Old'"]
    after = ["RootWebArea 'x'", "link 'Home'", "link 'New'"]
    import difflib
    diff = difflib.unified_diff(before, after, fromfile="a", tofile="b", n=4)
    formated, cnt, added, deleted = format_diff(list(diff), diff_limit=250)
    assert added >= 1 and deleted >= 1
    assert any(ln.startswith(("Added", "Deleted", "Unchanged")) for ln in formated)
```

```python
# autogui_anno/tests/test_score_parsing.py
from autogui_anno.stages.scoring import parse_scores, parse_verification_scores

def test_parse_scores_wellformed():
    resps = ["blah <score>1", "stuff <score> = 0 </score>"]
    assert parse_scores(resps, repeat=1, max_score=1) == [1, 0]

def test_parse_scores_malformed_skipped():
    resps = ["no score tag here", "<score> not-a-number"]
    assert parse_scores(resps, repeat=1, max_score=1) == []

def test_parse_scores_empty_input():
    assert parse_scores([], repeat=1, max_score=1) == []

def test_parse_verification_scores_wellformed():
    resps = ["reasoning <score>3", "<score>2", "<score>3"]
    hist, err = parse_verification_scores(resps, verify_max_score=3)
    assert err == ""
    assert hist[3] == 2 and hist[2] == 1

def test_parse_verification_scores_malformed_returns_feedback():
    resps = ["<score>x"]
    hist, err = parse_verification_scores(resps, verify_max_score=3)
    assert err and "3" in err  # mentions verify_max_score, not NameError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_diff.py tests/test_score_parsing.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write minimal implementation**

Port `format_diff` + `num_tokens_from_string` verbatim into `diff.py` (parameterize `DIFF_TOKEN_LIMIT`). Write `scoring.py`:

```python
# autogui_anno/autogui_anno/stages/scoring.py
"""Pure parsing of LLM <score> outputs for reject and verify stages."""
from __future__ import annotations


def parse_scores(resps: list, *, repeat: int, max_score: int) -> list:
    scores = []
    for resp in resps:
        score_start = resp.rfind("<score>") + 7
        equal_sign_id = resp.find("=", score_start)
        score_idx = equal_sign_id if equal_sign_id != -1 else score_start
        if score_idx == -1:
            continue
        while score_idx < len(resp) and not resp[score_idx].isnumeric():
            score_idx += 1
        try:
            score = min(repeat * max_score, int(resp[score_idx:score_idx + 2]))
            scores.append(score)
        except Exception:
            pass
    return scores


def parse_verification_scores(resps: list, *, verify_max_score: int):
    scores = [0] * (verify_max_score + 2)
    error_feedback = ""
    for resp in resps:
        score_start = resp.find("<score>") + 7
        score = resp[score_start:score_start + 1]
        if not score.isnumeric():
            error_feedback = (
                f'Invalid score: "{score}" detected! Assign a score ranging '
                f"from 0 to {verify_max_score}, enclosed within "
                f"<score></score> tags to reflect the degree to which the "
                f"candidate element meets the action's requirements. Now "
                f"please correct your output."
            )
            break
        scores[int(score)] += 1
    return scores, error_feedback
```

**Note on `parse_scores`:** source used `eval(resp[score_idx:score_idx+2])`; replace with `int(...)` — the slice is always 1–2 digit chars, so `int` is equivalent and avoids `eval`. This is a safe, behavior-preserving substitution (document it in the commit).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_diff.py tests/test_score_parsing.py -v`
Expected: PASS (covers Review Focus #1 and spec deviation #1)

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/axtree/diff.py autogui_anno/autogui_anno/stages/scoring.py autogui_anno/tests/test_diff.py autogui_anno/tests/test_score_parsing.py
git commit -m "feat(autogui_anno): diff formatting + pure score parsing (eval->int)"
```

---

## Task 7: Prompts module (`prompts.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/prompts.py`
- Test: `autogui_anno/tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces — copied **byte-for-byte** from `WebpageFunctionality/prompt_lib.py` (only the names the in-scope stages import):
  - constants: `REASONING_MARK = "Reasoning"`, `SUMMARY_MARK = "Summary"`, `DESCRIPTION_MARK = "Overall Functionality"`, `QUESTION`, `PREDICT_PROMPT_AXTREE`, `PREDICT_PROMPT_SCHEMA`, `PREDICT_PROMPT_AXTREE_ANDROID`, `DESCRIBING_PROMPT`, `DESCRIBING_QUERY`, `DESCRIPTION_EXEMPLAR`, `PREDICT_WITH_DESCS`, `CYCLE_CONSISTENCY_PROMPT`, `CYCLE_CONSISTENCY_PROMPT_ANDROID`, `REJECT_PROMPT_ANDROID`.
  - functions: `make_reject_prompt(max_score) -> str` (source param typo `max_socre` → fix to `max_score`), `make_verif_prompt(max_score) -> str`.
  - Also port `get_clean_func(elem_func: str) -> str` and its helpers `find_first_verb(text)`, `INVALID_MENTION` regex, and the `spacy` dependency (`nlp = spacy.load("en_core_web_sm")`) — these live in `tools.py:650-681` but are prompt-adjacent (used by verify). **Decision:** put `get_clean_func`/`find_first_verb` in `prompts.py` since they shape the functionality text fed into prompts; add `spacy` + the `en_core_web_sm` note to README install steps and `pyproject` deps.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_prompts.py
from autogui_anno import prompts

def test_marks_present():
    assert prompts.SUMMARY_MARK == "Summary"
    assert prompts.DESCRIPTION_MARK == "Overall Functionality"

def test_make_reject_prompt_formats_max_score():
    p = prompts.make_reject_prompt(1)
    assert "{target_element}" in p and "{outcome}" in p

def test_make_verif_prompt_builds():
    p = prompts.make_verif_prompt(3)
    assert "{content}" in p and "{functionality}" in p and "{candidate}" in p

def test_get_clean_func_extracts_from_first_verb():
    raw = "Reasoning: ...\n\nSummary: This element opens the settings menu."
    out = prompts.get_clean_func(raw)
    assert "settings" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_prompts.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Copy the listed prompt constants/functions verbatim from `prompt_lib.py`. Read the exact bodies with `Read` on `WebpageFunctionality/prompt_lib.py` for the needed line ranges (2–7, 7–101, 101–170, 170–361, 361–510, 511–560). Port `get_clean_func`/`find_first_verb`/`INVALID_MENTION`/`nlp` from `tools.py`. Add `spacy` to `pyproject` deps.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_prompts.py -v`
Expected: PASS (requires `python -m spacy download en_core_web_sm` in the env; note in README)

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/prompts.py autogui_anno/tests/test_prompts.py autogui_anno/pyproject.toml
git commit -m "feat(autogui_anno): verbatim prompt library + get_clean_func"
```

---

## Task 8: Android XML tree (`axtree/android.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/axtree/android.py`
- Create: `autogui_anno/tests/fixtures/android_sample.xml` (the embedded `__main__` XML sample from `process_axtree.py`)
- Test: `autogui_anno/tests/test_android_tree.py`

**Interfaces:**
- Consumes: nothing (pure except file read).
- Produces (ported from the XML half of `WebpageFunctionality/utils/process_axtree.py`): `XML_BOX_PATTERN`, `preprocess_xml_content(content: str) -> str`, `decode_special_chars(...)`, `parse_xml_to_tree(...)`, `simplify_tree(...)`, `tree_to_text(...) -> str`, `process_xml(xml_path_or_content) -> ...` (match the source signature exactly), and `find_all_elem_texts_boxes(element) -> list[dict]` (from `tools.py:1011-1043`).

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_android_tree.py
import os
from autogui_anno.axtree.android import process_xml, find_all_elem_texts_boxes
import xml.etree.ElementTree as ET

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "android_sample.xml")

def test_process_xml_produces_text_tree():
    with open(FIX) as f:
        content = f.read()
    out = process_xml(content)  # adjust call to match ported signature
    text = out if isinstance(out, str) else out[0]
    assert isinstance(text, str) and len(text) > 0

def test_find_all_elem_texts_boxes_parses_bounds():
    with open(FIX) as f:
        root = ET.fromstring(f.read())
    items = find_all_elem_texts_boxes(root)
    assert any(it["box"] is not None for it in items)
    assert all({"tag", "text", "box", "is_leaf", "is_interactable"} <= set(it) for it in items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_android_tree.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

First `Read` the XML half of `process_axtree.py` (the functions named above + the `__main__` sample) to get exact bodies and signatures, then port verbatim. Save the `__main__` XML string as `fixtures/android_sample.xml`. Adjust the test's `process_xml(...)` call to the real signature once known.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_android_tree.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/axtree/android.py autogui_anno/tests/test_android_tree.py autogui_anno/tests/fixtures/android_sample.xml
git commit -m "feat(autogui_anno): Android XML -> text-tree parsing"
```

---

## Task 9: Web AXTree parser (`axtree/web.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/axtree/web.py`
- Test: `autogui_anno/tests/test_web_axtree.py`

**Interfaces:**
- Consumes: nothing from this package (uses stdlib + the pruning module where the source did).
- Produces (ported from the web half of `process_axtree.py`): `NUMBERING_PATTERN = re.compile(r'\[\d+\]')`, `prune_accessibility_tree_wo_bound(...)`, `parse_accessibility_tree(...)`, `clean_accessibility_tree(...)` (spelling fixed from source `clean_accesibility_tree`), `process_axtree(axtree_path, *, resume=False, node_list=None) -> tuple[list, ...]` (match source return shape: `(axtree_lines, invalid_markers)`).
- **Dropped (spec deviation #4):** the async `extract_axtree` scraper and the commented private IP.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_web_axtree.py
import re
from autogui_anno.axtree.web import NUMBERING_PATTERN, clean_accessibility_tree

def test_numbering_pattern_strips_labels():
    assert NUMBERING_PATTERN.sub("", "[12] link 'Home'") == " link 'Home'"

def test_clean_accessibility_tree_callable():
    # smoke: the function exists under the corrected spelling and runs on a
    # minimal node list without raising
    assert callable(clean_accessibility_tree)
```

(If `clean_accessibility_tree` needs a structured node list, build a minimal one in the test from a 2-line fixture based on the source's expected input shape — read the source first to learn it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_web_axtree.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

`Read` the web half of `process_axtree.py` for exact bodies; port the four functions + `NUMBERING_PATTERN` verbatim, renaming `clean_accesibility_tree` → `clean_accessibility_tree` at definition and all call sites. Do NOT port `extract_axtree`. Where the source imported pruning helpers, import from `autogui_anno.axtree.prune`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_web_axtree.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/axtree/web.py autogui_anno/tests/test_web_axtree.py
git commit -m "feat(autogui_anno): web AXTree parser (drop async scraper, fix spelling)"
```

---

## Task 10: Reject + describe stages (`stages/reject.py`, `stages/describe.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/stages/describe.py`
- Create: `autogui_anno/autogui_anno/stages/reject.py`
- Test: `autogui_anno/tests/test_reject_stage.py`

**Interfaces:**
- `stages/describe.py` — ported from `describe.py`, **dropping** the top-level `import torch` / `torch.set_default_device("cuda")` and the `predict_with_webpage_descriptions` transformers import (spec deviation #5). Produces:
  - `describe_webpage(llm, *, saved_file='', content=None, content_file='', remove_hidden=False, repeat=1, desc_limit=150, add_tab_title=True, resume=True) -> list[str]`
  - `describe_predict(task_id, *, before_file='', action_str='', llm, exp_dir='', remove_hidden=False, content_dict=None, repeat=1, desc_limit=150, add_tab_title=True, resume=True) -> list[str]`. The navigation prediction (source called `predict_with_webpage_descriptions`) is reimplemented inline using `llm.query` with `prompts.PREDICT_WITH_DESCS` — same OpenAI-compatible client, no local transformers. Uses `prompts.DESCRIBING_PROMPT`.
  - Replace `llm.query_LLM(...)` → `llm.query(...)` throughout.
- `stages/reject.py` — ported from `reject.py`. Produces:
  - `is_navigation_change(*, num_deleted_lines, num_added_lines, len_before, len_after, variation, first_line_before, first_line_after, config, llm) -> bool` — the named predicate for the `DESC_PREDICTION_THRESHOLD`/`DIFF_LIMIT`/token-count/`content_before[0] != content_after[0]` branch (`reject.py:75-78`). Uses `llm.num_tokens` and `config.*`.
  - `reject_sample(*, task_id, action_str, llm, exp_dir, content_before=None, content_after=None, diff_info=None, config, resume=False) -> list | None` — returns the parsed score list, or `None` when the page is unchanged (zero diff lines). Uses `parse_scores(..., repeat=config.reject_repeat, max_score=config.reject_max_score)`, `prompts.make_reject_prompt(config.reject_max_score)`, and `describe.describe_webpage` for navigation cases. Writes the same `{task_id}_reject.txt` / `{task_id}_diff.txt` side-effect files as source.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_reject_stage.py
from autogui_anno.config import PipelineConfig
from autogui_anno.stages.reject import reject_sample

class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps
    def num_tokens(self, s): return len(s.split())

def test_reject_sample_zero_diff_returns_none(tmp_path):
    cfg = PipelineConfig()
    out = reject_sample(
        task_id=1, action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["<score>1"]), exp_dir=str(tmp_path),
        content_before=["RootWebArea 'x'"], content_after=["RootWebArea 'x'"],
        diff_info={"diff": [], "num_diff_lines": 0, "num_added_lines": 0, "num_deleted_lines": 0},
        config=cfg, resume=False,
    )
    assert out is None

def test_reject_sample_scores_valid_diff(tmp_path):
    cfg = PipelineConfig()
    diff_info = {"diff": ["Added link 'New'"], "num_diff_lines": 5,
                 "num_added_lines": 1, "num_deleted_lines": 1}
    out = reject_sample(
        task_id=2, action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["reasoning <score>1"]), exp_dir=str(tmp_path),
        content_before=["RootWebArea 'x'", "a", "b"],
        content_after=["RootWebArea 'x'", "a", "c"],
        diff_info=diff_info, config=cfg, resume=False,
    )
    assert out == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_reject_stage.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Port `describe.py` (dropping torch/transformers; inline nav prediction with `PREDICT_WITH_DESCS`). Port `reject.py`: extract the §75-78 branch into `is_navigation_change`, extract scoring to `parse_scores` (already in Task 6), thread `config`. Preserve the side-effect file writes. Covers Review Focus #3 (zero-diff → `None`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_reject_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/stages/describe.py autogui_anno/autogui_anno/stages/reject.py autogui_anno/tests/test_reject_stage.py
git commit -m "feat(autogui_anno): reject + describe stages (drop CUDA/transformers)"
```

---

## Task 11: Annotate stage (`stages/annotate.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/stages/annotate.py`
- Test: `autogui_anno/tests/test_annotate_stage.py`

**Interfaces:**
- Consumes: `prompts`, `describe`, `axtree.diff.format_diff`, `axtree.diff.num_tokens_from_string`, `config`.
- Produces:
  - `predict_with_diff(*, task_id, variation, action_str, llm, exp_dir, resume=True) -> list[str]` — ported from `one_stage_difflib.py:32-60`; the `while True: … continue` retry-until-`SUMMARY_MARK` becomes a bounded loop (`for _ in range(max_retries)`) that raises `RuntimeError` if `SUMMARY_MARK` never appears (spec deviation #3 applied to the annotate loop). Uses `prompts.PREDICT_PROMPT_AXTREE`.
  - `predict_functionality(*, task_id, content_before, content_after, diff_info=None, action_str, llm, exp_res_dir, config, repeat=1, resume=True, layout='axtree') -> tuple[list|None, bool, bool, bool]` — ported from `predict_func` (`one_stage_difflib.py:62-124`), returning `(predictions, is_valid, no_change, is_nav)`. Navigation branch delegates to `describe.describe_predict`; manipulation branch to `predict_with_diff`. Threads `config.diff_limit`, `config.desc_page_limit`, `config.desc_prediction_threshold`, `config.diff_token_limit`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_annotate_stage.py
from autogui_anno.config import PipelineConfig
from autogui_anno.stages.annotate import predict_functionality

class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps
    def num_tokens(self, s): return len(s.split())

def test_predict_functionality_no_change(tmp_path):
    cfg = PipelineConfig()
    preds, is_valid, no_change, is_nav = predict_functionality(
        task_id=1, content_before=["RootWebArea 'x'"],
        content_after=["RootWebArea 'x'"],
        diff_info={"diff": [], "num_diff_lines": 0, "num_added_lines": 0, "num_deleted_lines": 0},
        action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["Reasoning: ... Summary: This element ..."]),
        exp_res_dir=str(tmp_path), config=cfg, resume=False,
    )
    assert preds is None and no_change is True and is_valid is False

def test_predict_functionality_manipulation(tmp_path):
    cfg = PipelineConfig()
    diff_info = {"diff": ["Added link 'New'"], "num_diff_lines": 5,
                 "num_added_lines": 1, "num_deleted_lines": 1}
    preds, is_valid, no_change, is_nav = predict_functionality(
        task_id=2, content_before=["RootWebArea 'x'", "a", "b"],
        content_after=["RootWebArea 'x'", "a", "c"], diff_info=diff_info,
        action_str="clicking a <button> named \"X\"",
        llm=FakeLLM(["Reasoning: foo Summary: This element adds a link."]),
        exp_res_dir=str(tmp_path), config=cfg, resume=False,
    )
    assert is_valid and not no_change and not is_nav
    assert preds and "Summary:" in preds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_annotate_stage.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Port `predict_with_diff` + `predict_func` → `predict_functionality`, replacing module globals with `config.*`, `query_LLM` → `query`, the unbounded retry with a bounded one, and the `LAYOUT`/`PROMPT` module selection with the `layout` param (default `'axtree'` → `PREDICT_PROMPT_AXTREE`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_annotate_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/stages/annotate.py autogui_anno/tests/test_annotate_stage.py
git commit -m "feat(autogui_anno): annotate stage, bounded retry"
```

---

## Task 12: Verify stage (`stages/verify.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/stages/verify.py`
- Test: `autogui_anno/tests/test_verify_stage.py`

**Interfaces:**
- Consumes: `prompts.make_verif_prompt`, `prompts.get_clean_func`, `prompts.REASONING_MARK/SUMMARY_MARK/DESCRIPTION_MARK`, `axtree.prune.count_tab`, `stages.scoring.parse_verification_scores`, `config`.
- Produces:
  - `find_target_line(before_content, elem_text) -> int | None` — ported verbatim from `cycle_consis_checking.py:20-28`.
  - `verify_consistency(*, before_content, func_pred_content, elem_text, llm, result_file, elem_line_id=None, config, resume=True) -> tuple` — ported from `check_cycle_consistency_score` (`cycle_consis_checking.py:133-264`), returning `(resps, scores, elem_line_id, candidate, is_consistent)`. **Fixes (spec deviations #1, #2):** the error-feedback `MAX_SCORE` → `config.verify_max_score` (now via `parse_verification_scores`); `is_consistent = final_score == config.verify_max_score`. The resume short-circuit (`sum(scores)/len(scores) == 3`) uses `config.verify_max_score` instead of the literal `3`. The unbounded `while True` scoring-retry becomes bounded (`config.cycle_check_repeat`-aware, raises on exhaustion). Uses `config.cycle_check_line_limit` for the window.
  - `verify_consistency_multi(*, verifiers: list, **kwargs) -> tuple` — runs `verify_consistency` with each `LLMClient` in `verifiers` and majority-votes `is_consistent` (web passes a 1-element list; mobile passes 3). Returns the aggregated tuple.
  - **Dropped:** the dead `check_cycle_consistency` grounding variant (spec §4).

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_verify_stage.py
from autogui_anno.config import PipelineConfig
from autogui_anno.stages.verify import find_target_line, verify_consistency

def test_find_target_line_locates_clickable():
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    assert find_target_line(before, "Save") == 1

def test_find_target_line_absent_returns_none():
    assert find_target_line(["RootWebArea 'x'"], "Nope") is None

class FakeLLM:
    def __init__(self, resps): self._resps = resps; self.query_count = 0
    def query(self, messages, **kw): self.query_count += 1; return self._resps

def test_verify_consistency_full_score_is_consistent(tmp_path):
    cfg = PipelineConfig()  # verify_max_score=3, cycle_check_repeat=3
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    resps, scores, line_id, cand, is_consistent = verify_consistency(
        before_content=before,
        func_pred_content="Reasoning: foo\n\nSummary: This element saves the file.",
        elem_text="Save", llm=FakeLLM(["<score>3", "<score>3", "<score>3"]),
        result_file=str(tmp_path / "1_cycle.txt"), elem_line_id=1,
        config=cfg, resume=False,
    )
    assert is_consistent is True

def test_verify_consistency_mixed_score_not_consistent(tmp_path):
    cfg = PipelineConfig()
    before = ["RootWebArea 'x'", "button 'Save'", "StaticText 'hi'"]
    resps, scores, line_id, cand, is_consistent = verify_consistency(
        before_content=before,
        func_pred_content="Reasoning: foo\n\nSummary: This element saves the file.",
        elem_text="Save", llm=FakeLLM(["<score>3", "<score>1", "<score>2"]),
        result_file=str(tmp_path / "2_cycle.txt"), elem_line_id=1,
        config=cfg, resume=False,
    )
    assert is_consistent is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_verify_stage.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Port `find_target_line` + `check_cycle_consistency_score` → `verify_consistency`, applying the three fixes. Use `parse_verification_scores` for the score tally + feedback. Compute `final_score = sum(i*scores[i] for i in range(len(scores))) / sum(scores)` (source strict formula), `is_consistent = final_score == config.verify_max_score`. Wrap in `verify_consistency_multi`. Skip the dead grounding variant.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_verify_stage.py -v`
Expected: PASS (covers spec deviations #1, #2)

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/stages/verify.py autogui_anno/tests/test_verify_stage.py
git commit -m "feat(autogui_anno): verify stage, fix MAX_SCORE NameError + score scale"
```

---

## Task 13: Web orchestrator + stats + checkpoints (`pipeline.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/pipeline.py`
- Test: `autogui_anno/tests/test_pipeline_dryrun.py`
- Create: `autogui_anno/tests/fixtures/traj/` (one tiny fixture trajectory: 2 meta.json + 2 axtree.txt files forming one valid step pair; see Step 3)

**Interfaces:**
- Consumes: all stages, `axtree.web.process_axtree`, `axtree.prune.prune_static_text`, `axtree.diff.format_diff`, `config`.
- Produces:
  - `aggregate_stats(stats_list: dict, consis_dict: dict, *, do_rejecting: bool, do_cycle_checking: bool, start_time: float) -> dict` — the near-identical web/mobile summary math from `annotate_func.py:309-397` (minus the matplotlib plot, which moves to Task 16). Returns the `basic_stats` dict.
  - `_load_checkpoint(result_dir) -> dict | None` and `_save_checkpoint(result_dir, stats, cycle_results) -> None` — centralize the inline resume try/except from `annotate_func.py:64-85, 302-307`.
  - `run_web(config: PipelineConfig, llm: LLMClient, verifiers: list, *, data_dir: str, out_dir: str, resume: bool = False, do_rejecting: bool = True, do_cycle_checking: bool = True) -> dict` — ported from `annotate_func`, returning the overall stats dict (and writing `overall_stats.json` + per-traj files). Element location via markers; records `uncheckable` reasons (`Invalid AXTree`, `Incontinuous steps`, `Blank page`, `Target not in AXTree`, `Broken marker`, `Page not changed`). Replaces `llm.token_num[0/1]`/`query_cnt` with `llm.prompt_tokens`/`completion_tokens`/`query_count`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_pipeline_dryrun.py
import os, json
from autogui_anno.config import PipelineConfig
from autogui_anno.pipeline import run_web

class FakeLLM:
    """Canned reject/predict/verify responses; counts queries."""
    def __init__(self):
        self._prompt_tokens = 0; self._completion_tokens = 0; self._query_count = 0
    @property
    def prompt_tokens(self): return self._prompt_tokens
    @property
    def completion_tokens(self): return self._completion_tokens
    @property
    def query_count(self): return self._query_count
    def num_tokens(self, s): return len(s.split())
    def query(self, messages, **kw):
        self._query_count += 1
        text = messages[0]["content"].lower()
        if "score" in text and "candidate" in text:  # verify
            return ["<score>3"] * kw.get("repeat", 1)
        if "sufficient" in text or "predicting" in text:  # reject
            return ["<score>1"] * kw.get("repeat", 1)
        return ["Reasoning: foo\n\nSummary: This element adds a link."] * kw.get("repeat", 1)

def test_run_web_dryrun_produces_stats(tmp_path):
    cfg = PipelineConfig()
    data_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    out_dir = str(tmp_path / "out")
    llm = FakeLLM()
    stats = run_web(cfg, llm, [llm], data_dir=data_dir, out_dir=out_dir, resume=False)
    assert "basic_stats" in stats
    assert os.path.exists(os.path.join(out_dir, "overall_stats.json"))
    assert llm.query_count > 0
```

(Adjust the FakeLLM branch keys to match the actual first-line text of each prompt once the prompts are in place; the test asserts the orchestrator runs end-to-end and emits stats, which is the Review Focus #2/#3 coverage: the fixture includes one step whose marker IS present so it flows through all stages.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_pipeline_dryrun.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Port `annotate_func` → `run_web`, threading `config`/`data_dir`/`out_dir`, centralizing checkpoints, factoring `aggregate_stats`, removing the matplotlib block (Task 16), translating Chinese comments to English. Build the fixture trajectory: two `*_meta.json` (node lists with a `hint_marker_text` matching the action marker encoded in the second file's name, each node having `axtree_node.role.value`/`name.value` and a `rect`) and two `*_axtree.txt` whose pruned+diffed pair yields a non-empty diff. The second file name encodes `_action<marker>_` and `step001`. This exercises Review Focus #2 (marker present path) and #3 (reject returning scores).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_pipeline_dryrun.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/pipeline.py autogui_anno/tests/test_pipeline_dryrun.py autogui_anno/tests/fixtures/traj/
git commit -m "feat(autogui_anno): web orchestrator, checkpoints, stats, mock dry-run"
```

---

## Task 14: Mobile orchestrator + mobile gates (`pipeline.py` additions)

**Files:**
- Modify: `autogui_anno/autogui_anno/pipeline.py` (add `run_mobile`)
- Create: `autogui_anno/autogui_anno/stages/mobile_gates.py`
- Test: `autogui_anno/tests/test_mobile_gates.py`

**Interfaces:**
- `stages/mobile_gates.py` (pure, from `tools.py` + `annotate_func_android.py`):
  - `is_pure_color(image, roi, *, threshold: int = 5) -> bool` — the single canonical version (source has two; keep the `(x1,y1,x2,y2)` roi one at `tools.py:683-711`, drop the duplicate). numpy-based.
  - `is_element_too_large(box, screen_w, screen_h, *, ratio: float = 0.65) -> bool` — the element-area ratio gate from the mobile orchestrator.
  - `is_tap_action(action_type: str) -> bool` — `action_type == 'DualPoint'` check.
- `run_mobile(config, llm, verifiers, *, data_dir, out_dir, resume=False, ...) -> dict` — ported from `annotate_func_android.py`; XML via `axtree.android.process_xml`, target-box matching for element location (vs. web markers), the three mobile gates applied before reject, multi-LLM verify via `verify_consistency_multi`. Shares `aggregate_stats`/checkpoints with `run_web`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_mobile_gates.py
import numpy as np
from autogui_anno.stages.mobile_gates import is_pure_color, is_element_too_large, is_tap_action

def test_is_pure_color_true_on_flat_patch():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    assert is_pure_color(img, (0, 0, 10, 10)) is True

def test_is_pure_color_false_on_noisy_patch():
    img = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
    assert is_pure_color(img, (0, 0, 19, 19)) is False

def test_is_element_too_large():
    assert is_element_too_large([0, 0, 100, 100], 100, 100, ratio=0.65) is True
    assert is_element_too_large([0, 0, 10, 10], 100, 100, ratio=0.65) is False

def test_is_tap_action():
    assert is_tap_action("DualPoint") is True
    assert is_tap_action("Type") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_mobile_gates.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Write `mobile_gates.py` with the three pure gates (port `is_pure_color` verbatim, derive the two small gates from the mobile orchestrator's inline checks). First `Read` `annotate_func_android.py` for exact gate thresholds/branches, then port `run_mobile` into `pipeline.py` reusing shared helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_mobile_gates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/pipeline.py autogui_anno/autogui_anno/stages/mobile_gates.py autogui_anno/tests/test_mobile_gates.py
git commit -m "feat(autogui_anno): mobile orchestrator + pure invalidity gates"
```

---

## Task 15: Task generation (`tasks/generate.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/tasks/generate.py`
- Test: `autogui_anno/tests/test_generate_tasks.py`

**Interfaces:**
- Consumes: nothing from this package (uses `datasets`, `cv2`, `PIL`).
- Produces:
  - `normalize_coord(value: int, full: int) -> str` — `f"{int(value / full * 1000):03d}"` (the `[0,999]` normalization from `generate_ft_data.py:91`).
  - `build_grounding_sample(*, func: str, rect: dict, scale: int, full_w: int, full_h: int, image_rel_path: str, step_id: int) -> dict` — the per-element sample builder producing the `{id, image, bbox, center, downsize, conversations:[human,gpt]}` dict with task template `"Please locate the element supporting this functionality: {func}."` and `<point>[x,y]</point>` output (`generate_ft_data.py:60-120`). **Drops** the debug `cv2.rectangle`/`imwrite("bbox.png")`.
  - `generate_tasks(*, gts_path, traj_folder, image_out_dir, meta_out_dir, full_w: int = 2560, full_h: int = 1440, scales=(1,2,4), push_to_hub: str | None = None) -> list[dict]` — the top-level driver (from the module body), with hardcoded paths → args, `FULLW/FULLH` → args, and `push_to_hub` **opt-in** (only pushes when `push_to_hub` is a repo id; spec deviation #6). **Drops** the unused `FuncPredDataset` class.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_generate_tasks.py
from autogui_anno.tasks.generate import normalize_coord, build_grounding_sample

def test_normalize_coord_zero_padded():
    assert normalize_coord(1280, 2560) == "500"
    assert normalize_coord(0, 2560) == "000"

def test_build_grounding_sample_shape():
    rect = {"top": 100, "left": 200, "width": 50, "height": 40}
    s = build_grounding_sample(func="opens the menu", rect=rect, scale=1,
                               full_w=2560, full_h=1440,
                               image_rel_path="ui/img.png", step_id=5)
    assert s["conversations"][0]["value"].startswith("Please locate the element supporting this functionality:")
    assert s["conversations"][1]["value"].startswith("<point>[")
    assert s["bbox"].count(",") == 3
    assert s["image"] == "ui/img.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_generate_tasks.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Port `generate_ft_data.py` into functions: `normalize_coord`, `build_grounding_sample`, `generate_tasks`. Make `build_grounding_sample` compute `target_bbox`/`target_point` exactly as source (same `int(coord/FULL*1000):03d`), using the task template verbatim and `.rstrip('.')` on the func then appending `"."`. Keep image resize in `generate_tasks` (needs cv2) but gate `push_to_hub`. Drop `FuncPredDataset` and the debug imwrite.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_generate_tasks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/tasks/generate.py autogui_anno/tests/test_generate_tasks.py
git commit -m "feat(autogui_anno): task generation, opt-in push_to_hub"
```

---

## Task 16: CLI entrypoints + optional analysis (`cli.py`, `analysis.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/cli.py`
- Create: `autogui_anno/autogui_anno/analysis.py`
- Test: `autogui_anno/tests/test_cli.py`

**Interfaces:**
- `cli.py`:
  - `main_web(argv=None) -> int`, `main_mobile(argv=None) -> int`, `main_generate_tasks(argv=None) -> int` — argparse entrypoints matching spec §6:
    - web/mobile: `--config pipeline.yaml --models models.yaml --data DIR --out DIR [--resume] [--model NAME] [--verifiers NAME...]`
    - generate: `--gts PATH --traj DIR --images DIR --meta-out DIR [--full-w 2560] [--full-h 1440] [--scales 1 2 4] [--push-to-hub REPO_ID]`
  - Each resolves config → `load_registry` → builds `LLMClient`(s) (default model = registry `default`; verifiers default to `[default]`) → calls the orchestrator. A `--build-client` injection hook (default real) lets the test avoid network.
- `analysis.py`:
  - `plot_reject_performance(all_rejection_scores: dict, all_cycle_checking_results: dict, consis_summary: dict, out_path: str) -> None` — the matplotlib reject-perf curve lifted from `annotate_func.py:338-382`. Imported lazily; never called by the orchestrator.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_cli.py
from autogui_anno import cli

def test_main_web_parses_and_dispatches(tmp_path, monkeypatch):
    # minimal config + models files
    pipe = tmp_path / "pipe.yaml"; pipe.write_text("diff_limit: 250\n")
    models = tmp_path / "models.yaml"
    models.write_text(
        "default: default-fast\nmodels:\n  default-fast:\n"
        "    provider: openai\n    base_url: https://x/v1\n"
        "    api_key_env: FAKE_KEY\n    model: gpt-5-mini\n"
    )
    monkeypatch.setenv("FAKE_KEY", "k")
    called = {}
    def fake_run_web(config, llm, verifiers, **kw):
        called.update(kw); called["ok"] = True; return {"basic_stats": {}}
    monkeypatch.setattr(cli, "run_web", fake_run_web)
    # inject a dummy client builder to avoid real OpenAI construction
    rc = cli.main_web([
        "--config", str(pipe), "--models", str(models),
        "--data", str(tmp_path), "--out", str(tmp_path / "o"),
        "--build-client", "dummy",
    ])
    assert rc == 0 and called.get("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Write `cli.py` with the three argparse mains. Support `--build-client dummy` → a trivial object with `query`/token props (so the CLI wiring is testable without network); default builds a real `LLMClient`. `main_web`/`main_mobile` import `run_web`/`run_mobile` at module scope (so `monkeypatch.setattr(cli, "run_web", ...)` works). Write `analysis.py` with the lazy-matplotlib plot.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/cli.py autogui_anno/autogui_anno/analysis.py autogui_anno/tests/test_cli.py
git commit -m "feat(autogui_anno): CLI entrypoints + optional reject-perf analysis"
```

---

## Task 17: README, requirements, full offline test run

**Files:**
- Modify: `autogui_anno/README.md` (full usage)
- Modify: repo-root `requirements.txt` (add the §9 deps not already present)
- Test: run the whole suite.

**Interfaces:**
- Consumes: everything.
- Produces: documentation only; no new code interfaces.

- [ ] **Step 1: Write the failing check (doc presence)**

```python
# autogui_anno/tests/test_readme.py
import os
def test_readme_mentions_stages_and_env_var():
    p = os.path.join(os.path.dirname(__file__), "..", "README.md")
    txt = open(p).read()
    for stage in ["reject", "annotate", "verify", "generate"]:
        assert stage in txt.lower()
    assert "AUTOGUI_LLM_API_KEY" in txt
    assert "models.example.yaml" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_readme.py -v`
Expected: FAIL (README stub lacks these).

- [ ] **Step 3: Write minimal implementation**

Expand `README.md`: install (`pip install -e autogui_anno`, `python -m spacy download en_core_web_sm`, `playwright install` for web), the five-stage mapping to paper §3.2–3.5, the three CLI commands with example invocations, the `models.yaml` copy-from-example + env-var step (`export AUTOGUI_LLM_API_KEY=...`), and the opt-in `--push-to-hub`. Add missing deps to repo-root `requirements.txt` (`pyyaml` if absent, etc.; avoid pinning the three opencv variants per the known `opencv-three-pin-conflict` issue — add only `opencv-python` if not already present).

- [ ] **Step 4: Run the full offline suite**

Run: `cd autogui_anno && python -m pytest tests/ -v`
Expected: ALL PASS (no network; every test uses fakes/fixtures).

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/README.md autogui_anno/tests/test_readme.py requirements.txt
git commit -m "docs(autogui_anno): README, requirements, full offline suite green"
```

---

## Task 18: Live smoke test (manual, end-of-build)

**Files:**
- Create: `autogui_anno/scripts/smoke_test.py` (committed; reads git-ignored `models.yaml`)

**Interfaces:**
- Consumes: `run_web`, `load_registry`, `LLMClient`, `PipelineConfig`.
- Produces: a runnable script, not a pytest test (it needs the live endpoint + a real trajectory).

- [ ] **Step 1: Write the smoke script**

`scripts/smoke_test.py`: load `configs/models.yaml` (git-ignored, user-provided, pointing at the real OpenAI-compatible endpoint via `AUTOGUI_LLM_API_KEY`, model `gpt-5-mini`), build `LLMClient` for the default model + one verifier (`claude-haiku-4-5-20251001` if present, else default), run `run_web` on one real trajectory dir passed as `argv[1]`, print the resulting functionality prediction + verification score for one step. Guard with a clear message if `models.yaml` or the env var is missing.

- [ ] **Step 2: Run it against the live endpoint**

Run (user-side, after `source ~/.zshrc` for the key and creating `configs/models.yaml`):
`cd autogui_anno && python scripts/smoke_test.py <path-to-one-real-trajectory>`
Expected: prints a non-empty functionality string and a verification score in `[0, 3]`; no secrets printed.

- [ ] **Step 3: Commit**

```bash
git add autogui_anno/scripts/smoke_test.py
git commit -m "test(autogui_anno): live smoke-test script (reads git-ignored models.yaml)"
```

---

## Task 19: Visualizer record builder (`viz/build.py`)

**Files:**
- Create: `autogui_anno/autogui_anno/viz/__init__.py`
- Create: `autogui_anno/autogui_anno/viz/build.py`
- Test: `autogui_anno/tests/test_viz_build.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `LLMClient`, `axtree.web.process_axtree` (or raw before/after line lists), `axtree.prune.prune_static_text`, `axtree.diff.format_diff`, `stages.reject.reject_sample`, `stages.annotate.predict_functionality`, `stages.verify.verify_consistency`.
- Produces:
  - `STAGE_EXPLAINERS: dict[str, str]` — a plain-English one-paragraph blurb per stage key (`input`, `diff`, `reject`, `annotate`, `verify`, `result`), baked into every record so the viewer needs no stage knowledge.
  - `build_sample_record(*, meta: dict, before: list, after: list, diff_result, reject_result, annotate_result, verify_result, config: PipelineConfig) -> dict` — **pure** assembly (no LLM, no I/O): takes already-computed stage results and returns the complete record dict. Every stage key is always present; a stage that did not run (early exit) has its value set to `None` and the record's top-level `verdict` reflects where the pipeline stopped (`"no_change"`, `"uncheckable"`, `"rejected"`, `"inconsistent"`, `"kept"`). Record schema (keys the viewer reads, Task 20):
    ```
    {
      "schema_version": 1,
      "dataset": str, "sample_id": int, "label": str,
      "meta": {"action_str": str, "elem_type": str, "elem_text": str,
               "elem_id": int, "target_line": str | None, "target_line_idx": int | None},
      "input": {"before": [str], "after": [str], "explainer": str},
      "diff": {"lines": [str], "num_added": int, "num_deleted": int, "explainer": str} | null,
      "reject": {"scores": [int], "max_score": int, "kept": bool,
                 "reasoning": str, "explainer": str} | null,
      "annotate": {"functionality": str | None, "mode": "diff" | "describe",
                   "reasoning": str, "explainer": str} | null,
      "verify": {"scores": [int], "final_score": float, "max_score": int,
                 "consistent": bool, "candidate": str | None,
                 "reasoning": str, "explainer": str} | null,
      "result": {"functionality": str | None, "ground_truth": str | None,
                 "has_ground_truth": bool, "explainer": str},
      "verdict": str
    }
    ```
  - `run_builder(*, data_dir: str, out_dir: str, config: PipelineConfig, llm: LLMClient, verifiers: list, dataset_filter=None, resume: bool = True) -> dict` — the **impure** driver: discovers `Mind2Web_*` sample dirs + their sibling `<name>.json` metadata under `data_dir`, iterates samples, runs the real stages (reject → annotate → verify, honoring early exits), calls `build_sample_record`, writes `out_dir/<dataset>_<sample_id>.json`, and maintains `out_dir/index.json` (`[{dataset, sample_id, label, verdict, file}]`). Resumable: skips a sample whose output file already exists. Returns a summary dict (`{built, skipped, by_verdict}`).

The sample layout (confirmed on disk at `WebpageFunctionality/test_samples/`): each `Mind2Web_*` directory holds `<n>_before.txt` / `<n>_after.txt` AXTree line files; the sibling `Mind2Web_*.json` is a list of records with keys `sample_id, task_id, step, type, id, text, action, action_str, snapshot, gt` (where `gt` is the ground-truth functionality, present on a subset — 42 of 73 in `Mind2Web_v1`). The builder joins `<n>_{before,after}.txt` to the JSON entry with `sample_id == n`.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_viz_build.py
from autogui_anno.config import PipelineConfig
from autogui_anno.viz.build import build_sample_record, STAGE_EXPLAINERS

CFG = PipelineConfig()
META = {"action_str": "clicking a <svg> element", "elem_type": "svg",
        "elem_text": "", "elem_id": 9999, "target_line": "[9] button 'x'",
        "target_line_idx": 5}
BEFORE = ["[1] RootWebArea 'x'", "[9] button 'x'"]
AFTER = ["[1] RootWebArea 'x'", "[9] button 'x'", "[30] menu 'opts'"]

def test_explainers_cover_all_stages():
    assert set(STAGE_EXPLAINERS) == {"input", "diff", "reject", "annotate", "verify", "result"}
    assert all(isinstance(v, str) and v for v in STAGE_EXPLAINERS.values())

def test_record_full_pipeline_kept():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=AFTER,
        diff_result={"lines": ["Added [30] menu 'opts'"], "num_added": 1, "num_deleted": 0},
        reject_result={"scores": [1], "max_score": 1, "kept": True, "reasoning": "r"},
        annotate_result={"functionality": "Opens the options menu.", "mode": "diff", "reasoning": "a"},
        verify_result={"scores": [0, 0, 0, 3], "final_score": 3.0, "max_score": 3,
                       "consistent": True, "candidate": "[9] button 'x'", "reasoning": "v"},
        config=CFG,
    )
    assert rec["schema_version"] == 1
    assert rec["verdict"] == "kept"
    assert rec["result"]["functionality"] == "Opens the options menu."
    assert rec["input"]["explainer"] == STAGE_EXPLAINERS["input"]
    for k in ["input", "diff", "reject", "annotate", "verify", "result"]:
        assert k in rec

def test_record_early_exit_no_change_has_null_later_stages():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=BEFORE,
        diff_result={"lines": [], "num_added": 0, "num_deleted": 0},
        reject_result=None, annotate_result=None, verify_result=None, config=CFG,
    )
    assert rec["verdict"] == "no_change"
    assert rec["reject"] is None and rec["annotate"] is None and rec["verify"] is None
    assert rec["result"]["functionality"] is None
    assert rec["result"]["has_ground_truth"] is False

def test_record_rejected_stops_before_annotate():
    rec = build_sample_record(
        meta=META, before=BEFORE, after=AFTER,
        diff_result={"lines": ["Added [30] menu 'opts'"], "num_added": 1, "num_deleted": 0},
        reject_result={"scores": [0], "max_score": 1, "kept": False, "reasoning": "r"},
        annotate_result=None, verify_result=None, config=CFG,
    )
    assert rec["verdict"] == "rejected"
    assert rec["reject"]["kept"] is False
    assert rec["annotate"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_viz_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autogui_anno.viz.build'`

- [ ] **Step 3: Write minimal implementation**

Write `viz/build.py`:
- `STAGE_EXPLAINERS`: a one-paragraph plain-English blurb per stage, aimed at a non-expert (e.g. input: "The pipeline starts from two accessibility trees — the page before the interaction and the page after. Each describes every on-screen element as structured text."; diff/reject/annotate/verify/result similarly).
- `build_sample_record` — **pure**: assemble the schema dict from the passed-in stage results; derive `verdict` (`no_change` if diff empty; `uncheckable` if `meta["target_line"] is None`; `rejected` if `reject_result and not reject_result["kept"]`; `inconsistent` if `verify_result and not verify_result["consistent"]`; else `kept`); set each stage value to `None` when its result arg is `None`; set `result.has_ground_truth` from `bool(meta.get("gt"))` and `result.ground_truth` from `meta.get("gt")`; attach `STAGE_EXPLAINERS[k]` to each present stage; `label = f"#{sample_id} {elem_type}"`.
- `run_builder` — impure driver: glob `data_dir/Mind2Web_*/`, load sibling JSON, join by `sample_id`, read before/after files, compute diff via `format_diff`, run `reject_sample`/`predict_functionality`/`verify_consistency` with early-exit short-circuits, call `build_sample_record`, write per-sample files + `index.json`, skip existing on resume.

Covers Review Focus #6 (early-exit records are complete and null-filled, never half-written).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_viz_build.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/viz/__init__.py autogui_anno/autogui_anno/viz/build.py autogui_anno/tests/test_viz_build.py
git commit -m "feat(autogui_anno): visualizer record builder (pure assembly + driver)"
```

---

## Task 20: Static viewer page (`viz/web/`)

**Files:**
- Create: `autogui_anno/autogui_anno/viz/web/index.html`
- Create: `autogui_anno/autogui_anno/viz/web/app.js`
- Create: `autogui_anno/autogui_anno/viz/web/styles.css`
- Create: `autogui_anno/autogui_anno/viz/web/viz_data/example_record.json` (committed example)
- Create: `autogui_anno/autogui_anno/viz/web/viz_data/index.json` (committed, lists the example record)
- Test: `autogui_anno/tests/test_viz_viewer.py`

**Interfaces:**
- Consumes: record JSON produced by Task 19 (`schema_version: 1`); `index.json` for the left rail.
- Produces: a static single-page app. `app.js` defines `renderRecord(record, mountEl)` and `renderIndex(index, railEl)` on `window`, and on load fetches `viz_data/index.json` then the first record.

**Layout (guided vertical pipeline, per approved design):**
- Top banner: the six-stage flow (`Input → Diff → Reject → Annotate → Verify → Result`) with the current sample's per-stage verdict badge.
- Left rail: sample list from `index.json`, grouped by dataset, each row = `label` + verdict badge; click loads that record.
- Main panel: six stacked stage cards in order. Each card = stage title + `explainer` blurb + stage content:
  - **Input:** before/after trees side by side; the `target_line_idx` line highlighted in both.
  - **Diff:** `diff.lines` with Added (green) / Deleted (red) / Unchanged (muted) coloring; added/deleted counts.
  - **Reject:** score badge `scores`/`max_score`, kept/rejected pill, reasoning text.
  - **Annotate:** predicted functionality (prominent), `mode` tag (diff/describe), reasoning.
  - **Verify:** `final_score`/`max_score`, consistent pill, candidate line, reasoning.
  - **Result:** predicted functionality next to ground truth (or a "No ground-truth annotation for this sample" note when `has_ground_truth` is false).
- A null stage renders a muted "This stage did not run — the pipeline stopped at `<verdict>`." card instead of its content.

Vanilla JS + CSS only; no framework, no build step. Diff coloring is done in `app.js` (no external lib required). The page opens via `python -m http.server --directory autogui_anno/autogui_anno/viz/web`.

- [ ] **Step 1: Write the failing test**

A static-contract check (no JS runtime): assert the assets exist, that `app.js` exposes the two render entry points and references every stage key, and that the committed example record + index validate against the Task-19 schema. A headless-browser test is out of scope.

```python
# autogui_anno/tests/test_viz_viewer.py
import json, os

WEB = os.path.join(os.path.dirname(__file__), "..", "autogui_anno", "viz", "web")

def test_viewer_assets_exist():
    for f in ["index.html", "app.js", "styles.css", "viz_data/index.json",
              "viz_data/example_record.json"]:
        assert os.path.exists(os.path.join(WEB, f)), f

def test_app_js_renders_all_stages():
    src = open(os.path.join(WEB, "app.js")).read()
    assert "renderRecord" in src and "renderIndex" in src
    for stage in ["input", "diff", "reject", "annotate", "verify", "result"]:
        assert stage in src

def test_example_record_matches_schema():
    rec = json.load(open(os.path.join(WEB, "viz_data", "example_record.json")))
    assert rec["schema_version"] == 1
    for k in ["dataset", "sample_id", "label", "meta", "input", "result", "verdict"]:
        assert k in rec
    assert {"before", "after", "explainer"} <= set(rec["input"])

def test_index_lists_example():
    idx = json.load(open(os.path.join(WEB, "viz_data", "index.json")))
    assert isinstance(idx, list) and len(idx) >= 1
    assert {"dataset", "sample_id", "label", "verdict", "file"} <= set(idx[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_viz_viewer.py -v`
Expected: FAIL — assets do not exist.

- [ ] **Step 3: Write minimal implementation**

Write `index.html` (banner + rail + main mount points, loads `styles.css` and `app.js`), `app.js` (`renderIndex`, `renderRecord` handling all six stages + null-stage fallback, fetch-on-load), `styles.css` (two-column layout, stage cards, diff colors, usable down to ~400px width). Generate `example_record.json` by running the Task-19 builder once on a single public `Mind2Web_v1` sample, then hand-trim it small; write `index.json` listing it. **Grep the committed example for any endpoint/host/key string and confirm none is present before adding.**

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_viz_viewer.py -v`
Expected: PASS. Then manually: `python -m http.server --directory autogui_anno/autogui_anno/viz/web` and confirm the example renders all six cards.

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/viz/web/
git commit -m "feat(autogui_anno): static pipeline visualizer page + example record"
```

---

## Task 21: Visualizer CLI (build + serve), gitignore, README

**Files:**
- Modify: `autogui_anno/autogui_anno/cli.py` (add `main_visualize_build`, `main_visualize_serve`)
- Modify: `autogui_anno/pyproject.toml` (add `autogui-visualize-build`, `autogui-visualize-serve` scripts)
- Modify: `.gitignore` (repo root) — ignore `autogui_anno/autogui_anno/viz/web/viz_data/*`, then un-ignore `!…/viz_data/index.json` and `!…/viz_data/example_record.json`
- Modify: `autogui_anno/README.md` (visualizer section)
- Test: `autogui_anno/tests/test_viz_cli.py`

**Interfaces:**
- Consumes: `viz.build.run_builder`, `load_registry`, `LLMClient`, `PipelineConfig`.
- Produces:
  - `main_visualize_build(argv=None) -> int` — argparse: `--config pipeline.yaml --models models.yaml --data DIR --out DIR [--datasets NAME...] [--no-resume] [--model NAME] [--verifiers NAME...] [--build-client dummy]`. Resolves config/registry, builds client(s), calls `run_builder`, prints the summary. Default `--out` = the packaged `viz/web/viz_data`.
  - `main_visualize_serve(argv=None) -> int` — argparse: `[--dir DIR] [--port 8000] [--check-only]`; a thin wrapper over `http.server` serving the viewer dir (default the packaged `viz/web`). `--check-only` validates the dir and returns `0` without binding a socket (for tests); otherwise prints the URL and serves.

- [ ] **Step 1: Write the failing test**

```python
# autogui_anno/tests/test_viz_cli.py
from autogui_anno import cli

def test_main_visualize_build_dispatches(tmp_path, monkeypatch):
    pipe = tmp_path / "pipe.yaml"; pipe.write_text("diff_limit: 250\n")
    models = tmp_path / "models.yaml"
    models.write_text(
        "default: default-fast\nmodels:\n  default-fast:\n"
        "    provider: openai\n    base_url: https://x/v1\n"
        "    api_key_env: FAKE_KEY\n    model: gpt-5-mini\n"
    )
    monkeypatch.setenv("FAKE_KEY", "k")
    called = {}
    def fake_run_builder(**kw):
        called.update(kw); called["ok"] = True
        return {"built": 1, "skipped": 0, "by_verdict": {"kept": 1}}
    monkeypatch.setattr(cli, "run_builder", fake_run_builder)
    rc = cli.main_visualize_build([
        "--config", str(pipe), "--models", str(models),
        "--data", str(tmp_path), "--out", str(tmp_path / "viz"),
        "--build-client", "dummy",
    ])
    assert rc == 0 and called.get("ok")

def test_main_visualize_serve_check_only(tmp_path):
    rc = cli.main_visualize_serve(["--dir", str(tmp_path), "--check-only"])
    assert rc == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autogui_anno && python -m pytest tests/test_viz_cli.py -v`
Expected: FAIL — the two entrypoints do not exist.

- [ ] **Step 3: Write minimal implementation**

Add `main_visualize_build` (imports `run_builder` at module scope so the test can monkeypatch `cli.run_builder`; supports `--build-client dummy`) and `main_visualize_serve` (`--check-only` validates the dir and returns without serving; otherwise starts `http.server`). Register both console scripts in `pyproject.toml`. Add the `.gitignore` lines (ignore `viz_data/*`, un-ignore the two committed files) and confirm `git status` keeps `index.json` + `example_record.json` tracked while dropping freshly built records.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autogui_anno && python -m pytest tests/test_viz_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autogui_anno/autogui_anno/cli.py autogui_anno/pyproject.toml autogui_anno/README.md .gitignore autogui_anno/tests/test_viz_cli.py
git commit -m "feat(autogui_anno): visualizer build/serve CLI + gitignore + README"
```

**README visualizer section (Step 3 content):** document the two commands —
```
# 1. Build records (needs AUTOGUI_LLM_API_KEY + a models.yaml)
autogui-visualize-build --config configs/pipeline.yaml --models configs/models.yaml \
    --data /path/to/test_samples --out autogui_anno/autogui_anno/viz/web/viz_data
# 2. View (no key needed; works offline)
autogui-visualize-serve           # serves the packaged page at http://localhost:8000
```
— and note that bulk annotation is done with `autogui-annotate-web`/`-mobile`, not the visualizer, which is a didactic demo over a handful of samples.

---

## Self-Review

**1. Spec coverage:**
- §2 layout → Tasks 1–16 (every listed module has an owning task). ✓
- §3 LLM client & registry → Tasks 3, 4; `models.example.yaml` → Task 2. ✓
- §4 stages (reject/annotate/describe/verify) + config table + separate score scales → Tasks 2, 6, 10, 11, 12. ✓
- §5 prune/diff/web/android/generate + mobile gates → Tasks 5, 6, 8, 9, 14, 15. ✓
- §6 orchestrator/CLI/config → Tasks 13, 14, 16, 2. ✓
- §7 testing (4 unit suites + mock dry-run + live smoke) → Tasks 5, 6, 8, 12/6, 13, 18. ✓
- §8 security (gitignore, example-only, no secrets) → Task 1 (gitignore), Task 2 (example yaml), Global Constraints. ✓
- §9 dependencies → Tasks 1, 17. ✓
- §10 deviations: #1,#2 → Task 12 & 6; #3 → Tasks 4, 11, 12; #4 → Task 9; #5 → Task 10; #6 → Task 15. ✓
- **Visualizer (user-added feature, beyond the original spec)** → Tasks 19–21: record builder (19), static viewer + example record (20), build/serve CLI + gitignore + README (21). This extends the spec rather than implementing a section of it — the spec predates the visualizer request — so it has no §-number to map to; it is deliberately scoped as a didactic offline demo that reuses the Task 5–12 stage code without changing it. ✓

**2. Placeholder scan:** No "TBD"/"implement later"/"handle edge cases" without code. Three tasks (8, 9, 14) say "`Read` the source first for exact bodies then port verbatim" — this is deliberate (large verbatim ports whose exact text is too long to inline), not a vague placeholder; the signatures and drop-lists are fully specified. Task 20 Step 3 says to generate `example_record.json` by running the Task-19 builder on one sample and hand-trimming it — a build artifact, not a placeholder; its schema is pinned by the Task-20 schema test. ✓

**3. Type consistency:** `PipelineConfig`/`ModelSpec`/`RegistryConfig` fields (Task 2) are consumed with the same names in Tasks 3,4,10,11,12,13,16. `LLMClient.query(...)/.prompt_tokens/.completion_tokens/.query_count/.num_tokens` (Task 4) are the exact surface the fakes implement in Tasks 10–13 and that `run_web` reads in Task 13. `parse_scores`/`parse_verification_scores` signatures (Task 6) match their callers (Tasks 10, 12). `verify_consistency` return tuple is consistent between Task 12 definition and Task 13 usage. `clean_accessibility_tree` spelling is fixed once (Task 9) and never reverts. **Visualizer:** the `schema_version: 1` record keys produced by `build_sample_record` (Task 19) are exactly the keys the viewer's `renderRecord`/`renderIndex` read and the Task-20 `test_example_record_matches_schema`/`test_index_lists_example` assert (`dataset, sample_id, label, meta, input, diff, reject, annotate, verify, result, verdict`; stage sub-keys per the Task-19 schema block). `run_builder(*, data_dir, out_dir, config, llm, verifiers, dataset_filter, resume)` (Task 19) is the exact callable `main_visualize_build` resolves and the Task-21 test monkeypatches as `cli.run_builder`; `main_visualize_build`/`main_visualize_serve` names match pyproject scripts and the Task-21 tests. ✓

**4. Review Focus:** All six lines have an owning task with an explicit test — #1→Task 6 (`test_parse_scores_malformed_skipped`, `_empty_input`), #2→Task 13 (fixture marker-present path + orchestrator continues on missing marker), #3→Tasks 7/10/13 (`test_reject_sample_zero_diff_returns_none`, `test_predict_functionality_no_change`), #4→Task 3 (`test_resolve_api_key_missing_raises_named`), #5→Task 4 (`test_query_raises_after_exhaustion`), #6→Task 19 (`test_record_early_exit_no_change_has_null_later_stages`, `test_record_rejected_stops_before_annotate` — a sample the pipeline rejects/skips still yields a complete, renderable record with null later stages and a verdict reflecting the early exit). ✓
