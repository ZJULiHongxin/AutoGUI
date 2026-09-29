# AutoGUI Functionality-Annotation Pipeline — Design Spec

**Date:** 2026-09-29
**Status:** Approved (design), pending implementation plan
**Author:** Hongxin Li (with Claude Code)

## 1. Purpose

The AutoGUI paper (ACL 2025, `assets/2025.acl-long.510.pdf`) describes an
automatic pipeline for annotating UI-element *functionalities* by comparing
UI state before and after an interaction and asking LLMs to (a) predict the
functionality, (b) reject unpredictable samples, and (c) verify predictions
via cycle-consistency. The AutoGUI repo currently open-sources only the
training/evaluation code (`lmms_eval/`, `finetune/`, `pretrain/`,
`autogui_model/`); the annotation pipeline is described in the README but is
absent as code.

This spec covers porting that pipeline — currently living as tangled
top-level scripts in a separate `WebpageFunctionality` project — into the
AutoGUI repo as a clean, installable subpackage `autogui_anno/`, covering
both the **Web** (accessibility-tree) and **Mobile** (Android XML) paths.

### Intended outcome (design brief)

- A reader of the paper can `pip install` the AutoGUI repo and run the
  functionality-annotation pipeline end-to-end on their own trajectories.
- The pipeline is behavior-preserving relative to the source, but repackaged
  as a proper library: config dataclasses + CLI entrypoints instead of
  module-level hardcoded constants, one pluggable OpenAI-compatible LLM
  client instead of eight hardcoded providers, typed functions, docstrings,
  and no dead code.
- **No secrets, internal hostnames, or private IPs** land in the tracked
  repo. All API keys come from environment variables named in a committed
  `models.example.yaml`; the real `models.yaml` is git-ignored.

### Scope

**In scope:** Web + Mobile *core* pipeline — the five stages plus shared
infrastructure:
1. (Trajectory collection is assumed already done; raw AXTree/XML + meta
   files on disk are the input.)
2. Automatic functionality annotation (diff-mode + describe-mode).
3. LLM-aided rejection (predictability scoring).
4. LLM-based verification ("cycle-consistency" scoring).
5. Task generation → instruction-tuning dataset (grounding + captioning),
   optional HuggingFace `push_to_hub`.

**Out of scope:** subtask annotation, rebuttal scripts, LLM-eval prompt
variants, plotting one-offs, the live Playwright *scraper* (`extract_axtree`),
and the local-transformers describe path. Trajectory *collection* itself is
not ported; the pipeline consumes already-collected trajectories.

## 2. Architecture & Package Layout

The pipeline is a linear, resumable flow mirroring the paper's Figure 2:
**collect (external) → reject → annotate → verify → generate tasks**. The
package separates *pure logic* (testable offline) from *LLM/IO* (needs
network).

```
AutoGUI/
└── autogui_anno/
    ├── README.md                     # pipeline usage, mapped to paper stages
    ├── configs/
    │   ├── models.example.yaml       # model→provider→endpoint registry (committed)
    │   └── pipeline.example.yaml     # thresholds (DIFF_LIMIT, scores, repeats…)
    ├── autogui_anno/
    │   ├── __init__.py
    │   ├── config.py                 # dataclasses: PipelineConfig, ModelSpec, RegistryConfig; YAML loaders
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   ├── client.py             # LLMClient: OpenAI-compatible, key from env, retry, token accounting
    │   │   └── registry.py           # load models.yaml → ModelSpec
    │   ├── axtree/
    │   │   ├── __init__.py
    │   │   ├── prune.py              # prune_static_text, prune_schema_text, get_markers, count_tab (pure)
    │   │   ├── diff.py               # format_diff, unified-diff helpers (pure)
    │   │   ├── web.py                # process_axtree (Playwright accessibility-tree parsing)
    │   │   └── android.py            # process_xml / simplify_tree / tree_to_text
    │   ├── stages/
    │   │   ├── __init__.py
    │   │   ├── reject.py             # LLM-aided rejection (paper §3.3)
    │   │   ├── annotate.py           # functionality annotation (paper §3.2): diff-mode + describe-mode
    │   │   ├── describe.py           # webpage description fallback for navigation cases
    │   │   └── verify.py             # LLM verification / "cycle consistency" (paper §3.4)
    │   ├── tasks/
    │   │   ├── __init__.py
    │   │   └── generate.py           # triples → grounding/captioning instruction data (paper §3.5)
    │   ├── prompts.py                # curated prompt templates (web + android)
    │   ├── pipeline.py               # orchestrator: run_web & run_mobile, resume, stats
    │   ├── analysis.py               # optional matplotlib reject-perf plotting (not in hot path)
    │   └── cli.py                    # console-script entrypoints
    └── tests/
        ├── fixtures/                # small AXTree/XML samples
        ├── test_prune.py
        ├── test_diff.py
        ├── test_score_parsing.py
        ├── test_android_tree.py
        └── test_pipeline_dryrun.py  # mock LLMClient, full orchestrator, no network
```

**Key decisions:**

1. **Pure/impure split.** `axtree/prune.py`, `axtree/diff.py`, the
   score-parsing helpers, `axtree/android.py` tree logic, and
   `tasks/generate.py` coordinate math have zero network/secret dependencies
   → fully unit-testable with fixtures.
2. **One LLM client** replaces the source's eight-provider `model_info_list`
   with literal keys. Every provider (and the target endpoint) is
   OpenAI-compatible, so one `LLMClient` + a YAML registry suffices. Keys
   come from env vars named in the registry.
3. **Config objects replace module-level constants.** The source's
   `dataset_name = "..."`, `DIFF_LIMIT = 250`, `model_name = [...][1]`, etc.
   become `PipelineConfig` fields set via CLI/YAML.
4. **Two orchestrator entrypoints** (`run_web`, `run_mobile`) share stage
   code, differing only in the parser (AXTree vs. XML), the element-location
   strategy (marker vs. target-box), a few extra mobile invalidity gates, and
   a few prompts.

## 3. LLM Client & Model Registry

### `configs/models.example.yaml` (committed — fully generic)

```yaml
# Copy this file to models.yaml (git-ignored) and fill in your own
# OpenAI-compatible endpoint(s). Keys are read from environment variables.
default: default-fast
models:
  default-fast:
    provider: openai              # any OpenAI-compatible server
    base_url: https://your-endpoint.example.com/v1
    api_key_env: AUTOGUI_LLM_API_KEY   # name of the env var holding the key
    model: gpt-5-mini
  default-strong:
    provider: openai
    base_url: https://your-endpoint.example.com/v1
    api_key_env: AUTOGUI_LLM_API_KEY
    model: gpt-5.1
```

### `autogui_anno/llm/registry.py`

Loads the YAML into `ModelSpec` dataclasses; resolves `api_key_env` →
`os.environ[...]` at construction time, raising a clear error if the env var
is unset. Only `models.example.yaml` is committed; `models.yaml` is
git-ignored.

### `autogui_anno/llm/client.py`

One `LLMClient` class wrapping the OpenAI SDK (which every provider here
speaks). It preserves the source's public surface so stage code barely
changes:

- `query(messages, *, temperature, repeat, stop, do_sample) -> list[str]`
  (was `query_LLM`).
- Token accounting exposed as `prompt_tokens`, `completion_tokens`,
  `query_count` properties (replacing the source's `token_num[0]/[1]` list
  and `query_cnt`).
- Bounded retry with backoff on transient errors (default 3 attempts),
  **raising on exhaustion** rather than the source's unbounded
  `while True: … continue` loops that could hang forever.
- `num_tokens(str)` via tiktoken.

**Dropped from source:** (1) the per-provider `QwenModel`/`OpenAIModel`
subclasses — all OpenAI-compatible, so one class; (2) the eight hardcoded
`model_info_list` entries with literal API keys (OpenAI `sk-…`, NVIDIA
`nvapi-…`, Groq `gsk_…`, SiliconFlow, Fireworks, Together, Taobao proxy,
Hyperbolic JWTs) and the private IP `36.111.143.211`.

## 4. Pipeline Stages

Each source script becomes one stage module taking a config object and an
`LLMClient` instead of importing module-level constants. Pure
scoring/parsing logic is extracted into small helpers.

**Stage flow** (per adjacent AXTree/XML pair `(before, after)` in a
trajectory): build unified diff → **reject** (score predictability) →
**annotate** (diff-mode or describe-mode) → **verify** (cycle-consistency
score). A step survives only if reject passes and verify scores full marks.

### `stages/reject.py` (from `reject.py`)

`reject_sample(...) -> RejectResult`. Scores whether the diff is rich enough
to predict functionality; parses `<score>` tags. The inline score-parsing
(`resp.rfind("<score>")`, `eval(...)`, bare `except`) becomes a tested pure
helper `parse_scores(text, max_score) -> list[int]`. The
navigation-vs-manipulation branch (the `DESC_PREDICTION_THRESHOLD` /
`DIFF_LIMIT` / token-count check) becomes a named predicate
`is_navigation_change(diff_stats, config)`.

### `stages/annotate.py` (from `one_stage_difflib.py`)

`predict_functionality(...) -> Prediction`. Both paths preserved:
`predict_with_diff` (manipulation) and delegation to `describe.py`
(navigation). The `while True: … continue` retry-until-`SUMMARY_MARK` becomes
a bounded retry on the client.

### `stages/describe.py` (from `describe.py`)

`describe_webpage(...)` and `describe_predict(...)` for navigation cases.
**Dropped:** the top-level `torch` / `torch.set_default_device("cuda")`
import and the `predict_with_webpage_descriptions` transformers path — those
belong to a local-model variant out of scope for the OpenAI-compatible
pipeline. Navigation prediction uses the same `LLMClient`.

### `stages/verify.py` (from `cycle_consis_checking.py`)

`verify_consistency(...) -> VerifyResult`, the paper's cycle-consistency
check. Keeps `find_target_line` + parent-node context assembly +
`make_verif_prompt` scoring. **Fixes:**

- The `MAX_SCORE` free-variable reference (a latent `NameError` in the
  error-feedback and `is_consistent` branches — the function has no such
  local) → use the passed `verify_max_score` parameter.
- `is_consistent = final_score == MAX_SCORE` → `final_score == verify_max_score`
  (removes the score-scale confusion).
- Extract `parse_verification_scores(...)` as a pure, tested helper.
- Drop the dead `check_cycle_consistency` (grounding variant); the pipeline
  only uses the `_score` variant.

The verifier stage accepts **a list of `LLMClient`s**: web uses one, mobile
can use N (the source's 3 aux LLMs, majority vote), driven by config.

### Config objects

All scattered constants become fields on a `PipelineConfig` dataclass, loaded
from `configs/pipeline.example.yaml`, with the source's current values as
defaults:

| Field | Source value |
|-------|-------------|
| `desc_prediction_threshold` | 0.8 |
| `desc_page_limit` | 150 |
| `diff_limit` | 250 |
| `diff_context` | 4 |
| `menuitem_limit` | 3 |
| `tab_limit` | 10 |
| `reject_repeat` | 3 |
| `reject_temp` | 1.0 |
| `cycle_check_repeat` | 3 |
| `cycle_check_line_limit` | 20 |
| `remove_hidden` | true |
| `with_markers` | true |
| `reject_max_score` | 1 |
| `verify_max_score` | 3 |

Note: rejection and verification use **separate** score scales. The source
used `MAX_SCORE = 1` in `reject.py` and a full score of `3` in
cycle-consistency checking; these become the distinct `reject_max_score` and
`verify_max_score` fields (rather than one conflated `max_score`), so
`parse_scores` and `parse_verification_scores` each receive the correct
ceiling.

## 5. AXTree/XML Processing & Task Generation

### `axtree/prune.py` (pure)

Text-pruning helpers from `tools.py`: `prune_static_text`,
`prune_schema_text`, `get_markers`, `count_tab`, `remove_labels`,
`extract_labels`. No network, no Playwright → unit-tested against fixtures.

### `axtree/diff.py` (pure)

`format_diff` and diff-stats bundling. Unit-tested.

### `axtree/web.py` (from `process_axtree.py`)

The Playwright-dependent web parser: `prune_accessibility_tree_wo_bound`,
`parse_accessibility_tree`, `clean_accessibility_tree` (spelling fixed from
`clean_accesibility_tree`), `process_axtree`. **Dropped:** the async
`extract_axtree` scraper — it has a latent bug (`await`s the non-async
`parse_accessibility_tree`/`clean_accesibility_tree`) and scraping/collection
is out of scope. The commented private IP is removed.

### `axtree/android.py` (from XML half of `process_axtree.py`)

`process_xml`, `simplify_tree`, `tree_to_text`, `parse_xml_to_tree`,
`preprocess_xml_content`, `decode_special_chars`. Pure except file I/O; the
embedded `__main__` XML sample becomes a test fixture.

### `tasks/generate.py` (from `eval/generate_ft_data.py`)

Turns verified `(image, element-box, functionality)` triples into
instruction-tuning samples. **Preserves:** the grounding template
(`"Please locate the element supporting this functionality: …"`), `[0,999]`
coordinate normalization, `<point>`/bbox output format, optional
`push_to_hub`. **Cleanups:** hardcoded `/data0/jingran/…` paths and
`FULLW/FULLH = 2560/1440` become config/CLI args; the debug
`cv2.rectangle` + `imwrite("bbox.png")` and the unused `FuncPredDataset`
class are dropped; `push_to_hub` is opt-in (`--push-to-hub REPO_ID`, off by
default — no accidental uploads).

### Mobile-specific gates

The extra reject-time invalidity gates fold into the mobile orchestrator
path: `is_pure_color(screenshot, box)` (element not displayed),
element-too-large ratio, not-a-tap-action, step-gap. The multi-LLM verifier
is the general "list of verifiers" from §4.

## 6. Orchestrator, CLI & Config

### `pipeline.py`

- `run_web(config, llm, verifiers) -> Stats` (from `annotate_func.py`)
- `run_mobile(config, llm, verifiers) -> Stats` (from
  `annotate_func_android.py`)

Both share the skeleton: iterate trajectories → process state pairs →
per-step gates → reject → annotate → verify → accumulate stats + write
per-trajectory `stats.json` / `cycle_check_result.json`. **Cleanups:**

- Resume logic centralized into `_load_checkpoint` / `_save_checkpoint`
  instead of duplicated inline try/except.
- Matplotlib reject-perf plotting moved to optional `analysis.py`, not run by
  default. The core emits `overall_stats.json` + the ranked rejection list
  (the data); plotting is a separate opt-in call.
- Chinese comments translated to English.
- Stats aggregation (near-identical between web & mobile) factored into
  shared `aggregate_stats(...)`.

### `cli.py` — console-script entrypoints (registered in `pyproject.toml`)

- `autogui-annotate-web --config pipeline.yaml --models models.yaml --data DIR [--resume]`
- `autogui-annotate-mobile …`
- `autogui-generate-tasks --results DIR --images DIR --out DIR [--push-to-hub REPO_ID]`

Each resolves config → builds `LLMClient`(s) from the registry → calls the
orchestrator. No module-level constants, no `[...][index]` selectors.

### `config.py`

Two dataclasses: `PipelineConfig` (thresholds, §4 table) and
`ModelSpec`/`RegistryConfig` (model registry). Both have `from_yaml(path)`
loaders; unknown keys raise clearly.

## 7. Testing

1. **Offline unit tests** (`pytest`, no network):
   - `test_prune.py` — `prune_static_text`, `count_tab`, `get_markers`, label
     removal against fixtures.
   - `test_diff.py` — `format_diff` line counting, diff-limit truncation,
     prefix injection.
   - `test_score_parsing.py` — `parse_scores` / `parse_verification_scores`
     across well-formed, malformed, and empty LLM outputs.
   - `test_android_tree.py` — `process_xml` on the embedded XML sample →
     expected text tree.
2. **Mock-LLM dry run** (`test_pipeline_dryrun.py`) — a `FakeLLMClient`
   returning canned reject/predict/verify responses; runs `run_web`
   end-to-end on one fixture trajectory, asserts stats shape and that a step
   flows through all stages. No network.
3. **Live smoke test** — a small script (git-ignored `models.yaml` pointing
   at the real endpoint via env var, model `gpt-5-mini`) running one real
   trajectory through all stages, printing the resulting functionality +
   verification score. Run once at the end to confirm wiring against a live
   OpenAI-compatible server.

## 8. Security & Secrets

- **Committed:** all package code, `configs/*.example.yaml`, tests,
  `README.md`.
- **Git-ignored:** `models.yaml` (real endpoint/keys). A
  `autogui_anno/configs/models.yaml` entry is added to `.gitignore`.
- No secrets, no internal hostnames, no private IPs anywhere in tracked
  files. The eight literal API keys and the private IP found in the source
  `tools.py` are never copied; keys come from env vars only.

## 9. Dependencies

Added to the repo's `requirements.txt` (most already present): `openai`,
`tiktoken`, `datasets`, `pyyaml`, `numpy`, `pillow`, `opencv-python`,
`playwright` (web parsing only), `tqdm`. `matplotlib` is used only by the
optional `analysis.py`.

## 10. Deviations from Source (behavior-affecting)

These are intentional fixes where the source had latent bugs; behavior is
preserved on the common paths and corrected on the buggy ones:

1. `verify.py`: `MAX_SCORE` free-variable reference → `verify_max_score`
   parameter (avoids `NameError` on the error-feedback and consistency
   branches).
2. `verify.py`: `is_consistent` compares against `verify_max_score`, not the
   undefined module `MAX_SCORE`.
3. LLM client: unbounded `while True` retry loops → bounded retry that raises.
4. `web.py`: dropped the async `extract_axtree` (awaits non-async helpers).
5. `describe.py`: dropped the unconditional CUDA/torch import.
6. `generate.py`: `push_to_hub` is opt-in, not unconditional.

All other behavior (thresholds, prompts, scoring math, diff generation,
coordinate normalization) is preserved exactly.
