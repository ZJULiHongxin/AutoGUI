# autogui_anno

The AutoGUI UI-element functionality annotation pipeline (ACL 2025).

This package turns raw GUI observations into verified, task-grounded
functionality annotations through five stages. Each stage maps to a section of
the paper:

1. **collect** — gather raw GUI screenshots and accessibility trees (§3.1, Data Collection).
2. **reject** — filter out low-quality or unpredictable elements before annotation, using the LLM's predictability score (§3.2, Element Rejection).
3. **annotate** — generate candidate functionality descriptions for UI elements by diffing the before/after accessibility trees and describing the observed effect (§3.3, Functionality Annotation).
4. **verify** — validate candidate annotations through a cycle-consistency check that re-grounds each description back to its element (§3.4, Annotation Verification).
5. **generate-tasks** — compose the verified annotations into instruction-following grounding tasks at multiple image scales (§3.5, Task Generation).

## Installation

Install the package (editable) from the repository root:

```bash
pip install -e autogui_anno
```

The **verify** and **annotate** stages' text cleaning
(`autogui_anno.prompts.get_clean_func`) uses spaCy. After installing, download
the English model once:

```bash
python -m spacy download en_core_web_sm
```

The web scraping path additionally needs Playwright browsers:

```bash
pip install -e "autogui_anno[web]"
playwright install
```

Optional extras: `autogui_anno[analysis]` (matplotlib for `analysis.py`) and
`autogui_anno[dev]` (pytest).

## Configuring the LLM

The pipeline talks to any OpenAI-compatible endpoint through a model registry
YAML file. Copy the example and edit it to point at **your own** endpoint:

```bash
cp autogui_anno/configs/models.example.yaml autogui_anno/configs/models.yaml
```

Each entry names an environment variable (`api_key_env`) that holds the API
key; the default is `AUTOGUI_LLM_API_KEY`. Export your key before running:

```bash
export AUTOGUI_LLM_API_KEY=your-api-key-here
```

A pipeline-tuning file is also provided — copy
`autogui_anno/configs/pipeline.example.yaml` to `pipeline.yaml` and adjust the
thresholds and limits as needed. Both files mirror their dataclass defaults, so
every field is documented inline.

## Usage

The three entrypoints are exposed both as console scripts and as `python -m`
module mains (`main_web`, `main_mobile`, `main_generate_tasks` in
`autogui_anno.cli`).

### Web annotation

Runs collect → reject → annotate → verify over a directory of web
observations.

```bash
autogui-annotate-web \
  --config autogui_anno/configs/pipeline.yaml \
  --models autogui_anno/configs/models.yaml \
  --data path/to/web_observations \
  --out path/to/output \
  --model default-strong \
  --verifiers default-fast default-strong \
  --resume
```

Flags: `--config` (pipeline YAML), `--models` (registry YAML), `--data` (input
directory), and `--out` (output directory) are required. `--model` picks the
registry model to annotate with (defaults to the registry's `default`);
`--verifiers` lists one or more registry models for the verification stage
(defaults to `[default]`); `--resume` continues from existing checkpoints.

### Mobile annotation

Same flags as the web command, over Android observations:

```bash
autogui-annotate-mobile \
  --config autogui_anno/configs/pipeline.yaml \
  --models autogui_anno/configs/models.yaml \
  --data path/to/mobile_observations \
  --out path/to/output
```

### Task generation

Composes verified functionalities into multi-scale grounding tasks:

```bash
autogui-generate-tasks \
  --gts path/to/functionalities.json \
  --traj path/to/trajectory_dir \
  --images path/to/image_out \
  --meta-out path/to/meta_out \
  --full-w 2560 --full-h 1440 \
  --scales 1 2 4
```

Flags: `--gts` (ground-truth functionalities JSON), `--traj` (trajectory
directory), `--images` (resized-image output directory), and `--meta-out`
(metadata output directory) are required. `--full-w`/`--full-h` set the
reference resolution (default `2560`/`1440`); `--scales` lists the downsize
factors (default `1 2 4`).

#### Publishing to the Hugging Face Hub (opt-in)

Task generation uploads nothing by default. To publish the assembled dataset,
pass `--push-to-hub` with a repo id:

```bash
autogui-generate-tasks ... --push-to-hub your-org/your-dataset
```

## Visualizer (didactic demo)

The visualizer runs the pipeline over a handful of samples and renders each
run as an inspectable per-sample record in a small static web page. It is a
teaching demo, not the bulk annotator — for production annotation of many
observations use `autogui-annotate-web`/`autogui-annotate-mobile`, which write
checkpoints rather than viewer records.

```
# 1. Build records (needs AUTOGUI_LLM_API_KEY + a models.yaml)
autogui-visualize-build --config configs/pipeline.yaml --models configs/models.yaml \
    --data /path/to/test_samples --out autogui_anno/autogui_anno/viz/web/viz_data
# 2. View (no key needed; works offline)
autogui-visualize-serve           # serves the packaged page at http://localhost:8000
```

Building needs your LLM key (`AUTOGUI_LLM_API_KEY`) and a `models.yaml`;
viewing needs neither — the page is fully static and works offline.

## Running the tests

The full suite is offline — every test uses fakes and fixtures, so no network
or API key is required:

```bash
cd autogui_anno && python -m pytest tests/ -v
```

One test is skipped unless the spaCy `en_core_web_sm` model is installed (see
Installation).
