"""Visualizer record builder for the annotation pipeline.

Turns pipeline runs into per-sample JSON records that a static web viewer
(Task 20) renders. ``build_sample_record`` is a pure assembly step; ``run_builder``
is the impure driver that reads sample data, runs the real stages, and writes the
records plus an ``index.json``.
"""
from autogui_anno.viz.build import (
    STAGE_EXPLAINERS,
    build_sample_record,
    run_builder,
)

__all__ = ["STAGE_EXPLAINERS", "build_sample_record", "run_builder"]
