#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility CLI entry point.

The pipeline implementation lives in apps.pipeline.
"""

from apps.pipeline import (
    PipelineOptions,
    build_arg_parser,
    build_pipeline_steps,
    main,
    run_full_pipeline,
    run_pipeline,
)

__all__ = [
    "PipelineOptions",
    "build_arg_parser",
    "build_pipeline_steps",
    "main",
    "run_full_pipeline",
    "run_pipeline",
]


if __name__ == "__main__":
    main()
