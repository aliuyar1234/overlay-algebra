from __future__ import annotations

from pathlib import Path

from .artifacts import RunPaths, build_run_paths
from .compiler import compile_targets
from .config import AppConfig, ArtifactLayout, load_app_config
from .metrics import score_prediction
from .parsers import ParseError, parse_condition

__all__ = [
    "AppConfig",
    "ArtifactLayout",
    "ParseError",
    "RunPaths",
    "build_run_paths",
    "compile_targets",
    "load_app_config",
    "parse_condition",
    "score_prediction",
]

__version__ = "0.1.0a0"
PACKAGE_ROOT = Path(__file__).resolve().parent
