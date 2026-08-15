"""Application-level configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class PathsConfig:
    assets_dir: str
    data_dir: str
    calibration_dir: str
    sample_videos_dir: str


@dataclass(frozen=True)
class ModelConfig:
    pose_model_path: str
    racket_model_path: str
    model_complexity: int


@dataclass(frozen=True)
class AppConfig:
    environment: Environment
    paths: PathsConfig
    model: ModelConfig
