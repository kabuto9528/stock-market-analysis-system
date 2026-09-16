from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_config, resolve_project_path


def _write_config(root: Path, *, train: float = 0.70) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "default.yaml"
    config_path.write_text(
        f"""
project:
  name: 测试项目
  development_status: 阶段 1

data:
  split_ratios:
    train: {train}
    validation: 0.15
    test: 0.15
  window_size: 60
lstm:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
  loss_function: MSELoss
  optimizer: Adam
paths:
  data_dir: data
  artifacts_dir: artifacts
  cache_dir: data/cache
  models_dir: artifacts/models
  experiments_dir: artifacts/experiments
  logs_dir: artifacts/logs
random_seed: 42
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_load_config_resolves_paths_and_creates_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-placeholder-token")

    config = load_config(project_root=tmp_path)

    assert config.split_ratios.train == pytest.approx(0.70)
    assert config.window_size == 60
    assert config.random_seed == 42
    assert config.tushare_token == "test-placeholder-token"
    assert config.paths.data_dir == (tmp_path / "data").resolve()
    assert config.paths.models_dir == (tmp_path / "artifacts/models").resolve()
    assert all(path.is_dir() for path in config.paths.directories())


def test_load_config_rejects_ratios_that_do_not_sum_to_one(tmp_path: Path) -> None:
    _write_config(tmp_path, train=0.60)

    with pytest.raises(ConfigError, match="比例之和必须等于 1"):
        load_config(project_root=tmp_path, create_dirs=False)


def test_resolve_project_path_rejects_parent_directory_escape(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不能超出项目根目录"):
        resolve_project_path(tmp_path, "../outside")


def test_default_config_matches_stage_zero_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root=project_root, create_dirs=False)

    assert config.split_ratios.train == pytest.approx(0.70)
    assert config.split_ratios.validation == pytest.approx(0.15)
    assert config.split_ratios.test == pytest.approx(0.15)
    assert config.lstm.hidden_size == 64
    assert config.lstm.num_layers == 2
    assert config.lstm.dropout == pytest.approx(0.2)
    assert config.lstm.batch_size == 32
    assert config.lstm.learning_rate == pytest.approx(0.001)
    assert config.lstm.max_epochs == 100
    assert config.lstm.early_stopping_patience == 10
