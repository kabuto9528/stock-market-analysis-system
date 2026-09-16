"""项目配置加载、校验与运行目录管理。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """配置内容不合法或无法安全解析。"""


@dataclass(frozen=True)
class SplitRatios:
    """按时间顺序划分训练、验证和测试集的比例。"""

    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not 0 < value < 1 for value in values):
            raise ConfigError("训练、验证、测试比例都必须位于 0 与 1 之间。")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError("训练、验证、测试比例之和必须等于 1。")


@dataclass(frozen=True)
class LSTMSettings:
    """阶段 0 冻结的 LSTM 默认超参数。"""

    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    loss_function: str
    optimizer: str

    def __post_init__(self) -> None:
        integer_values = (
            self.hidden_size,
            self.num_layers,
            self.batch_size,
            self.max_epochs,
            self.early_stopping_patience,
        )
        if any(value <= 0 for value in integer_values):
            raise ConfigError("LSTM 整数超参数必须为正整数。")
        if not 0 <= self.dropout < 1:
            raise ConfigError("dropout 必须位于 [0, 1) 区间。")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate 必须大于 0。")


@dataclass(frozen=True)
class RuntimePaths:
    """已解析为项目内绝对路径的运行目录。"""

    data_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    models_dir: Path
    experiments_dir: Path
    logs_dir: Path

    def directories(self) -> tuple[Path, ...]:
        """返回需要在运行前创建的全部目录。"""

        return (
            self.data_dir,
            self.artifacts_dir,
            self.cache_dir,
            self.models_dir,
            self.experiments_dir,
            self.logs_dir,
        )


@dataclass(frozen=True)
class ProjectConfig:
    """经过校验、可供服务层和页面使用的项目配置。"""

    project_root: Path
    config_path: Path
    project_name: str
    development_status: str
    split_ratios: SplitRatios
    window_size: int
    lstm: LSTMSettings
    paths: RuntimePaths
    random_seed: int
    tushare_token: str | None


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"配置项 {field_name} 必须是映射。")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"配置项 {field_name} 必须为正整数。")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"配置项 {field_name} 必须为数值。")
    return float(value)


def resolve_project_path(project_root: str | Path, relative_path: str | Path) -> Path:
    """将配置中的相对路径解析到项目内，并拒绝绝对路径与目录穿越。"""

    root = Path(project_root).expanduser().resolve()
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ConfigError(f"运行目录必须使用项目相对路径：{relative_path}")
    resolved = (root / raw_path).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(f"运行目录不能超出项目根目录：{relative_path}")
    return resolved


def create_runtime_directories(paths: RuntimePaths) -> None:
    """创建配置声明的数据、模型、实验与日志运行目录。"""

    for directory in paths.directories():
        directory.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    create_dirs: bool = True,
) -> ProjectConfig:
    """读取 YAML 与 ``TUSHARE_TOKEN`` 环境变量并返回强校验配置。"""

    root = Path(project_root).expanduser().resolve() if project_root else PROJECT_ROOT
    selected_path = Path(config_path) if config_path else Path("configs/default.yaml")
    if not selected_path.is_absolute():
        selected_path = root / selected_path
    selected_path = selected_path.resolve()
    if not selected_path.is_file():
        raise ConfigError(f"配置文件不存在：{selected_path}")

    try:
        raw = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败：{selected_path}") from exc
    root_config = _as_mapping(raw, "root")

    try:
        project = _as_mapping(root_config["project"], "project")
        data = _as_mapping(root_config["data"], "data")
        ratios = _as_mapping(data["split_ratios"], "data.split_ratios")
        lstm_raw = _as_mapping(root_config["lstm"], "lstm")
        paths_raw = _as_mapping(root_config["paths"], "paths")

        split_ratios = SplitRatios(
            train=_number(ratios["train"], "data.split_ratios.train"),
            validation=_number(ratios["validation"], "data.split_ratios.validation"),
            test=_number(ratios["test"], "data.split_ratios.test"),
        )
        lstm = LSTMSettings(
            hidden_size=_positive_int(lstm_raw["hidden_size"], "lstm.hidden_size"),
            num_layers=_positive_int(lstm_raw["num_layers"], "lstm.num_layers"),
            dropout=_number(lstm_raw["dropout"], "lstm.dropout"),
            batch_size=_positive_int(lstm_raw["batch_size"], "lstm.batch_size"),
            learning_rate=_number(lstm_raw["learning_rate"], "lstm.learning_rate"),
            max_epochs=_positive_int(lstm_raw["max_epochs"], "lstm.max_epochs"),
            early_stopping_patience=_positive_int(
                lstm_raw["early_stopping_patience"], "lstm.early_stopping_patience"
            ),
            loss_function=str(lstm_raw["loss_function"]),
            optimizer=str(lstm_raw["optimizer"]),
        )
        runtime_paths = RuntimePaths(
            **{
                key: resolve_project_path(root, paths_raw[key])
                for key in (
                    "data_dir",
                    "artifacts_dir",
                    "cache_dir",
                    "models_dir",
                    "experiments_dir",
                    "logs_dir",
                )
            }
        )
        random_seed = _positive_int(root_config["random_seed"], "random_seed")
        window_size = _positive_int(data["window_size"], "data.window_size")
        project_name = str(project["name"]).strip()
        development_status = str(project["development_status"]).strip()
    except KeyError as exc:
        raise ConfigError(f"缺少必需配置项：{exc.args[0]}") from exc

    if not project_name or not development_status:
        raise ConfigError("项目名称和开发状态不能为空。")

    load_dotenv(dotenv_path=root / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN", "").strip() or None
    config = ProjectConfig(
        project_root=root,
        config_path=selected_path,
        project_name=project_name,
        development_status=development_status,
        split_ratios=split_ratios,
        window_size=window_size,
        lstm=lstm,
        paths=runtime_paths,
        random_seed=random_seed,
        tushare_token=token,
    )
    if create_dirs:
        create_runtime_directories(config.paths)
    return config
