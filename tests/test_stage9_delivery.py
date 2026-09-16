from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_experiment import main as run_experiment_main
from src.config import LSTMSettings, ProjectConfig, RuntimePaths, SplitRatios
from src.data import (
    UNIVARIATE_FEATURES,
    ScalerManager,
    assert_partition_dates_disjoint,
    assert_target_dates_disjoint,
    build_windows,
    clean_market_data,
    load_market_csv,
    split_by_time,
)
from src.models import LSTMPredictor, ModelArtifactError, load_model_artifact
from src.services import WebApplicationService, WebServiceError

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "demo" / "offline_market.csv"
DEMO_EXPERIMENT = ROOT / "demo" / "offline_demo_stage9"


def _config(root: Path) -> ProjectConfig:
    paths = RuntimePaths(
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        cache_dir=root / "data" / "cache",
        models_dir=root / "artifacts" / "models",
        experiments_dir=root / "artifacts" / "experiments",
        logs_dir=root / "artifacts" / "logs",
    )
    for directory in paths.directories():
        directory.mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        project_root=root,
        config_path=root / "config.yaml",
        project_name="阶段 9 测试",
        development_status="阶段 9",
        split_ratios=SplitRatios(0.70, 0.15, 0.15),
        window_size=5,
        lstm=LSTMSettings(
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            batch_size=8,
            learning_rate=0.01,
            max_epochs=2,
            early_stopping_patience=1,
            loss_function="MSELoss",
            optimizer="Adam",
        ),
        paths=paths,
        random_seed=42,
        tushare_token=None,
    )


def test_stage9_small_csv_end_to_end_and_saved_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "experiments"
    exit_code = run_experiment_main(
        [
            "--file", str(DEMO_CSV),
            "--config", str(ROOT / "tests" / "fixtures" / "stage6_test_config.yaml"),
            "--output-dir", str(output),
            "--experiment-id", "stage9_e2e",
            "--result-kind", "test",
            "--device", "cpu",
        ]
    )
    assert exit_code == 0

    directory = output / "stage9_e2e"
    required = {
        "experiment_record.json", "experiment_config.json", "alignment.json",
        "daily_predictions.csv", "metrics.json", "metrics.csv",
        "model_comparison.csv", "selected_lstm.pt",
        "selected_lstm.scaler.json", "training_history.json",
    }
    assert required <= {path.name for path in directory.iterdir()}

    record = json.loads((directory / "experiment_record.json").read_text(encoding="utf-8"))
    history = json.loads((directory / "training_history.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(directory / "daily_predictions.csv", parse_dates=["target_date"])
    comparison = pd.read_csv(directory / "model_comparison.csv")
    assert record["status"] == "completed"
    assert record["evaluation_partition"] == "test"
    assert record["test_evaluation_count"] == 1
    assert history["test_samples_used_for_training_or_early_stopping"] == 0
    assert history["best_epoch"] == min(
        history["history"], key=lambda row: row["validation_loss"]
    )["epoch"]
    assert comparison[["rmse", "mae", "direction_accuracy"]].notna().all().all()
    date_sets = [
        set(group["target_date"])
        for _, group in predictions.groupby("model_name", sort=False)
    ]
    assert date_sets and all(dates == date_sets[0] for dates in date_sets[1:])


def test_stage9_leakage_and_inverse_scaling_audit() -> None:
    cleaned = clean_market_data(load_market_csv(DEMO_CSV).data).data
    split = split_by_time(cleaned)
    assert_partition_dates_disjoint(split)
    scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)
    assert scaler.training_start_date == split.train.boundary.start_date
    assert scaler.training_end_date == split.train.boundary.end_date
    assert scaler.feature_scaler.data_max_[0] == pytest.approx(split.train.frame["close"].max())

    windows = build_windows(split, scaler, UNIVARIATE_FEATURES, window_size=5)
    assert_target_dates_disjoint(windows)
    for samples in windows:
        assert np.all(samples.input_end_dates < samples.target_dates)

    changed = cleaned.copy()
    first_test_index = split.test.boundary.start_index
    changed.loc[first_test_index:, "close"] += 10000.0
    changed_split = split_by_time(changed)
    changed_windows = build_windows(changed_split, scaler, UNIVARIATE_FEATURES, window_size=5)
    np.testing.assert_allclose(changed_windows.test.X[0], windows.test.X[0])
    assert changed_windows.test.raw_targets[0] != windows.test.raw_targets[0]

    artifact = load_model_artifact(
        DEMO_EXPERIMENT / "selected_lstm.pt",
        DEMO_EXPERIMENT / "selected_lstm.scaler.json",
        expected_feature_columns=UNIVARIATE_FEATURES,
        expected_window_size=5,
    )
    predictor = LSTMPredictor(
        artifact.model, artifact.scaler,
        feature_columns=UNIVARIATE_FEATURES, target_column="close",
        window_size=5, device="cpu", model_name="LSTM",
    )
    predicted = predictor.predict_samples(windows.test)
    with torch.inference_mode():
        scaled = artifact.model(torch.as_tensor(windows.test.X, dtype=torch.float32)).numpy()
    expected = artifact.scaler.target_scaler.inverse_transform(scaled).reshape(-1)
    np.testing.assert_allclose(predicted["predicted"], expected, rtol=1e-6, atol=1e-6)


def test_model_loader_rejects_feature_window_and_input_dimension_mismatch(tmp_path: Path) -> None:
    model_path = DEMO_EXPERIMENT / "selected_lstm.pt"
    scaler_path = DEMO_EXPERIMENT / "selected_lstm.scaler.json"
    with pytest.raises(ModelArtifactError, match="特征顺序"):
        load_model_artifact(
            model_path, scaler_path,
            expected_feature_columns=("open", "close"),
        )
    with pytest.raises(ModelArtifactError, match="窗口长度"):
        load_model_artifact(model_path, scaler_path, expected_window_size=60)

    envelope = torch.load(model_path, map_location="cpu", weights_only=True)
    envelope["metadata"]["model_parameters"]["input_size"] = 2
    tampered = tmp_path / "input_size_mismatch.pt"
    torch.save(envelope, tampered)
    with pytest.raises(ModelArtifactError, match="input_size"):
        load_model_artifact(tampered, scaler_path)


def test_bundled_offline_resources_install_and_friendly_model_errors(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    shutil.copy2(DEMO_CSV, tmp_path / "demo" / "offline_market.csv")
    shutil.copytree(DEMO_EXPERIMENT, tmp_path / "demo" / "offline_demo_stage9")
    service = WebApplicationService(_config(tmp_path))

    installed = tmp_path / "artifacts" / "experiments" / "offline_demo_stage9"
    assert (installed / "selected_lstm.pt").is_file()
    assert (installed / "selected_lstm.scaler.json").is_file()
    assert "offline_demo_stage9" in service.experiments.list_experiments()

    context = service.data.load_offline_demo()
    output = service.training.load_pretrained_prediction(
        "offline_demo_stage9", context.frame, device="cpu"
    )
    assert len(output) == 1 and np.isfinite(output.loc[0, "predicted"])

    with pytest.raises(WebServiceError, match="实验配置读取失败"):
        service.training.load_pretrained_prediction("missing-model", context.frame)
    wrong_stock = context.frame.copy()
    wrong_stock["ts_code"] = "000001.SZ"
    with pytest.raises(WebServiceError, match="配置不匹配"):
        service.training.load_pretrained_prediction(
            "offline_demo_stage9", wrong_stock, device="cpu"
        )


def test_streamlit_training_is_explicit_and_slow_operations_show_status() -> None:
    training_path = ROOT / "pages" / "3_模型训练.py"
    training = training_path.read_text(encoding="utf-8")
    tree = ast.parse(training, filename=str(training_path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    ]
    assert len(calls) == 1
    assert "if submitted:" in training
    assert training.index("if submitted:") < training.index("service.training.train")
    assert "st.spinner" in training and "st.progress" in training

    acquisition = (ROOT / "pages" / "1_数据获取.py").read_text(encoding="utf-8")
    analysis = (ROOT / "pages" / "2_数据分析.py").read_text(encoding="utf-8")
    prediction = (ROOT / "pages" / "4_预测分析.py").read_text(encoding="utf-8")
    assert acquisition.count("st.spinner") >= 4
    assert "st.spinner" in analysis
    assert "正在校验并加载预训练模型" in prediction


def test_production_files_contain_no_hardcoded_secret_or_user_absolute_path() -> None:
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "pages", ROOT / "configs"]
    files = [ROOT / "app.py"]
    for directory in roots:
        files.extend(path for path in directory.rglob("*") if path.suffix in {".py", ".yaml", ".yml"})
    forbidden_path = re.compile(r"(?:[A-Za-z]:\\Users\\|D:\\1graduation_project)", re.IGNORECASE)
    secret_assignment = re.compile(
        r"(?:password|passwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
        re.IGNORECASE,
    )
    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if forbidden_path.search(text):
            failures.append(f"absolute path: {path.relative_to(ROOT)}")
        if secret_assignment.search(text):
            failures.append(f"secret literal: {path.relative_to(ROOT)}")
        if re.search(r"(?:print|logger\.[a-z]+)\([^\n]*(?:_token|password|secret)", text, re.IGNORECASE):
            failures.append(f"sensitive log: {path.relative_to(ROOT)}")
    assert not failures, failures
    assert (ROOT / ".env.example").read_text(encoding="utf-8").strip() == (
        "TUSHARE_TOKEN=your_tushare_token_here"
    )
