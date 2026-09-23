from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import plotly.io as pio
import pytest

from src.config import LSTMSettings, ProjectConfig, RuntimePaths, SplitRatios
from src.services import TrainingRequest, WebApplicationService, WebServiceError


@pytest.fixture()
def web_service(tmp_path: Path) -> WebApplicationService:
    paths = RuntimePaths(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "data" / "cache",
        models_dir=tmp_path / "artifacts" / "models",
        experiments_dir=tmp_path / "artifacts" / "experiments",
        logs_dir=tmp_path / "artifacts" / "logs",
    )
    for directory in paths.directories():
        directory.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        project_name="测试 Web 系统",
        development_status="阶段 7 测试",
        split_ratios=SplitRatios(0.7, 0.15, 0.15),
        window_size=5,
        lstm=LSTMSettings(
            hidden_size=8, num_layers=1, dropout=0.0, batch_size=8,
            learning_rate=0.01, max_epochs=2, early_stopping_patience=1,
            loss_function="MSELoss", optimizer="Adam",
        ),
        paths=paths,
        random_seed=42,
        tushare_token=None,
    )
    return WebApplicationService(config)


def fixture_bytes() -> bytes:
    return (Path(__file__).parent / "fixtures" / "stage6_small_market.csv").read_bytes()


def test_csv_offline_flow_cache_quality_and_plotly(web_service: WebApplicationService) -> None:
    context = web_service.data.import_csv_bytes(
        "market.csv", fixture_bytes(), "600000.SH", save_cache=True
    )
    assert context.ts_code == "600000.SH"
    assert len(context.frame) == 90
    assert context.quality["passed"] is True
    assert web_service.data.list_cached_stocks() == ("600000.SH",)
    cached = web_service.data.load_cache("600000.SH")
    assert cached.frame["trade_date"].is_monotonic_increasing
    figures = web_service.analysis.figures(cached.frame)
    assert set(figures) == {"kline", "close_ma", "volume", "returns", "correlation"}
    assert all(figure.layout.title.text for figure in figures.values())
    assert all('"bdata"' not in pio.to_json(figure, validate=False) for figure in figures.values())


def test_missing_token_returns_friendly_error(web_service: WebApplicationService, monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    with pytest.raises(WebServiceError, match="Token") as captured:
        web_service.data.fetch_tushare("600000.SH", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date())
    assert "CSV" in captured.value.user_message()


def test_training_progress_artifacts_prediction_and_download(web_service: WebApplicationService) -> None:
    context = web_service.data.import_csv_bytes("market.csv", fixture_bytes(), "600000.SH")
    epochs: list[int] = []
    summary = web_service.training.train(
        context.frame,
        TrainingRequest(
            mode="univariate", window_size=5, hidden_size=8, num_layers=1,
            dropout=0.0, batch_size=8, learning_rate=0.01,
            max_epochs=2, patience=1, random_seed=42, device="cpu",
        ),
        progress_callback=lambda record: epochs.append(record.epoch),
    )
    assert epochs == [1, 2]
    assert summary.best_epoch in {1, 2}
    assert not summary.history.empty
    assert {"Naive", "MA5", "MA10", "MA20", "SES", "ARIMA", "LSTM-单变量"} <= set(summary.metrics["model_name"])
    dashboard = web_service.experiments.load(summary.experiment_id)
    assert dashboard.record.status == "completed"
    assert dashboard.record.result_kind == "test"
    assert not dashboard.predictions.empty
    assert (dashboard.directory / "selected_lstm.pt").is_file()
    assert web_service.experiments.csv_bytes(dashboard.comparison).startswith(b"\xef\xbb\xbf")
    assert summary.experiment_id in web_service.experiments.list_experiments_for_stock("600000.SH")
    assert web_service.experiments.list_experiments_for_stock("000001.SZ") == ()
    prediction = web_service.training.load_pretrained_prediction(summary.experiment_id, context.frame, device="cpu")
    assert len(prediction) == 1
    assert prediction["predicted"].notna().all()

    multivariate = web_service.training.train(
        context.frame,
        TrainingRequest(
            mode="multivariate", window_size=5, hidden_size=8, num_layers=1,
            dropout=0.0, batch_size=8, learning_rate=0.01,
            max_epochs=1, patience=1, random_seed=42, device="cpu",
        ),
    )
    multi_dashboard = web_service.experiments.load(multivariate.experiment_id)
    assert set(multi_dashboard.comparison["model_name"]) == {"Naive", "LSTM-多变量"}
    combined, combined_predictions = web_service.experiments.combine_compatible(
        dashboard, multi_dashboard
    )
    assert {"LSTM-单变量", "LSTM-多变量"} <= set(combined["model_name"])
    assert {"LSTM-单变量", "LSTM-多变量"} <= set(combined_predictions["model_name"])


def test_insufficient_data_and_configuration_errors_are_friendly(web_service: WebApplicationService) -> None:
    context = web_service.data.import_csv_bytes("market.csv", fixture_bytes(), "600000.SH")
    with pytest.raises(WebServiceError, match="模型训练失败") as captured:
        web_service.training.train(context.frame.head(12), TrainingRequest(window_size=10, max_epochs=1, patience=1, device="cpu"))
    assert "建议" in captured.value.user_message()
    with pytest.raises(WebServiceError, match="Dropout"):
        TrainingRequest(dropout=1.0).validate()


def test_pages_delegate_core_work_to_service_layer() -> None:
    root = Path(__file__).resolve().parents[1]
    page_paths = [root / "app.py", *sorted((root / "pages").glob("*.py"))]
    forbidden = {"src.data", "src.models", "src.evaluation", "src.baselines"}
    for path in page_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any(module in forbidden or any(module.startswith(item + ".") for item in forbidden) for module in imported), path.name
    training_page = (root / "pages" / "3_模型训练.py").read_text(encoding="utf-8")
    assert "if submitted:" in training_page
    assert "service.training.train" in training_page
    assert training_page.index("if submitted:") < training_page.index("service.training.train")

    runtime = (root / "src" / "web_runtime.py").read_text(encoding="utf-8")
    cached_analysis_block = runtime.split("def cached_analysis", 1)[0].rsplit("\n", 3)[-3:]
    assert not any("cache_data" in line for line in cached_analysis_block)

    for path in (root / "pages").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "plotly_chart"
            ):
                assert any(keyword.arg == "key" for keyword in node.keywords), path.name
