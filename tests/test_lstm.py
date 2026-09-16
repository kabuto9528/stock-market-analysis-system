from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    MULTIVARIATE_FEATURES,
    UNIVARIATE_FEATURES,
    ScalerManager,
    build_windows,
    clean_market_data,
    create_dataloaders,
    split_by_time,
)
from src.models import (
    LSTMPredictor,
    LSTMRegressor,
    LSTMTrainer,
    ModelArtifactError,
    ModelConfigurationError,
    TrainerConfig,
    load_model_artifact,
    save_model_artifact,
    set_random_seed,
)


def _market_frame(rows: int = 64) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 10.0 + index * 0.08 + np.sin(index / 4.0) * 0.15
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": pd.bdate_range("2024-01-02", periods=rows),
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "vol": 1000.0 + index * 5.0,
            "amount": (1000.0 + index * 5.0) * close,
        }
    )


def _pipeline(
    features: tuple[str, ...] = UNIVARIATE_FEATURES,
    *,
    rows: int = 64,
    window_size: int = 5,
    batch_size: int = 8,
):
    cleaned = clean_market_data(_market_frame(rows)).data
    split = split_by_time(cleaned)
    scaler = ScalerManager.fit(split, features)
    windows = build_windows(split, scaler, features, window_size=window_size)
    loaders = create_dataloaders(
        windows, batch_size=batch_size, train_shuffle=False, seed=42
    )
    return cleaned, scaler, windows, loaders


@pytest.mark.parametrize(
    ("features", "input_size"),
    [(UNIVARIATE_FEATURES, 1), (MULTIVARIATE_FEATURES, 6)],
)
def test_univariate_and_multivariate_forward_output_shape(
    features: tuple[str, ...], input_size: int
) -> None:
    model = LSTMRegressor(
        input_size=input_size, hidden_size=8, num_layers=2, dropout=0.1
    )
    inputs = torch.randn(4, 7, len(features))

    outputs = model(inputs)

    assert model.lstm.batch_first is True
    assert tuple(outputs.shape) == (4, 1)
    assert model.get_config()["input_size"] == input_size


def test_small_sample_training_runs_on_cpu_and_records_history() -> None:
    _, _, _, loaders = _pipeline()
    set_random_seed(42)
    model = LSTMRegressor(input_size=1, hidden_size=6, num_layers=1, dropout=0.2)
    result = LSTMTrainer(
        TrainerConfig(learning_rate=0.01, max_epochs=3, early_stopping_patience=3),
        device="cpu",
    ).fit(model, loaders["train"], loaders["validation"])

    assert result.device == "cpu"
    assert 1 <= result.best_epoch <= len(result.history) == 3
    assert result.best_validation_loss >= 0
    assert all(record.epoch == index for index, record in enumerate(result.history, 1))
    assert all(record.train_loss >= 0 for record in result.history)
    assert all(record.validation_loss >= 0 for record in result.history)
    assert all(record.elapsed_seconds >= 0 for record in result.history)


def test_early_stopping_restores_lowest_validation_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, loaders = _pipeline()
    set_random_seed(7)
    model = LSTMRegressor(input_size=1, hidden_size=3, num_layers=1, dropout=0.0)
    first_parameter = next(model.parameters())
    initial = first_parameter.detach().clone()
    validation_losses = iter((1.0, 0.5, 0.6, 0.7))

    def fake_mean_loss(model, loader, criterion, device, *, optimizer):
        if optimizer is not None:
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            return 1.0
        return next(validation_losses)

    monkeypatch.setattr("src.models.training._mean_loss", fake_mean_loss)
    result = LSTMTrainer(
        TrainerConfig(max_epochs=10, early_stopping_patience=2), device="cpu"
    ).fit(model, loaders["train"], loaders["validation"])

    assert result.stopped_early is True
    assert len(result.history) == 4
    assert result.best_epoch == 2
    assert result.best_validation_loss == pytest.approx(0.5)
    assert torch.equal(next(model.parameters()).detach(), initial + 2.0)


def test_test_split_cannot_be_used_for_early_stopping() -> None:
    _, _, _, loaders = _pipeline()
    model = LSTMRegressor(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)

    with pytest.raises(ModelConfigurationError, match="只允许使用 validation"):
        LSTMTrainer(TrainerConfig(max_epochs=1), device="cpu").fit(
            model, loaders["train"], loaders["test"]
        )


def test_model_and_scaler_save_load_keep_predictions_and_validate_metadata(
    tmp_path: Path,
) -> None:
    _, scaler, windows, _ = _pipeline(MULTIVARIATE_FEATURES)
    set_random_seed(11)
    model = LSTMRegressor(input_size=6, hidden_size=5, num_layers=2, dropout=0.1)
    model.eval()
    reference_input = torch.as_tensor(windows.validation.X[:3], dtype=torch.float32)
    with torch.inference_mode():
        expected = model(reference_input).numpy()
    model_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.json"
    scaler.save(scaler_path)
    save_model_artifact(
        model_path,
        model,
        feature_columns=MULTIVARIATE_FEATURES,
        target_column="close",
        window_size=5,
        random_seed=11,
        best_epoch=3,
        best_validation_loss=0.02,
    )

    loaded = load_model_artifact(
        model_path,
        scaler_path,
        expected_feature_columns=MULTIVARIATE_FEATURES,
        expected_window_size=5,
        expected_model_parameters=model.get_config(),
    )
    with torch.inference_mode():
        actual = loaded.model(reference_input).numpy()

    assert np.array_equal(actual, expected)
    assert loaded.metadata.best_epoch == 3
    assert loaded.metadata.feature_columns == MULTIVARIATE_FEATURES
    assert loaded.scaler.feature_columns == MULTIVARIATE_FEATURES
    with pytest.raises(ModelArtifactError, match="窗口长度不一致"):
        load_model_artifact(model_path, scaler_path, expected_window_size=6)
    with pytest.raises(ModelArtifactError, match="特征顺序"):
        load_model_artifact(
            model_path,
            scaler_path,
            expected_feature_columns=tuple(reversed(MULTIVARIATE_FEATURES)),
        )
    wrong_parameters = model.get_config() | {"hidden_size": 7}
    with pytest.raises(ModelArtifactError, match="结构参数不一致"):
        load_model_artifact(
            model_path,
            scaler_path,
            expected_model_parameters=wrong_parameters,
        )


def test_batch_and_next_day_prediction_inverse_scale_and_keep_dates() -> None:
    cleaned, scaler, windows, _ = _pipeline()
    set_random_seed(3)
    model = LSTMRegressor(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)
    predictor = LSTMPredictor(
        model,
        scaler,
        feature_columns=UNIVARIATE_FEATURES,
        window_size=5,
        device="cpu",
    )

    batch = predictor.predict_samples(windows.validation, batch_size=3)
    next_date = cleaned.iloc[-1]["trade_date"] + pd.offsets.BDay(1)
    next_prediction = predictor.predict_next_trading_day(
        cleaned, target_date=next_date
    )

    assert list(batch.columns) == [
        "target_date",
        "previous_close",
        "actual",
        "predicted",
        "model_name",
    ]
    assert batch["target_date"].tolist() == pd.to_datetime(
        windows.validation.target_dates
    ).tolist()
    assert np.isfinite(batch["predicted"]).all()
    assert next_prediction.iloc[0]["target_date"] == next_date
    assert next_prediction.iloc[0]["previous_close"] == pytest.approx(
        cleaned.iloc[-1]["close"]
    )
    assert pd.isna(next_prediction.iloc[0]["actual"])
    assert np.isfinite(next_prediction.iloc[0]["predicted"])


def test_fixed_seed_reproduces_initialization_and_short_training() -> None:
    _, _, windows, loaders = _pipeline()

    def train_once():
        set_random_seed(42)
        model = LSTMRegressor(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)
        result = LSTMTrainer(
            TrainerConfig(learning_rate=0.005, max_epochs=2, early_stopping_patience=2),
            device="cpu",
        ).fit(model, loaders["train"], loaders["validation"])
        predictor_input = torch.as_tensor(windows.validation.X, dtype=torch.float32)
        with torch.inference_mode():
            predictions = model(predictor_input).cpu().numpy()
        return result, predictions

    first_result, first_predictions = train_once()
    second_result, second_predictions = train_once()

    assert first_result.best_epoch == second_result.best_epoch
    assert first_result.best_validation_loss == pytest.approx(
        second_result.best_validation_loss, rel=0, abs=0
    )
    assert [record.train_loss for record in first_result.history] == pytest.approx(
        [record.train_loss for record in second_result.history], rel=0, abs=0
    )
    assert np.array_equal(first_predictions, second_predictions)


def test_train_model_cli_uses_local_csv_and_writes_three_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.train_model import main

    csv_path = tmp_path / "market.csv"
    _market_frame(50).assign(
        trade_date=lambda frame: frame["trade_date"].dt.strftime("%Y%m%d")
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: CLI测试
  development_status: 阶段 5
data:
  split_ratios: {train: 0.70, validation: 0.15, test: 0.15}
  window_size: 5
lstm:
  hidden_size: 4
  num_layers: 1
  dropout: 0.0
  batch_size: 8
  learning_rate: 0.01
  max_epochs: 2
  early_stopping_patience: 2
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
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "--file",
            str(csv_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "univariate",
            "--device",
            "cpu",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["test_samples_used_for_training_or_early_stopping"] == 0
    assert Path(output["model_path"]).is_file()
    assert Path(output["scaler_path"]).is_file()
    assert Path(output["history_path"]).is_file()
