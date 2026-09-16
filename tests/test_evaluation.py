from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_experiment import main as run_experiment_main
from src.evaluation import (
    BacktestError,
    EvaluationInputError,
    ExperimentRecord,
    ExperimentStore,
    direction_accuracy,
    mae,
    r2_score,
    relative_naive_rmse_improvement,
    rmse,
)
from src.services import UnifiedTestBacktestService


def _dates(count: int = 4) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-06-03", periods=count)


def _prediction_frame(
    name: str,
    dates: pd.DatetimeIndex,
    *,
    predicted: list[float],
    actual: list[float] | None = None,
    previous: list[float] | None = None,
) -> pd.DataFrame:
    actual_values = actual or [11.0, 10.0, 10.0, 12.0][: len(dates)]
    previous_values = previous or [10.0, 11.0, 10.0, 10.0][: len(dates)]
    return pd.DataFrame(
        {
            "target_date": dates,
            "previous_close": previous_values,
            "actual": actual_values,
            "predicted": predicted,
            "model_name": name,
        }
    )


def _backtest_result():
    dates = _dates()
    predictions = {
        "Naive": _prediction_frame("Naive", dates, predicted=[10.0, 11.0, 10.0, 10.0]),
        "ModelA": _prediction_frame("ModelA", dates, predicted=[11.0, 10.0, 10.0, 11.5]),
    }
    return UnifiedTestBacktestService(dates).run(predictions)


def test_hand_calculated_regression_metrics() -> None:
    dates = _dates(3)
    actual = [2.0, 4.0, 4.0]
    predicted = [1.0, 5.0, 3.0]

    assert rmse(actual, predicted, target_dates=dates) == pytest.approx(1.0)
    assert mae(actual, predicted, target_dates=dates) == pytest.approx(1.0)
    assert r2_score(actual, predicted, target_dates=dates) == pytest.approx(-0.125)


def test_direction_accuracy_handles_up_down_and_flat_as_three_classes() -> None:
    dates = _dates(3)
    previous = [10.0, 10.0, 10.0]
    actual = [11.0, 9.0, 10.0]

    assert direction_accuracy(
        actual, [12.0, 8.0, 10.0], previous, target_dates=dates
    ) == pytest.approx(1.0)
    assert direction_accuracy(
        actual, [9.0, 10.0, 11.0], previous, target_dates=dates
    ) == pytest.approx(0.0)


def test_relative_naive_rmse_improvement_supports_positive_and_negative_values() -> None:
    assert relative_naive_rmse_improvement(8.0, 10.0) == pytest.approx(20.0)
    assert relative_naive_rmse_improvement(12.0, 10.0) == pytest.approx(-20.0)
    with pytest.raises(EvaluationInputError, match="naive_rmse 必须大于 0"):
        relative_naive_rmse_improvement(0.0, 0.0)


@pytest.mark.parametrize(
    ("actual", "predicted", "dates", "message"),
    [
        ([1.0, 2.0], [1.0], _dates(2), "长度必须一致"),
        ([1.0, np.nan], [1.0, 2.0], _dates(2), "NaN 或 Infinity"),
        ([1.0, 2.0], [1.0, np.inf], _dates(2), "NaN 或 Infinity"),
        ([1.0], [1.0], _dates(1), "至少需要 2"),
        ([1.0, 2.0], [1.0, 2.0], _dates(2)[::-1], "严格升序"),
        ([1.0, 2.0], [1.0, 2.0], [_dates(1)[0], _dates(1)[0]], "重复日期"),
    ],
)
def test_metric_inputs_reject_length_nan_infinity_minimum_and_date_misalignment(
    actual: list[float],
    predicted: list[float],
    dates: object,
    message: str,
) -> None:
    with pytest.raises(EvaluationInputError, match=message):
        rmse(actual, predicted, target_dates=dates)  # type: ignore[arg-type]


def test_backtest_uses_explicit_common_test_dates_and_records_exclusions() -> None:
    dates = _dates()
    naive = _prediction_frame("Naive", dates, predicted=[10.0, 11.0, 10.0, 10.0])
    model = _prediction_frame(
        "ModelA",
        dates[1:],
        predicted=[10.0, 10.0, 11.5],
        actual=[10.0, 10.0, 12.0],
        previous=[11.0, 10.0, 10.0],
    )

    result = UnifiedTestBacktestService(dates).run({"Naive": naive, "ModelA": model})

    assert result.alignment.common_target_dates == tuple(
        date.date().isoformat() for date in dates[1:]
    )
    assert result.alignment.excluded_target_dates["Naive"] == (
        dates[0].date().isoformat(),
    )
    assert result.metrics["Naive"].sample_count == 3
    assert all(
        frame["target_date"].tolist() == list(dates[1:])
        for frame in result.predictions.values()
    )


def test_backtest_rejects_validation_date_in_final_test_evaluation() -> None:
    test_dates = _dates()
    invalid_dates = pd.DatetimeIndex([test_dates[0] - pd.offsets.BDay(1), *test_dates[1:]])
    naive = _prediction_frame(
        "Naive", invalid_dates, predicted=[10.0, 11.0, 10.0, 10.0]
    )
    model = _prediction_frame(
        "ModelA", invalid_dates, predicted=[11.0, 10.0, 10.0, 11.0]
    )

    with pytest.raises(BacktestError, match="非测试集目标日期"):
        UnifiedTestBacktestService(test_dates).run({"Naive": naive, "ModelA": model})


def test_model_ranking_uses_da_and_relative_improvement_not_r2() -> None:
    dates = _dates()
    naive = _prediction_frame("Naive", dates, predicted=[10.0, 11.0, 10.0, 10.0])
    high_da_bad_level = _prediction_frame(
        "HighDA", dates, predicted=[100.0, -100.0, 10.0, 100.0]
    )
    low_da_good_level = _prediction_frame(
        "LowDA", dates, predicted=[10.9, 10.1, 10.1, 9.9]
    )

    result = UnifiedTestBacktestService(dates).run(
        {"Naive": naive, "HighDA": high_da_bad_level, "LowDA": low_da_good_level}
    )
    comparison = result.comparison.set_index("model_name")

    assert comparison.loc["HighDA", "direction_accuracy"] > comparison.loc[
        "LowDA", "direction_accuracy"
    ]
    assert comparison.loc["HighDA", "r2"] < comparison.loc["LowDA", "r2"]
    assert comparison.loc["HighDA", "rank"] < comparison.loc["LowDA", "rank"]


def test_experiment_save_and_load_writes_all_required_artifacts(tmp_path: Path) -> None:
    result = _backtest_result()
    store = ExperimentStore(tmp_path / "experiments")
    record = ExperimentRecord(
        experiment_id="stage6_save_read",
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        result_kind="test",
        status="completed",
        ts_code="600000.SH",
        data_range={"start_date": "2024-01-02", "end_date": "2024-06-30"},
        split_boundaries={
            "train": {"start_date": "2024-01-02", "end_date": "2024-04-30"},
            "validation": {"start_date": "2024-05-01", "end_date": "2024-05-31"},
            "test": {"start_date": "2024-06-03", "end_date": "2024-06-06"},
        },
        feature_columns=("close",),
        window_size=5,
        model_parameters={"Naive": {}, "ModelA": {"parameter": 1}},
        random_seed=42,
        metrics={},
        elapsed_seconds={"total": 0.1},
        artifact_paths={"model": "stage6_save_read/model.pt"},
        notes=("固定测试记录，不是正式实验结果。",),
    )

    saved = store.save(record, result)
    loaded = store.load(record.experiment_id)
    names = {path.name for path in store.list_artifact_files(record.experiment_id)}

    assert loaded == saved
    assert loaded.metrics["ModelA"]["sample_count"] == 4
    assert {
        "experiment_config.json",
        "metrics.json",
        "metrics.csv",
        "daily_predictions.csv",
        "model_comparison.csv",
        "alignment.json",
        "experiment_record.json",
    }.issubset(names)
    config = json.loads(
        (store.experiment_directory(record.experiment_id) / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["result_kind"] == "test"
    assert config["test_evaluation_count"] == 1


def test_run_experiment_fixed_small_sample_end_to_end_is_marked_test(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "experiments"
    exit_code = run_experiment_main(
        [
            "--file",
            str(project_root / "tests/fixtures/stage6_small_market.csv"),
            "--config",
            str(project_root / "tests/fixtures/stage6_test_config.yaml"),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "stage6_fixed_small_test",
            "--result-kind",
            "test",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    experiment_dir = output_dir / "stage6_fixed_small_test"
    record = json.loads(
        (experiment_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(experiment_dir / "model_comparison.csv")
    history = json.loads(
        (experiment_dir / "training_history.json").read_text(encoding="utf-8")
    )
    alignment = json.loads((experiment_dir / "alignment.json").read_text(encoding="utf-8"))

    assert record["result_kind"] == "test"
    assert record["status"] == "completed"
    assert record["test_evaluation_count"] == 1
    assert set(comparison["model_name"]) == {
        "Naive",
        "MA5",
        "MA10",
        "MA20",
        "SES",
        "ARIMA",
        "LSTM",
    }
    assert history["test_samples_used_for_training_or_early_stopping"] == 0
    assert alignment["comparison_partition"] == "test"
    assert len(alignment["common_target_dates"]) == 15
    assert math.isfinite(float(comparison["rmse"].max()))

