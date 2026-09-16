from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    MULTIVARIATE_FEATURES,
    UNIVARIATE_FEATURES,
    MarketDataValidationError,
    ScalerManager,
    StockWindowDataset,
    assert_target_dates_disjoint,
    build_windows,
    clean_market_data,
    create_dataloaders,
    split_by_time,
)


def _market_frame(rows: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    close = np.arange(rows, dtype=float) + 10.0
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * rows,
            "trade_date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "vol": np.arange(rows, dtype=float) + 1000.0,
            "amount": (np.arange(rows, dtype=float) + 1000.0) * close,
        }
    )


def test_cleaning_sorts_deduplicates_converts_and_only_forward_fills() -> None:
    frame = pd.DataFrame(
        [
            ["600000.SH", "2024-01-04", "13", "14", "12", "", "130", "1690"],
            ["600000.SH", "2024-01-01", "", "", "", "", "", ""],
            ["600000.SH", "2024-01-03", "12", "13", "11", "12.5", "120", "1500"],
            ["600000.SH", "2024-01-02", "11", "12", "10", "11.5", "110", "1265"],
            ["600000.SH", "2024-01-03", "12.2", "13.2", "11.2", "12.8", "121", "1548.8"],
        ],
        columns=("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"),
    )

    result = clean_market_data(frame)

    assert result.data["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert result.data.loc[1, "close"] == pytest.approx(12.8)
    assert result.data.loc[2, "close"] == pytest.approx(12.8)
    assert all(str(dtype) == "float64" for dtype in result.data.iloc[:, 2:].dtypes)
    assert result.report.duplicate_rows_removed == 1
    assert result.report.leading_rows_removed == 1
    assert result.report.leading_dates_removed == ("2024-01-01",)
    assert result.report.forward_filled_by_column["close"] == 1


def test_cleaning_rejects_non_finite_values() -> None:
    frame = _market_frame(10)
    frame.loc[4, "close"] = np.inf

    with pytest.raises(MarketDataValidationError, match="正负无穷"):
        clean_market_data(frame)


def test_time_split_is_ordered_70_15_15_and_dates_do_not_overlap() -> None:
    data = clean_market_data(_market_frame(100)).data

    split = split_by_time(data)

    assert (len(split.train.frame), len(split.validation.frame), len(split.test.frame)) == (70, 15, 15)
    assert split.train.boundary.start_date == "2024-01-02"
    assert split.train.boundary.end_date == data.iloc[69]["trade_date"].date().isoformat()
    assert split.validation.boundary.start_date == data.iloc[70]["trade_date"].date().isoformat()
    assert split.test.boundary.start_date == data.iloc[85]["trade_date"].date().isoformat()
    train_dates = set(split.train.frame["trade_date"])
    validation_dates = set(split.validation.frame["trade_date"])
    test_dates = set(split.test.frame["trade_date"])
    assert not train_dates & validation_dates
    assert not train_dates & test_dates
    assert not validation_dates & test_dates


@pytest.mark.parametrize(
    "ratios",
    [(0.7, 0.2, 0.2), (0.0, 0.5, 0.5), (0.7, 0.3)],
)
def test_time_split_rejects_invalid_ratios(ratios: tuple[float, ...]) -> None:
    with pytest.raises(MarketDataValidationError, match="比例"):
        split_by_time(clean_market_data(_market_frame(20)).data, ratios)


def test_scaler_is_fit_only_on_training_partition_and_rejects_raw_frame() -> None:
    frame = _market_frame(100)
    frame.loc[70:, "close"] = frame.loc[70:, "close"] + 10_000.0
    split = split_by_time(clean_market_data(frame).data)

    scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)

    assert scaler.feature_scaler.data_max_[0] == pytest.approx(split.train.frame["close"].max())
    assert scaler.target_scaler.data_max_[0] == pytest.approx(split.train.frame["close"].max())
    assert scaler.transform_features(split.validation.frame).max() > 1.0
    with pytest.raises(MarketDataValidationError, match="禁止传入全量"):
        ScalerManager.fit(split.data, UNIVARIATE_FEATURES)  # type: ignore[arg-type]


def test_scaler_round_trip_safe_load_inverse_and_feature_order_check(tmp_path: Path) -> None:
    split = split_by_time(clean_market_data(_market_frame(100)).data)
    scaler = ScalerManager.fit(split, MULTIVARIATE_FEATURES)
    path = scaler.save(tmp_path / "scaler.json")

    loaded = ScalerManager.load(
        path,
        expected_feature_columns=MULTIVARIATE_FEATURES,
        expected_target_column="close",
    )
    raw = split.validation.frame[["close"]].iloc[:4]
    scaled = loaded.transform_target(raw)

    np.testing.assert_allclose(loaded.inverse_transform_target(scaled), raw.to_numpy())
    assert loaded.training_start_date == split.train.boundary.start_date
    assert loaded.training_end_date == split.train.boundary.end_date
    with pytest.raises(MarketDataValidationError, match="顺序"):
        loaded.transform_features(
            split.validation.frame,
            feature_columns=tuple(reversed(MULTIVARIATE_FEATURES)),
        )

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["training_interval"]["end_date"] = "2099-01-01"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(MarketDataValidationError, match="完整性校验失败"):
        ScalerManager.load(path)


def test_windows_align_dates_targets_and_cross_boundary_context() -> None:
    data = clean_market_data(_market_frame(20)).data
    split = split_by_time(data)
    scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)

    bundle = build_windows(split, scaler, UNIVARIATE_FEATURES, window_size=3)

    assert bundle.train.X.shape == (11, 3, 1)
    assert bundle.validation.X.shape == (3, 3, 1)
    assert bundle.test.X.shape == (3, 3, 1)
    assert bundle.validation.target_dates[0] == data.iloc[14]["trade_date"].to_datetime64()
    assert bundle.validation.input_start_dates[0] == data.iloc[11]["trade_date"].to_datetime64()
    assert bundle.validation.input_end_dates[0] == data.iloc[13]["trade_date"].to_datetime64()
    assert bundle.validation.previous_close[0] == pytest.approx(data.iloc[13]["close"])
    assert bundle.validation.raw_targets[0] == pytest.approx(data.iloc[14]["close"])
    assert bundle.test.target_dates[0] == data.iloc[17]["trade_date"].to_datetime64()
    assert bundle.test.input_start_dates[0] == data.iloc[14]["trade_date"].to_datetime64()
    assert np.all(bundle.validation.input_end_dates < bundle.validation.target_dates)
    assert_target_dates_disjoint(bundle)


def test_univariate_and_multivariate_window_shapes_and_dataloaders() -> None:
    split = split_by_time(clean_market_data(_market_frame(100)).data)
    univariate_scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)
    multivariate_scaler = ScalerManager.fit(split, MULTIVARIATE_FEATURES)

    univariate = build_windows(split, univariate_scaler, UNIVARIATE_FEATURES, window_size=10)
    multivariate = build_windows(split, multivariate_scaler, MULTIVARIATE_FEATURES, window_size=10)

    assert univariate.train.X.shape == (60, 10, 1)
    assert multivariate.train.X.shape == (60, 10, 6)
    dataset = StockWindowDataset(multivariate.train)
    features, target = dataset[0]
    assert tuple(features.shape) == (10, 6)
    assert tuple(target.shape) == (1,)
    loaders = create_dataloaders(multivariate, batch_size=16)
    batch_features, batch_targets = next(iter(loaders["validation"]))
    assert tuple(batch_features.shape[1:]) == (10, 6)
    assert tuple(batch_targets.shape[1:]) == (1,)


def test_preprocessing_errors_are_clear_for_missing_columns_and_short_training_set() -> None:
    missing = _market_frame(20).drop(columns=["amount"])
    with pytest.raises(MarketDataValidationError, match="缺少标准行情字段.*amount"):
        clean_market_data(missing)

    split = split_by_time(clean_market_data(_market_frame(20)).data)
    scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)
    with pytest.raises(MarketDataValidationError, match="必须大于 window_size"):
        build_windows(split, scaler, UNIVARIATE_FEATURES, window_size=14)
