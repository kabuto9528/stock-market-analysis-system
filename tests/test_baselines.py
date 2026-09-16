from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.baselines import (
    PREDICTION_COLUMNS,
    ARIMAForecastModel,
    BaselineDataError,
    BaselineFitError,
    BaselineNotFittedError,
    ForecastStrategy,
    MovingAverageForecastModel,
    NaiveForecastModel,
    SimpleExponentialSmoothingModel,
    default_baseline_models,
    run_baseline_suite,
)
from src.data import (
    UNIVARIATE_FEATURES,
    ScalerManager,
    build_windows,
    clean_market_data,
    split_by_time,
)


def _market_frame(values: np.ndarray | list[float], *, start: str = "2024-01-02") -> pd.DataFrame:
    close = np.asarray(values, dtype=float)
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * len(close),
            "trade_date": dates,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "vol": np.arange(len(close), dtype=float) + 100.0,
            "amount": (np.arange(len(close), dtype=float) + 100.0) * close,
        }
    )


def _split(values: np.ndarray | list[float], ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)):
    return split_by_time(clean_market_data(_market_frame(values)).data, ratios)


def test_naive_uses_previous_real_close_on_hand_calculable_sequence() -> None:
    split = _split(list(range(1, 11)))
    targets = split.validation.frame["trade_date"].tolist()

    result = NaiveForecastModel().fit(split.train).predict(split.data, targets)

    assert tuple(result.columns) == PREDICTION_COLUMNS
    assert result["previous_close"].tolist() == pytest.approx([6.0, 7.0])
    assert result["predicted"].tolist() == pytest.approx([6.0, 7.0])
    assert result["actual"].tolist() == pytest.approx([7.0, 8.0])


def test_moving_average_uses_only_values_before_each_target() -> None:
    split = _split(list(range(1, 11)))
    targets = split.validation.frame["trade_date"].tolist()

    result = MovingAverageForecastModel(3).fit(split.train).predict(split.data, targets)

    assert result["predicted"].tolist() == pytest.approx([5.0, 6.0])
    assert result["previous_close"].tolist() == pytest.approx([6.0, 7.0])


def test_static_multi_step_and_rolling_one_step_are_explicitly_different() -> None:
    split = _split(list(range(1, 11)))
    targets = split.validation.frame["trade_date"].tolist()
    model = NaiveForecastModel().fit(split.train)

    rolling = model.predict(split.data, targets, strategy=ForecastStrategy.ROLLING_ONE_STEP)
    static = model.predict(split.data, targets, strategy=ForecastStrategy.STATIC_MULTI_STEP)

    assert rolling["predicted"].tolist() == pytest.approx([6.0, 7.0])
    assert static["predicted"].tolist() == pytest.approx([6.0, 6.0])


def test_ses_and_arima_produce_finite_predictions_on_small_stable_series() -> None:
    values = 20.0 + np.sin(np.arange(40, dtype=float) / 4.0) * 0.2
    split = _split(values, (0.7, 0.15, 0.15))
    targets = split.validation.frame["trade_date"].tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ses = SimpleExponentialSmoothingModel(0.3).fit(split.train).predict(split.data, targets)
        arima = ARIMAForecastModel((1, 0, 0)).fit(split.train).predict(split.data, targets)

    assert np.isfinite(ses["predicted"]).all()
    assert np.isfinite(arima["predicted"]).all()
    assert ses["target_date"].tolist() == arima["target_date"].tolist()


def test_future_target_and_later_values_cannot_change_first_prediction() -> None:
    base = 10.0 + np.arange(40, dtype=float) * 0.05
    split = _split(base, (0.7, 0.15, 0.15))
    first_target = split.validation.frame["trade_date"].iloc[0]
    changed = split.data.copy()
    first_position = split.validation.boundary.start_index
    changed.loc[first_position:, "close"] += 100_000.0

    models = (
        NaiveForecastModel(),
        MovingAverageForecastModel(5),
        SimpleExponentialSmoothingModel(0.25),
        ARIMAForecastModel((1, 0, 0)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for model in models:
            model.fit(split.train)
            original_prediction = model.predict(split.data, [first_target]).loc[0, "predicted"]
            changed_prediction = model.predict(changed, [first_target]).loc[0, "predicted"]
            assert changed_prediction == pytest.approx(original_prediction)


def test_default_six_models_have_identical_fields_and_stage3_target_dates() -> None:
    values = 30.0 + np.arange(50, dtype=float) * 0.1
    split = _split(values, (0.7, 0.15, 0.15))
    scaler = ScalerManager.fit(split, UNIVARIATE_FEATURES)
    windows = build_windows(split, scaler, UNIVARIATE_FEATURES, window_size=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = run_baseline_suite(
            default_baseline_models(arima_order=(1, 0, 0)),
            split.train,
            split.data,
            windows.test.target_dates,
        )

    assert result.model_names == ("Naive", "MA5", "MA10", "MA20", "SES", "ARIMA")
    assert len(result.predictions) == 6
    expected_dates = pd.DatetimeIndex(windows.test.target_dates)
    for model_name, frame in result.predictions.items():
        assert tuple(frame.columns) == PREDICTION_COLUMNS
        assert pd.DatetimeIndex(frame["target_date"]).equals(expected_dates)
        assert frame["model_name"].unique().tolist() == [model_name]
    assert len(result.combined()) == 6 * len(expected_dates)


def test_not_fitted_short_history_non_finite_and_invalid_order_errors_are_clear() -> None:
    split = _split(list(range(1, 11)))
    with pytest.raises(BaselineNotFittedError, match="尚未拟合"):
        NaiveForecastModel().predict(split.data, split.validation.frame["trade_date"].tolist())
    with pytest.raises(BaselineDataError, match="至少需要 20"):
        MovingAverageForecastModel(20).fit(split.train)
    with pytest.raises(BaselineDataError, match="三个非负整数"):
        ARIMAForecastModel((1, -1, 0))

    model = NaiveForecastModel().fit(split.train)
    invalid = split.data.copy()
    invalid.loc[8, "close"] = np.inf
    with pytest.raises(BaselineDataError, match="正负无穷"):
        model.predict(invalid, [invalid.loc[8, "trade_date"]])


def test_arima_fit_failure_is_converted_to_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    split = _split(np.arange(1.0, 31.0), (0.7, 0.15, 0.15))

    def fail_arima(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced statsmodels failure")

    monkeypatch.setattr("src.baselines.arima.ARIMA", fail_arima)
    with pytest.raises(BaselineFitError, match=r"ARIMA\(1, 0, 0\) 拟合失败.*forced"):
        ARIMAForecastModel((1, 0, 0)).fit(split.train)


def test_expected_trading_calendar_detects_missing_date_instead_of_using_false_previous_day() -> None:
    full = clean_market_data(_market_frame(np.arange(1.0, 16.0))).data
    split = split_by_time(full, (0.6, 0.2, 0.2))
    missing_date = full.loc[10, "trade_date"]
    incomplete = full.drop(index=10).reset_index(drop=True)
    target = incomplete.loc[11, "trade_date"]

    with pytest.raises(BaselineDataError, match="缺少预期开市日期"):
        NaiveForecastModel().fit(split.train).predict(
            incomplete,
            [target],
            expected_trading_dates=full["trade_date"].tolist(),
        )
    assert missing_date not in set(incomplete["trade_date"])


def test_expected_trading_calendar_must_cover_training_history() -> None:
    full = clean_market_data(_market_frame(np.arange(1.0, 16.0))).data
    split = split_by_time(full, (0.6, 0.2, 0.2))
    missing_training_date = full.loc[3, "trade_date"]
    incomplete = full.drop(index=3).reset_index(drop=True)
    target = full.loc[9, "trade_date"]

    with pytest.raises(BaselineDataError, match="缺少预期开市日期"):
        NaiveForecastModel().fit(
            split_by_time(incomplete, (0.6, 0.2, 0.2)).train
        ).predict(
            incomplete,
            [target],
            expected_trading_dates=full["trade_date"].tolist(),
        )
    assert missing_training_date not in set(incomplete["trade_date"])


def test_expected_trading_calendar_rejects_incomplete_calendar() -> None:
    full = clean_market_data(_market_frame(np.arange(1.0, 16.0))).data
    split = split_by_time(full, (0.6, 0.2, 0.2))
    target = split.validation.frame["trade_date"].iloc[0]
    incomplete_calendar = full.loc[1:, "trade_date"].tolist()

    with pytest.raises(BaselineDataError, match="必须覆盖训练起止日期"):
        NaiveForecastModel().fit(split.train).predict(
            split.data,
            [target],
            expected_trading_dates=incomplete_calendar,
        )


def test_target_dates_must_be_sorted_unique_and_after_training() -> None:
    split = _split(list(range(1, 11)))
    model = NaiveForecastModel().fit(split.train)
    targets = split.validation.frame["trade_date"].tolist()

    with pytest.raises(BaselineDataError, match="重复"):
        model.predict(split.data, [targets[0], targets[0]])
    with pytest.raises(BaselineDataError, match="严格升序"):
        model.predict(split.data, list(reversed(targets)))
    with pytest.raises(BaselineDataError, match="晚于训练区间"):
        model.predict(split.data, [split.train.frame["trade_date"].iloc[-1]])
