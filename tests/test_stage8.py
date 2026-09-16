from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.stage8 import (
    REQUIRED_PARAMETER_SPACE,
    Stage8Error,
    _metric_dict,
    load_stage8_config,
    prepare_available_stocks,
    preflight,
)


def test_stage8_config_freezes_protocol_and_parameter_space() -> None:
    config = load_stage8_config("configs/stage8_formal.yaml")

    assert config.split_ratios == (0.7, 0.15, 0.15)
    assert config.random_seeds == (42, 2024, 2026)
    assert len(set(config.random_seeds)) == 3
    assert config.univariate_features == ("close",)
    assert config.multivariate_features == ("open", "high", "low", "close", "vol", "amount")
    assert {name: list(values) for name, values in config.parameter_space.items()} == REQUIRED_PARAMETER_SPACE
    assert len({stock.industry for stock in config.stocks}) == 5


def test_stage8_preflight_accepts_real_formal_data() -> None:
    config = load_stage8_config("configs/stage8_formal.yaml")
    prepared, inventory = prepare_available_stocks(config)

    assert len(prepared) >= config.minimum_stocks
    assert inventory.loc[inventory["status"] == "ready", "industry"].nunique() >= 3
    checked, checked_inventory = preflight(config)
    assert [item.spec.ts_code for item in checked] == [item.spec.ts_code for item in prepared]
    assert set(checked_inventory.loc[checked_inventory["status"] == "ready", "data_source"]) == {
        "BaoStock historical K data"
    }


def test_stage8_metric_recalculation_uses_daily_predictions() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2025-01-02", periods=3, freq="B"),
            "previous_close": [10.0, 11.0, 10.0],
            "actual": [11.0, 10.0, 10.0],
            "predicted": [10.5, 10.5, 10.0],
            "model_name": ["LSTM-univariate"] * 3,
        }
    )

    result = _metric_dict(frame, naive_rmse=1.0)

    assert result["sample_count"] == 3
    assert result["rmse"] == pytest.approx((0.5**2 + 0.5**2) ** 0.5 / 3**0.5)
    assert result["direction_accuracy"] == pytest.approx(1.0)
    assert result["relative_naive_rmse_improvement_pct"] > 0
