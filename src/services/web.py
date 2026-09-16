"""Streamlit 页面使用的统一业务服务。页面不直接实现数据、训练或评价算法。"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.baselines import default_baseline_models, run_baseline_suite
from src.config import ProjectConfig, load_config
from src.data import (
    MULTIVARIATE_FEATURES,
    UNIVARIATE_FEATURES,
    DataQualityReport,
    MarketDataRepository,
    ScalerManager,
    TushareClient,
    build_windows,
    clean_market_data,
    create_dataloaders,
    generate_quality_report,
    load_market_csv,
    split_by_time,
)
from src.evaluation import ExperimentRecord, ExperimentStore
from src.models import (
    EpochRecord,
    LSTMPredictor,
    LSTMRegressor,
    LSTMTrainer,
    TrainerConfig,
    load_model_artifact,
    save_model_artifact,
    set_random_seed,
)
from src.services.backtesting import UnifiedTestBacktestService


class WebServiceError(RuntimeError):
    """可直接展示给普通用户的中文业务错误。"""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def user_message(self) -> str:
        return f"{self.message}\n\n建议：{self.hint}" if self.hint else self.message


@dataclass(frozen=True)
class MarketContext:
    frame: pd.DataFrame
    ts_code: str
    source: str
    quality: dict[str, object]
    duplicate_rows_removed: int = 0


@dataclass(frozen=True)
class TrainingRequest:
    mode: str = "univariate"
    window_size: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    random_seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.mode not in {"univariate", "multivariate"}:
            raise WebServiceError("模型模式无效。", hint="请选择单变量或多变量 LSTM。")
        integer_fields = (
            "window_size", "hidden_size", "num_layers", "batch_size",
            "max_epochs", "patience", "random_seed",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WebServiceError(f"参数 {field} 必须为正整数。")
        if not 0 <= self.dropout < 1:
            raise WebServiceError("Dropout 必须位于 [0, 1) 区间。")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise WebServiceError("学习率必须为有限正数。")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise WebServiceError("设备参数只能是 auto、cpu 或 cuda。")


@dataclass(frozen=True)
class TrainingSummary:
    experiment_id: str
    ts_code: str
    mode: str
    feature_columns: tuple[str, ...]
    best_epoch: int
    best_validation_loss: float
    elapsed_seconds: float
    stopped_early: bool
    device: str
    history: pd.DataFrame
    metrics: pd.DataFrame
    next_prediction: pd.DataFrame


@dataclass(frozen=True)
class ExperimentDashboard:
    record: ExperimentRecord
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    history: pd.DataFrame
    next_prediction: pd.DataFrame
    directory: Path


class MarketDataWebService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.repository = MarketDataRepository(config.paths.cache_dir)

    @staticmethod
    def _context(frame: pd.DataFrame, source: str, report: DataQualityReport | None = None,
                 duplicate_rows_removed: int = 0) -> MarketContext:
        if frame.empty:
            raise WebServiceError("行情数据为空。", hint="请调整日期范围或检查 CSV 内容。")
        ts_codes = frame["ts_code"].astype(str).unique().tolist()
        if len(ts_codes) != 1:
            raise WebServiceError("一次只能加载一只股票的数据。")
        quality = (report or generate_quality_report(frame)).to_dict()
        return MarketContext(
            frame=frame.copy(), ts_code=ts_codes[0], source=source,
            quality=quality, duplicate_rows_removed=duplicate_rows_removed,
        )

    def import_csv_bytes(self, file_name: str, payload: bytes,
                         expected_ts_code: str | None = None,
                         *, save_cache: bool = False) -> MarketContext:
        if not payload:
            raise WebServiceError("上传的 CSV 文件为空。")
        suffix = Path(file_name).suffix.lower()
        if suffix != ".csv":
            raise WebServiceError("仅支持 CSV 文件。", hint="请将行情文件另存为 UTF-8 CSV。")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            loaded = load_market_csv(temp_path, expected_ts_code=expected_ts_code or None)
            context = self._context(
                loaded.data, f"CSV：{file_name}", loaded.quality_report,
                loaded.duplicate_rows_removed,
            )
            if save_cache:
                self.repository.overwrite(context.frame, quality_report=generate_quality_report(context.frame))
            return context
        except WebServiceError:
            raise
        except Exception as exc:
            raise WebServiceError(
                f"CSV 导入失败：{exc}",
                hint="确认包含 ts_code、trade_date、open、high、low、close、vol、amount 字段。",
            ) from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def fetch_tushare(self, ts_code: str, start_date: date, end_date: date,
                      *, save_cache: bool = True) -> MarketContext:
        try:
            client = TushareClient(self.config.tushare_token)
            frame = client.fetch_daily(ts_code, start_date, end_date)
            report = generate_quality_report(frame)
            if save_cache:
                self.repository.incremental_update(frame, quality_report=report)
            return self._context(frame, "Tushare", report)
        except Exception as exc:
            text = str(exc)
            if "Token" in text or "TOKEN" in text or "token" in text:
                raise WebServiceError(
                    "未配置可用的 Tushare Token。",
                    hint="在 .env 中设置 TUSHARE_TOKEN，或改用 CSV / 本地缓存离线演示。",
                ) from exc
            raise WebServiceError(
                f"Tushare 获取失败：{text}",
                hint="检查股票代码、日期范围、网络和接口权限；也可使用本地 CSV。",
            ) from exc

    def list_cached_stocks(self) -> tuple[str, ...]:
        codes: list[str] = []
        for manifest in sorted(self.config.paths.cache_dir.glob("*.cache.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                code = str(payload.get("ts_code", "")).strip()
                if code and self.repository.exists(code):
                    codes.append(code)
            except (OSError, ValueError, TypeError):
                continue
        return tuple(dict.fromkeys(codes))

    def load_cache(self, ts_code: str) -> MarketContext:
        try:
            frame = self.repository.load(ts_code)
            return self._context(frame, "本地缓存")
        except Exception as exc:
            raise WebServiceError(
                f"本地缓存加载失败：{exc}", hint="请先从 Tushare 获取或上传 CSV 并保存缓存。"
            ) from exc

    def load_offline_demo(self) -> MarketContext:
        demo = self.config.project_root / "tests" / "fixtures" / "stage6_small_market.csv"
        try:
            loaded = load_market_csv(demo, expected_ts_code="600000.SH")
            return self._context(
                loaded.data, "内置固定小型测试样例（非正式实验）",
                loaded.quality_report, loaded.duplicate_rows_removed,
            )
        except Exception as exc:
            raise WebServiceError("离线演示样例不可用。", hint=str(exc)) from exc

    @staticmethod
    def quality_table(quality: dict[str, object]) -> pd.DataFrame:
        rows = [
            ("总行数", quality.get("total_rows")),
            ("起始日期", quality.get("start_date")),
            ("结束日期", quality.get("end_date")),
            ("重复日期数", quality.get("duplicate_dates")),
            ("负成交量", quality.get("negative_volume")),
            ("负成交额", quality.get("negative_amount")),
            ("质量检查", "通过" if quality.get("passed") else "存在警告"),
        ]
        warnings = quality.get("warnings") or []
        rows.append(("警告", "；".join(map(str, warnings)) if warnings else "无"))
        return pd.DataFrame(rows, columns=["项目", "结果"])


class AnalysisWebService:
    @staticmethod
    def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
        try:
            return clean_market_data(frame).data
        except Exception as exc:
            raise WebServiceError(f"数据分析前清洗失败：{exc}") from exc

    def figures(self, frame: pd.DataFrame) -> dict[str, go.Figure]:
        data = self._prepared(frame)
        figures: dict[str, go.Figure] = {}
        candle = go.Figure(go.Candlestick(
            x=data["trade_date"], open=data["open"], high=data["high"],
            low=data["low"], close=data["close"], name="K线",
        ))
        candle.update_layout(title="股票日线 K 线图", xaxis_title="交易日期", yaxis_title="价格",
                             xaxis_rangeslider_visible=False)
        figures["kline"] = candle

        ma = data[["trade_date", "close"]].copy()
        for window in (5, 10, 20):
            ma[f"MA{window}"] = ma["close"].rolling(window).mean()
        close = go.Figure()
        close.add_trace(go.Scatter(x=ma["trade_date"], y=ma["close"], name="收盘价"))
        for window in (5, 10, 20):
            close.add_trace(go.Scatter(x=ma["trade_date"], y=ma[f"MA{window}"], name=f"MA{window}"))
        close.update_layout(title="收盘价与移动平均线", xaxis_title="交易日期", yaxis_title="价格")
        figures["close_ma"] = close

        volume = px.bar(data, x="trade_date", y="vol", title="成交量走势",
                        labels={"trade_date": "交易日期", "vol": "成交量"})
        figures["volume"] = volume

        returns = data["close"].pct_change().dropna()
        distribution = px.histogram(x=returns, nbins=40, title="日收益率分布",
                                    labels={"x": "日收益率", "count": "频数"})
        distribution.add_vline(x=0, line_dash="dash", annotation_text="零收益")
        distribution.update_layout(xaxis_title="日收益率", yaxis_title="频数")
        figures["returns"] = distribution

        columns = ["open", "high", "low", "close", "vol", "amount"]
        corr = data[columns].corr()
        heatmap = go.Figure(go.Heatmap(
            z=corr.to_numpy(), x=columns, y=columns, zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=True, text=np.round(corr.to_numpy(), 3),
            texttemplate="%{text}", colorbar_title="相关系数",
        ))
        heatmap.update_layout(title="行情特征相关性热力图", xaxis_title="特征", yaxis_title="特征")
        figures["correlation"] = heatmap
        return figures


class TrainingWebService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.store = ExperimentStore(config.paths.experiments_dir)

    @staticmethod
    def _history_payload(result) -> dict[str, object]:
        return {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "stopped_early": result.stopped_early,
            "total_elapsed_seconds": result.total_elapsed_seconds,
            "device": result.device,
            "history": [asdict(item) for item in result.history],
            "validation_only_for_selection_and_early_stopping": True,
            "test_samples_used_for_training_or_early_stopping": 0,
        }

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)

    def train(self, frame: pd.DataFrame, request: TrainingRequest,
              *, progress_callback: Callable[[EpochRecord], None] | None = None) -> TrainingSummary:
        request.validate()
        started = time.perf_counter()
        features = UNIVARIATE_FEATURES if request.mode == "univariate" else MULTIVARIATE_FEATURES
        model_label = "LSTM-单变量" if request.mode == "univariate" else "LSTM-多变量"
        try:
            cleaned = clean_market_data(frame)
            split = split_by_time(cleaned.data, ratios=self.config.split_ratios)
            scaler = ScalerManager.fit(split, features, target_column="close")
            windows = build_windows(
                split, scaler, features, window_size=request.window_size, target_column="close"
            )
            set_random_seed(request.random_seed)
            loaders = create_dataloaders(
                windows, batch_size=request.batch_size, train_shuffle=False, seed=request.random_seed
            )
            model = LSTMRegressor(
                input_size=len(features), hidden_size=request.hidden_size,
                num_layers=request.num_layers, dropout=request.dropout,
            )
            trainer = LSTMTrainer(
                TrainerConfig(
                    learning_rate=request.learning_rate,
                    max_epochs=request.max_epochs,
                    early_stopping_patience=request.patience,
                ),
                device=request.device,
            )
            result = trainer.fit(
                model, loaders["train"], loaders["validation"],
                epoch_callback=progress_callback,
            )
            predictor = LSTMPredictor(
                model, scaler, feature_columns=features, window_size=request.window_size,
                device=request.device, model_name=model_label,
            )
            lstm_prediction = predictor.predict_samples(windows.test, batch_size=request.batch_size)
            target_dates = pd.DatetimeIndex(windows.test.target_dates)
            baseline_models = default_baseline_models()
            if request.mode == "multivariate":
                # 多变量 LSTM 属于特征消融；仅保留 Naive 以定义相对提升率，
                # 不把 MA/SES/ARIMA 混入多变量实验的主比较产物。
                baseline_models = baseline_models[:1]
            baseline_result = run_baseline_suite(
                baseline_models, split.train, split.data, target_dates,
                expected_trading_dates=split.data["trade_date"],
            )
            predictions = dict(baseline_result.predictions)
            predictions[model_label] = lstm_prediction
            backtest = UnifiedTestBacktestService(target_dates).run(predictions)

            last_date = pd.Timestamp(cleaned.data["trade_date"].iloc[-1])
            next_date = last_date + pd.offsets.BDay(1)
            next_prediction = predictor.predict_next_trading_day(cleaned.data, target_date=next_date)
            next_prediction["date_note"] = "按下一工作日估算，节假日请以交易所日历为准"

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_code = "".join(c if c.isalnum() else "_" for c in str(cleaned.data.iloc[0]["ts_code"]))
            experiment_id = f"web_{safe_code}_{request.mode}_{stamp}_{uuid4().hex[:6]}"
            directory = self.store.experiment_directory(experiment_id, create=True)
            model_path = directory / "selected_lstm.pt"
            scaler_path = directory / "selected_lstm.scaler.json"
            history_path = directory / "training_history.json"
            next_path = directory / "next_prediction.csv"
            save_model_artifact(
                model_path, model, feature_columns=features, target_column="close",
                window_size=request.window_size, random_seed=request.random_seed,
                best_epoch=result.best_epoch, best_validation_loss=result.best_validation_loss,
            )
            scaler.save(scaler_path)
            self._write_json(history_path, self._history_payload(result))
            next_prediction.to_csv(next_path, index=False, encoding="utf-8-sig")

            elapsed = time.perf_counter() - started
            model_params = {
                name: baseline_model.get_params()
                for name, baseline_model in zip(baseline_result.model_names, baseline_models)
            }
            model_params[model_label] = {
                "input_size": len(features), "hidden_size": request.hidden_size,
                "num_layers": request.num_layers, "dropout": request.dropout,
                "learning_rate": request.learning_rate, "batch_size": request.batch_size,
                "max_epochs": request.max_epochs, "early_stopping_patience": request.patience,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
            }
            record = ExperimentRecord(
                experiment_id=experiment_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                result_kind="test",
                status="completed",
                ts_code=str(cleaned.data.iloc[0]["ts_code"]),
                data_range={
                    "start_date": pd.Timestamp(cleaned.data["trade_date"].iloc[0]).date().isoformat(),
                    "end_date": pd.Timestamp(cleaned.data["trade_date"].iloc[-1]).date().isoformat(),
                },
                split_boundaries=split.boundaries_dict(),
                feature_columns=tuple(features),
                window_size=request.window_size,
                model_parameters=model_params,
                random_seed=request.random_seed,
                metrics={},
                elapsed_seconds={"LSTM_training": result.total_elapsed_seconds, "total": elapsed},
                artifact_paths={
                    "lstm_model": str(model_path.relative_to(self.store.root)),
                    "scaler": str(scaler_path.relative_to(self.store.root)),
                    "training_history": str(history_path.relative_to(self.store.root)),
                    "next_prediction": str(next_path.relative_to(self.store.root)),
                },
                validation_usage=("early_stopping",),
                notes=(
                    "交互训练结果标记为 test，不是阶段 8 正式实验结论。",
                    "多变量 LSTM 仅用于特征消融展示，不参与传统单变量模型主排名。",
                    "R² 仅作价格水平拟合度参考，不用于模型排名。",
                ),
            )
            saved = self.store.save(record, backtest)
            history = pd.DataFrame([asdict(item) for item in result.history])
            metrics = pd.DataFrame(
                [{"model_name": name, **values} for name, values in saved.metrics.items()]
            )
            return TrainingSummary(
                experiment_id=experiment_id, ts_code=saved.ts_code, mode=request.mode,
                feature_columns=tuple(features), best_epoch=result.best_epoch,
                best_validation_loss=result.best_validation_loss,
                elapsed_seconds=elapsed, stopped_early=result.stopped_early,
                device=result.device, history=history, metrics=metrics,
                next_prediction=next_prediction,
            )
        except WebServiceError:
            raise
        except Exception as exc:
            text = str(exc)
            hint = "减少窗口或使用更多历史数据后重试。" if any(
                key in text for key in ("样本", "window_size", "DataLoader", "至少")
            ) else "检查数据质量、参数范围和设备配置后重试。"
            raise WebServiceError(f"模型训练失败：{text}", hint=hint) from exc

    def load_pretrained_prediction(self, experiment_id: str, frame: pd.DataFrame,
                                   *, device: str = "auto") -> pd.DataFrame:
        directory = self.store.experiment_directory(experiment_id)
        try:
            record = self.store.load(experiment_id)
            current_codes = frame["ts_code"].astype(str).drop_duplicates().tolist()
            if current_codes != [record.ts_code]:
                raise WebServiceError(
                    f"模型股票为 {record.ts_code}，当前数据股票为 {current_codes or ['未知']}，配置不匹配。",
                    hint="加载与当前股票一致的缓存/CSV，或选择对应股票的实验。",
                )
        except WebServiceError:
            raise
        except Exception as exc:
            raise WebServiceError(f"实验配置读取失败：{exc}") from exc
        model_path = directory / "selected_lstm.pt"
        scaler_path = directory / "selected_lstm.scaler.json"
        if not model_path.is_file() or not scaler_path.is_file():
            raise WebServiceError("所选实验没有可加载的预训练模型或 Scaler。")
        try:
            artifact = load_model_artifact(model_path, scaler_path, map_location="cpu")
            cleaned = clean_market_data(frame)
            predictor = LSTMPredictor(
                artifact.model, artifact.scaler,
                feature_columns=artifact.metadata.feature_columns,
                target_column=artifact.metadata.target_column,
                window_size=artifact.metadata.window_size,
                device=device, model_name="LSTM",
            )
            next_date = pd.Timestamp(cleaned.data["trade_date"].iloc[-1]) + pd.offsets.BDay(1)
            output = predictor.predict_next_trading_day(cleaned.data, target_date=next_date)
            output["date_note"] = "按下一工作日估算，节假日请以交易所日历为准"
            return output
        except Exception as exc:
            raise WebServiceError(
                f"预训练模型加载或预测失败：{exc}",
                hint="确认当前数据的股票、特征列和历史长度与模型配置匹配。",
            ) from exc


class ExperimentWebService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.store = ExperimentStore(config.paths.experiments_dir)

    def list_experiments(self) -> tuple[str, ...]:
        items: list[tuple[float, str]] = []
        for path in self.config.paths.experiments_dir.iterdir():
            record = path / "experiment_record.json"
            if path.is_dir() and record.is_file():
                items.append((record.stat().st_mtime, path.name))
        return tuple(name for _, name in sorted(items, reverse=True))

    def stamp(self, experiment_id: str) -> int:
        path = self.store.experiment_directory(experiment_id) / "experiment_record.json"
        return path.stat().st_mtime_ns if path.is_file() else 0

    def load(self, experiment_id: str) -> ExperimentDashboard:
        try:
            record = self.store.load(experiment_id)
            directory = self.store.experiment_directory(experiment_id)
            comparison = pd.read_csv(directory / "model_comparison.csv")
            predictions = pd.read_csv(directory / "daily_predictions.csv", parse_dates=["target_date"])
            history_path = directory / "training_history.json"
            history = pd.DataFrame()
            if history_path.is_file():
                payload = json.loads(history_path.read_text(encoding="utf-8"))
                history = pd.DataFrame(payload.get("history", []))
            next_path = directory / "next_prediction.csv"
            next_prediction = pd.read_csv(next_path) if next_path.is_file() else pd.DataFrame()
            return ExperimentDashboard(record, comparison, predictions, history, next_prediction, directory)
        except Exception as exc:
            raise WebServiceError(f"实验结果加载失败：{exc}", hint="请选择完整且状态为 completed 的实验。") from exc

    def combine_compatible(
        self, primary: ExperimentDashboard, secondary: ExperimentDashboard | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """合并兼容实验中的 LSTM 行；传统基线只保留一份。"""

        comparison = primary.comparison.copy()
        predictions = primary.predictions.copy()
        if secondary is None or secondary.record.experiment_id == primary.record.experiment_id:
            return comparison, predictions
        if (primary.record.ts_code != secondary.record.ts_code or
                primary.record.data_range != secondary.record.data_range or
                primary.record.split_boundaries != secondary.record.split_boundaries):
            raise WebServiceError(
                "两个实验的数据区间或时间划分不兼容，不能合并比较。",
                hint="请选择同一股票、同一数据区间及同一 70/15/15 划分的单变量与多变量实验。",
            )
        left_dates = set(pd.to_datetime(primary.predictions["target_date"]).dt.normalize())
        right_dates = set(pd.to_datetime(secondary.predictions["target_date"]).dt.normalize())
        if left_dates != right_dates:
            raise WebServiceError("两个实验的测试目标日期不一致，不能直接比较。")
        existing = set(comparison["model_name"].astype(str))
        extra_rows = secondary.comparison[
            ~secondary.comparison["model_name"].astype(str).isin(existing)
        ]
        extra_predictions = secondary.predictions[
            secondary.predictions["model_name"].astype(str).isin(extra_rows["model_name"].astype(str))
        ]
        if not extra_rows.empty:
            comparison = pd.concat([comparison, extra_rows], ignore_index=True)
            predictions = pd.concat([predictions, extra_predictions], ignore_index=True)
        return comparison, predictions

    @staticmethod
    def loss_figure(history: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if not history.empty:
            fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], name="训练损失"))
            fig.add_trace(go.Scatter(x=history["epoch"], y=history["validation_loss"], name="验证损失"))
        fig.update_layout(title="训练与验证损失", xaxis_title="训练轮次（Epoch）", yaxis_title="MSE 损失")
        return fig

    @staticmethod
    def prediction_figure(predictions: pd.DataFrame, model_name: str) -> go.Figure:
        data = predictions[predictions["model_name"] == model_name].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["target_date"], y=data["actual"], name="真实收盘价"))
        fig.add_trace(go.Scatter(x=data["target_date"], y=data["predicted"], name="预测收盘价"))
        fig.update_layout(title=f"{model_name}：测试集真实值与预测值", xaxis_title="目标交易日", yaxis_title="收盘价")
        return fig

    @staticmethod
    def error_figure(predictions: pd.DataFrame, model_name: str) -> go.Figure:
        data = predictions[predictions["model_name"] == model_name].copy()
        data["error"] = data["predicted"] - data["actual"]
        fig = px.bar(data, x="target_date", y="error", title=f"{model_name}：逐日预测误差",
                     labels={"target_date": "目标交易日", "error": "预测值 - 真实值"})
        fig.add_hline(y=0, line_dash="dash")
        return fig

    @staticmethod
    def metric_figures(comparison: pd.DataFrame) -> dict[str, go.Figure]:
        labels = {
            "rmse": ("RMSE 对比（越低越好）", "RMSE"),
            "mae": ("MAE 对比（越低越好）", "MAE"),
            "direction_accuracy": ("方向准确率 DA 对比（越高越好）", "DA"),
            "relative_naive_rmse_improvement_pct": ("相对 Naive 的 RMSE 提升率", "提升率（%）"),
        }
        figures: dict[str, go.Figure] = {}
        for column, (title, y_title) in labels.items():
            fig = px.bar(comparison, x="model_name", y=column, color="model_name", title=title,
                         labels={"model_name": "模型", column: y_title})
            fig.update_layout(showlegend=False)
            figures[column] = fig
        return figures

    @staticmethod
    def csv_bytes(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8-sig")

    @staticmethod
    def json_bytes(payload: object) -> bytes:
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class WebApplicationService:
    """供 Streamlit 页面缓存为 resource 的服务容器。"""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or load_config()
        self.data = MarketDataWebService(self.config)
        self.analysis = AnalysisWebService()
        self.training = TrainingWebService(self.config)
        self.experiments = ExperimentWebService(self.config)

