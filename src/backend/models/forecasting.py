"""Forecast demand without using future information.

The data is split into three periods:

1. model selection: choose the forecasting method and LightGBM settings;
2. policy calibration: measure forecast errors for inventory settings; and
3. final evaluation: report the final result once all choices are fixed.

A forecast for today may use demand observed through yesterday, never today's
actual demand.
"""

import json
import logging
import os

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from features.feature_engineering import create_features, get_feature_cols

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _config = json.load(f)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_PARAMS_DIR = os.path.join(_ROOT, _config["paths"]["params"])

_DATE_COL = _config["data"]["date_col"]
_SKU_COL = _config["data"]["sku_col"]
_TARGET_COL = _config["data"]["target_col"]
_VALIDATION_START = pd.Timestamp(_config["data"]["validation_start"])
_VALIDATION_END = pd.Timestamp(
    _config["data"].get(
        "validation_end",
        pd.Timestamp(_config["data"]["final_test_start"]) - pd.Timedelta(days=1),
    )
)
_FINAL_TEST_START = pd.Timestamp(_config["data"]["final_test_start"])

_N_TRIALS = _config["forecasting"]["optuna_n_trials"]
_OPTUNA_SEED = _config["forecasting"]["optuna_seed"]
_FIXED_PARAMS = _config["forecasting"]["lgbm_fixed_params"]
_SEASON_LENGTH = _config["forecasting"]["season_length"]
_FOLD_DAYS = _config["backtest"]["validation_fold_days"]
_N_FOLDS = _config["backtest"]["validation_n_folds"]
_SELECTION_DAYS = _FOLD_DAYS * _N_FOLDS

# Read the three forecasting windows from config. The fallback values only
# support older config files; this project defines every boundary explicitly.
_MODEL_SELECTION_START = pd.Timestamp(
    _config["data"].get("model_selection_start", _VALIDATION_START)
)
_configured_calibration_start = _config["data"].get("policy_calibration_start")
_configured_selection_end = _config["data"].get("model_selection_end")
if _configured_selection_end is not None:
    _MODEL_SELECTION_END = pd.Timestamp(_configured_selection_end)
elif _configured_calibration_start is not None:
    _MODEL_SELECTION_END = (
        pd.Timestamp(_configured_calibration_start) - pd.Timedelta(days=1)
    )
else:
    _MODEL_SELECTION_END = (
        _MODEL_SELECTION_START + pd.Timedelta(days=_SELECTION_DAYS - 1)
    )

_POLICY_CALIBRATION_START = pd.Timestamp(
    _configured_calibration_start
    if _configured_calibration_start is not None
    else _MODEL_SELECTION_END + pd.Timedelta(days=1)
)
_POLICY_CALIBRATION_END = pd.Timestamp(
    _config["data"].get("policy_calibration_end", _VALIDATION_END)
)

if not (
    _MODEL_SELECTION_START
    <= _MODEL_SELECTION_END
    < _POLICY_CALIBRATION_START
    <= _POLICY_CALIBRATION_END
    < _FINAL_TEST_START
):
    raise ValueError(
        "Forecast windows must be ordered and disjoint: "
        "model_selection_start <= model_selection_end < "
        "policy_calibration_start <= policy_calibration_end < final_test_start. "
        f"Received selection={_MODEL_SELECTION_START.date()}.."
        f"{_MODEL_SELECTION_END.date()}, calibration="
        f"{_POLICY_CALIBRATION_START.date()}..{_POLICY_CALIBRATION_END.date()}, "
        f"final_test_start={_FINAL_TEST_START.date()}."
    )

logger = logging.getLogger(__name__)


def _set_sku_categories(
    feature_rows: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:
    """Give train and test rows the same SKU category list."""
    prepared_features = feature_rows.copy()
    prepared_features[_SKU_COL] = pd.Categorical(
        prepared_features[_SKU_COL],
        categories=categories,
    )
    return prepared_features


def _wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate weighted absolute percentage error."""
    actual_demand = np.asarray(y_true, dtype=float)
    forecast_demand = np.asarray(y_pred, dtype=float)
    total_demand = np.abs(actual_demand).sum()

    if total_demand == 0:
        return np.nan

    total_absolute_error = np.abs(actual_demand - forecast_demand).sum()
    return float(total_absolute_error / total_demand)


def _nonnegative_forecast(values: pd.Series | np.ndarray) -> np.ndarray:
    """Replace impossible negative demand forecasts with zero."""
    return np.clip(np.asarray(values, dtype=float), a_min=0.0, a_max=None)


def _window_dates(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label: str,
) -> pd.DatetimeIndex:
    """Return sorted dates inside an inclusive start/end window."""
    date_is_in_window = (
        (frame[_DATE_COL] >= start)
        & (frame[_DATE_COL] <= end)
    )
    dates_in_window = frame.loc[date_is_in_window, _DATE_COL].unique()
    dates = pd.DatetimeIndex(sorted(dates_in_window))

    if dates.empty:
        raise ValueError(
            f"No {label} dates found in configured window "
            f"{start.date()}..{end.date()}."
        )
    return dates


def _date_folds(feature_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split model-selection dates into contiguous validation folds."""
    selection_dates = _window_dates(
        feature_df,
        _MODEL_SELECTION_START,
        _MODEL_SELECTION_END,
        label="model-selection",
    )
    if len(selection_dates) < _SELECTION_DAYS:
        raise ValueError(
            f"Need {_N_FOLDS} x {_FOLD_DAYS} model-selection days; "
            f"found {len(selection_dates)} in "
            f"{_MODEL_SELECTION_START.date()}..{_MODEL_SELECTION_END.date()}."
        )
    selection_dates = selection_dates[-_SELECTION_DAYS:]

    folds = []
    date_chunks = np.array_split(selection_dates, _N_FOLDS)
    for date_chunk in date_chunks:
        if len(date_chunk) > 0:
            fold_start = date_chunk[0]
            fold_end = date_chunk[-1]
            folds.append((fold_start, fold_end))

    return folds


def _selection_validation_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the same selection dates for every candidate model."""
    dates = _window_dates(
        frame,
        _MODEL_SELECTION_START,
        _MODEL_SELECTION_END,
        label="model-selection",
    )
    if len(dates) < _SELECTION_DAYS:
        raise ValueError(
            f"Need {_SELECTION_DAYS} model-selection dates; found {len(dates)}."
        )
    return dates[-_SELECTION_DAYS:]


def _policy_calibration_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return dates used to measure policy-calibration forecast errors."""
    return _window_dates(
        frame,
        _POLICY_CALIBRATION_START,
        _POLICY_CALIBRATION_END,
        label="policy-calibration",
    )


def _lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        **_FIXED_PARAMS,
    }


def tune_lgbm(feature_df: pd.DataFrame) -> dict:
    """Choose LightGBM settings using only model-selection folds."""
    feature_columns = get_feature_cols(feature_df)
    sku_categories = sorted(feature_df[_SKU_COL].unique().tolist())
    folds = _date_folds(feature_df)

    def objective(trial: optuna.Trial) -> float:
        """Return average validation WAPE for one set of trial settings."""
        fold_scores = []
        params = _lgbm_params(trial)

        for validation_start, validation_end in folds:
            training_rows = feature_df[_DATE_COL] < validation_start
            validation_rows = (
                (feature_df[_DATE_COL] >= validation_start)
                & (feature_df[_DATE_COL] <= validation_end)
            )
            training_data = feature_df.loc[training_rows]
            validation_data = feature_df.loc[validation_rows]

            training_features = _set_sku_categories(
                training_data[feature_columns],
                sku_categories,
            )
            validation_features = _set_sku_categories(
                validation_data[feature_columns],
                sku_categories,
            )

            model = lgb.LGBMRegressor(**params)
            model.fit(
                training_features,
                training_data[_TARGET_COL],
                eval_set=[
                    (
                        validation_features,
                        validation_data[_TARGET_COL],
                    )
                ],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            validation_forecast = _nonnegative_forecast(
                model.predict(validation_features)
            )
            fold_wape = _wape(
                validation_data[_TARGET_COL],
                validation_forecast,
            )
            fold_scores.append(fold_wape)

        return float(np.nanmean(fold_scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_OPTUNA_SEED)
    )
    study.optimize(objective, n_trials=_N_TRIALS)
    logger.info("Best model-selection WAPE: %.4f", study.best_value)
    best_params = {**study.best_params, **_FIXED_PARAMS}

    # Choose the number of trees from the same folds. This value is then fixed
    # for calibration and final forecasting.
    best_iterations = []
    for validation_start, validation_end in folds:
        training_rows = feature_df[_DATE_COL] < validation_start
        validation_rows = (
            (feature_df[_DATE_COL] >= validation_start)
            & (feature_df[_DATE_COL] <= validation_end)
        )
        training_data = feature_df.loc[training_rows]
        validation_data = feature_df.loc[validation_rows]

        training_features = _set_sku_categories(
            training_data[feature_columns],
            sku_categories,
        )
        validation_features = _set_sku_categories(
            validation_data[feature_columns],
            sku_categories,
        )

        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            training_features,
            training_data[_TARGET_COL],
            eval_set=[
                (
                    validation_features,
                    validation_data[_TARGET_COL],
                )
            ],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        best_iteration = model.best_iteration_ or best_params["n_estimators"]
        best_iterations.append(best_iteration)

    best_params["n_estimators"] = int(np.median(best_iterations))
    return best_params


def train_lgbm(
    feature_df: pd.DataFrame,
    params: dict,
    train_before: pd.Timestamp | str | None = None,
) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Fit one LightGBM model using rows before the selected cutoff."""
    if train_before is None:
        cutoff_date = _FINAL_TEST_START
    else:
        cutoff_date = pd.Timestamp(train_before)

    feature_columns = get_feature_cols(feature_df)
    training_rows = feature_df[_DATE_COL] < cutoff_date
    training_data = feature_df.loc[training_rows]

    if training_data.empty:
        raise ValueError(
            f"Cannot train LightGBM: no observations before "
            f"{cutoff_date.date()}."
        )

    sku_categories = sorted(feature_df[_SKU_COL].unique().tolist())
    training_features = _set_sku_categories(
        training_data[feature_columns],
        sku_categories,
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(training_features, training_data[_TARGET_COL])
    return model, sku_categories


def _selection_forecast_lgbm(
    feature_df: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Create out-of-fold LightGBM forecasts for model comparison."""
    feature_columns = get_feature_cols(feature_df)
    sku_categories = sorted(feature_df[_SKU_COL].unique().tolist())
    predictions = []

    for selection_start, selection_end in _date_folds(feature_df):
        training_rows = feature_df[_DATE_COL] < selection_start
        selection_rows = (
            (feature_df[_DATE_COL] >= selection_start)
            & (feature_df[_DATE_COL] <= selection_end)
        )
        training_data = feature_df.loc[training_rows]
        selection_data = feature_df.loc[selection_rows].copy()

        training_features = _set_sku_categories(
            training_data[feature_columns],
            sku_categories,
        )
        selection_features = _set_sku_categories(
            selection_data[feature_columns],
            sku_categories,
        )

        model = lgb.LGBMRegressor(**params)
        model.fit(
            training_features,
            training_data[_TARGET_COL],
        )
        selection_data["forecast"] = _nonnegative_forecast(
            model.predict(selection_features)
        )
        selection_data["model"] = (
            "LightGBM (model selection OOF; rolling one-step)"
        )

        output_columns = [
            _SKU_COL,
            _DATE_COL,
            _TARGET_COL,
            "forecast",
            "model",
        ]
        predictions.append(
            selection_data[output_columns]
        )

    return pd.concat(predictions, ignore_index=True)


def calibration_forecast_ax_ay(
    df: pd.DataFrame, classes: list[str], params: dict
) -> pd.DataFrame:
    """Forecast calibration dates with settings fixed after selection."""
    if not params:
        raise ValueError(
            "Policy-calibration forecasting requires frozen LightGBM parameters."
        )

    class_is_selected = df["class"].isin(classes)
    model_data = df.loc[class_is_selected].copy()
    demand_columns = [_SKU_COL, _DATE_COL, _TARGET_COL]
    feature_df = create_features(model_data[demand_columns])
    feature_columns = get_feature_cols(feature_df)

    # The model sees no actual demand from the calibration period when fitted.
    model, sku_categories = train_lgbm(
        feature_df,
        params,
        train_before=_POLICY_CALIBRATION_START,
    )

    calibration_dates = _policy_calibration_dates(feature_df)
    is_calibration_date = feature_df[_DATE_COL].isin(calibration_dates)
    calibration_data = feature_df.loc[is_calibration_date].copy()
    calibration_features = _set_sku_categories(
        calibration_data[feature_columns],
        sku_categories,
    )
    calibration_data["forecast"] = _nonnegative_forecast(
        model.predict(calibration_features)
    )
    calibration_data["model"] = (
        "LightGBM (policy calibration; estimator frozen before window; "
        "rolling one-step)"
    )

    output_columns = [
        _SKU_COL,
        _DATE_COL,
        _TARGET_COL,
        "forecast",
        "model",
    ]
    return calibration_data[output_columns].reset_index(drop=True)


def validation_forecast_ax_ay(
    df: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune LightGBM, then return selection and calibration forecasts."""
    class_is_selected = df["class"].isin(classes)
    model_data = df.loc[class_is_selected].copy()
    demand_columns = [_SKU_COL, _DATE_COL, _TARGET_COL]
    feature_df = create_features(model_data[demand_columns])

    params = tune_lgbm(feature_df)
    selection_forecast = _selection_forecast_lgbm(feature_df, params)
    calibration_forecast = calibration_forecast_ax_ay(df, classes, params)
    return selection_forecast, calibration_forecast, params


def compare_strategic_candidates(
    lgbm_selection: pd.DataFrame, df: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Choose LightGBM or a simple baseline on the same selection rows."""
    selection_dates = pd.DatetimeIndex(
        sorted(lgbm_selection[_DATE_COL].unique())
    )
    if selection_dates.empty:
        raise ValueError("Strategic model comparison received no selection rows.")

    if (
        selection_dates.min() < _MODEL_SELECTION_START
        or selection_dates.max() > _MODEL_SELECTION_END
    ):
        raise ValueError(
            "Strategic model comparison may use only configured model-selection rows."
        )

    class_is_selected = df["class"].isin(classes)
    baseline_data = _rolling_baseline_frame(df.loc[class_is_selected])
    date_is_selected = baseline_data[_DATE_COL].isin(selection_dates)
    baseline_data = baseline_data.loc[date_is_selected].copy()

    candidate_forecasts = {}
    comparison_columns = [_SKU_COL, _DATE_COL, _TARGET_COL, "forecast"]
    candidate_forecasts["LightGBM"] = lgbm_selection[
        comparison_columns
    ].copy()

    for model_name in ["Naive", "SeasonalNaive", "HistoricMean"]:
        baseline_columns = [_SKU_COL, _DATE_COL, _TARGET_COL, model_name]
        forecast_frame = baseline_data[baseline_columns].rename(
            columns={model_name: "forecast"}
        )
        candidate_forecasts[model_name] = forecast_frame

    metric_rows = []
    for model_name, forecast_frame in candidate_forecasts.items():
        rows_with_forecast = forecast_frame.dropna(subset=["forecast"])
        forecast_error = (
            rows_with_forecast["forecast"] - rows_with_forecast[_TARGET_COL]
        )
        metric_rows.append(
            {
                "model": model_name,
                "MAE": forecast_error.abs().mean(),
                "RMSE": np.sqrt((forecast_error**2).mean()),
                "WAPE": _wape(
                    rows_with_forecast[_TARGET_COL],
                    rows_with_forecast["forecast"],
                ),
                "Bias": forecast_error.mean(),
                "n_obs": len(rows_with_forecast),
            }
        )

    comparison = pd.DataFrame(metric_rows)
    comparison = comparison.sort_values("WAPE").reset_index(drop=True)
    selected_model = str(comparison.loc[0, "model"])
    selected_selection = candidate_forecasts[selected_model].dropna(
        subset=["forecast"]
    ).copy()
    selected_selection["model"] = (
        f"{selected_model} (model selection OOF; rolling one-step)"
    )
    return comparison, selected_model, selected_selection


def forecast_ax_ay(
    df: pd.DataFrame, classes: list[str], params: dict | None = None
) -> pd.DataFrame:
    """Forecast final evaluation dates after all choices are fixed."""
    class_is_selected = df["class"].isin(classes)
    model_data = df.loc[class_is_selected].copy()
    demand_columns = [_SKU_COL, _DATE_COL, _TARGET_COL]
    feature_df = create_features(model_data[demand_columns])

    if params:
        frozen_params = params
    else:
        frozen_params = tune_lgbm(feature_df)

    model, sku_categories = train_lgbm(feature_df, frozen_params)
    feature_columns = get_feature_cols(feature_df)

    final_rows = feature_df[_DATE_COL] >= _FINAL_TEST_START
    final_data = feature_df.loc[final_rows].copy()
    final_features = _set_sku_categories(
        final_data[feature_columns],
        sku_categories,
    )

    output_columns = [_SKU_COL, _DATE_COL, _TARGET_COL]
    result = final_data[output_columns].copy()
    result["forecast"] = _nonnegative_forecast(
        model.predict(final_features)
    )
    result["model"] = (
        "LightGBM (locked final evaluation; rolling one-step)"
    )

    save_params(
        frozen_params,
        feature_cols=feature_columns,
        classes=classes,
    )
    save_feature_importance(model, feature_columns)
    return result.reset_index(drop=True)


def _rolling_baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create three simple forecasts using only earlier demand."""
    demand_columns = [_SKU_COL, _DATE_COL, _TARGET_COL]
    baseline_data = df[demand_columns].copy()
    baseline_data = baseline_data.sort_values(
        [_SKU_COL, _DATE_COL]
    ).reset_index(drop=True)
    demand_by_sku = baseline_data.groupby(
        _SKU_COL,
        observed=True,
    )[_TARGET_COL]

    baseline_data["Naive"] = demand_by_sku.shift(1)
    baseline_data["SeasonalNaive"] = demand_by_sku.shift(_SEASON_LENGTH)
    baseline_data["HistoricMean"] = demand_by_sku.transform(
        lambda sku_demand: sku_demand.shift(1)
        .expanding(min_periods=1)
        .mean()
    )
    return baseline_data


def select_baseline_model(df: pd.DataFrame, classes: list[str]) -> str:
    """Choose the simple method with the lowest selection-period WAPE."""
    class_is_selected = df["class"].isin(classes)
    candidates = _rolling_baseline_frame(df.loc[class_is_selected])

    selection_dates = _selection_validation_dates(candidates)
    date_is_selected = candidates[_DATE_COL].isin(selection_dates)
    selection_data = candidates.loc[date_is_selected]

    scores = {}
    model_names = ["Naive", "SeasonalNaive", "HistoricMean"]
    for model_name in model_names:
        forecast_exists = selection_data[model_name].notna()
        actual_demand = selection_data.loc[forecast_exists, _TARGET_COL]
        forecast_demand = selection_data.loc[forecast_exists, model_name]
        scores[model_name] = _wape(actual_demand, forecast_demand)

    selected_model = min(scores, key=scores.get)
    logger.info(
        "Selected %s for %s from model-selection WAPE: %s",
        selected_model,
        classes,
        scores,
    )
    return selected_model


def _baseline_forecast_for_dates(
    frame: pd.DataFrame,
    selected: str,
    dates: pd.DatetimeIndex,
    *,
    phase_label: str,
) -> pd.DataFrame:
    """Apply one selected baseline method to a set of dates."""
    if selected not in {"Naive", "SeasonalNaive", "HistoricMean"}:
        raise ValueError(f"Unknown baseline model: {selected}.")

    date_is_selected = frame[_DATE_COL].isin(dates)
    result = frame.loc[date_is_selected].copy()
    result["forecast"] = result[selected]
    result["model"] = f"{selected} ({phase_label}; rolling one-step)"

    if result.empty:
        raise ValueError(f"{selected} produced no {phase_label} rows.")

    if result["forecast"].isna().any():
        missing = int(result["forecast"].isna().sum())
        raise ValueError(
            f"{selected} produced {missing} missing "
            f"{phase_label} forecasts."
        )

    output_columns = [
        _SKU_COL,
        _DATE_COL,
        _TARGET_COL,
        "forecast",
        "model",
    ]
    return result[output_columns].reset_index(drop=True)


def calibration_forecast_baseline_class(
    df: pd.DataFrame, classes: list[str], selected: str
) -> pd.DataFrame:
    """Apply an already-selected method to policy-calibration dates."""
    class_is_selected = df["class"].isin(classes)
    baseline_data = _rolling_baseline_frame(df.loc[class_is_selected])
    calibration_dates = _policy_calibration_dates(baseline_data)

    return _baseline_forecast_for_dates(
        baseline_data,
        selected,
        calibration_dates,
        phase_label="policy calibration; method frozen after selection",
    )


def validation_forecast_baseline_class(
    df: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Choose a baseline, then forecast selection and calibration periods."""
    selected_model = select_baseline_model(df, classes)
    class_is_selected = df["class"].isin(classes)
    baseline_data = _rolling_baseline_frame(df.loc[class_is_selected])

    selection_dates = _selection_validation_dates(baseline_data)
    selection_forecast = _baseline_forecast_for_dates(
        baseline_data,
        selected_model,
        selection_dates,
        phase_label="model selection",
    )

    calibration_dates = _policy_calibration_dates(baseline_data)
    calibration_forecast = _baseline_forecast_for_dates(
        baseline_data,
        selected_model,
        calibration_dates,
        phase_label="policy calibration; method frozen after selection",
    )

    return selection_forecast, calibration_forecast, selected_model


def forecast_baseline_class(
    df: pd.DataFrame, classes: list[str], selected: str | None = None
) -> pd.DataFrame:
    """Apply the selected baseline to final evaluation dates."""
    if selected:
        selected_model = selected
    else:
        selected_model = select_baseline_model(df, classes)

    class_is_selected = df["class"].isin(classes)
    baseline_data = _rolling_baseline_frame(df.loc[class_is_selected])
    final_date_mask = baseline_data[_DATE_COL] >= _FINAL_TEST_START
    final_dates = pd.DatetimeIndex(
        sorted(baseline_data.loc[final_date_mask, _DATE_COL].unique())
    )

    if final_dates.empty:
        raise ValueError(
            f"No locked final-evaluation dates on or after "
            f"{_FINAL_TEST_START.date()}."
        )
    return _baseline_forecast_for_dates(
        baseline_data,
        selected_model,
        final_dates,
        phase_label="locked final evaluation",
    )


def combine_and_save(*forecast_frames: pd.DataFrame) -> pd.DataFrame:
    """Combine final forecasts, validate them and save one canonical file."""
    forecast_df = pd.concat(forecast_frames, ignore_index=True)
    has_duplicate_sku_dates = forecast_df.duplicated(
        [_SKU_COL, _DATE_COL]
    ).any()
    if has_duplicate_sku_dates:
        raise ValueError("Forecast output contains duplicate SKU-date rows.")

    has_negative_forecast = (forecast_df["forecast"] < 0).any()
    if has_negative_forecast:
        raise ValueError("Forecast output must be nonnegative unit demand.")

    os.makedirs(_PRO_DIR, exist_ok=True)
    output_path = os.path.join(_PRO_DIR, "forecast.csv")
    forecast_df.to_csv(output_path, index=False)
    return forecast_df


def evaluate_models(
    y_true: pd.Series,
    predictions: dict[str, pd.Series | np.ndarray],
) -> pd.DataFrame:
    """Calculate common forecast metrics for each candidate model."""
    metric_rows = []
    actual_series = pd.Series(y_true).reset_index(drop=True)

    for model_name, prediction in predictions.items():
        forecast_series = pd.Series(prediction).reset_index(drop=True)
        comparison = pd.DataFrame(
            {
                "actual": actual_series,
                "forecast": forecast_series,
            }
        ).dropna()

        if comparison.empty:
            continue

        actual_demand = comparison["actual"]
        forecast_demand = comparison["forecast"]
        mae = mean_absolute_error(actual_demand, forecast_demand)
        rmse = root_mean_squared_error(actual_demand, forecast_demand)
        bias = float((forecast_demand - actual_demand).mean())
        average_demand = actual_demand.mean()

        if average_demand == 0:
            normalized_mae = np.nan
            normalized_bias = np.nan
        else:
            normalized_mae = mae / average_demand
            normalized_bias = bias / average_demand

        metric_rows.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": _wape(actual_demand, forecast_demand),
                "NMAE_by_mean": normalized_mae,
                "Bias": bias,
                "Bias_pct_of_mean": normalized_bias,
                "n_obs": len(comparison),
            }
        )

    return pd.DataFrame(metric_rows).set_index("model")


def evaluate_forecast_hierarchy(
    forecast_df: pd.DataFrame, history_df: pd.DataFrame, sku_class_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate metrics for each SKU, each class and the full portfolio."""
    evaluation = forecast_df.merge(
        sku_class_df[[_SKU_COL, "class"]],
        on=_SKU_COL,
        how="left",
        validate="many_to_one",
    )

    # MASE and RMSSE compare model errors with a one-day naive forecast.
    history_rows = history_df[_DATE_COL] < _FINAL_TEST_START
    training_history = history_df.loc[history_rows].copy()
    sorted_history = training_history.sort_values([_SKU_COL, _DATE_COL])
    naive_error = sorted_history.groupby(
        _SKU_COL,
        observed=True,
    )[_TARGET_COL].diff()

    scale_data = training_history.assign(
        abs_naive=naive_error.abs(),
        sq_naive=naive_error**2,
    )
    scale_by_sku = (
        scale_data
        .groupby(_SKU_COL, observed=True)
        .agg(mae_scale=("abs_naive", "mean"), mse_scale=("sq_naive", "mean"))
        .reset_index()
    )
    evaluation = evaluation.merge(
        scale_by_sku,
        on=_SKU_COL,
        validate="many_to_one",
    )

    evaluation["error"] = (
        evaluation["forecast"] - evaluation[_TARGET_COL]
    )
    evaluation["abs_error"] = evaluation["error"].abs()
    evaluation["sq_error"] = evaluation["error"] ** 2

    sku_rows = []
    for sku_id, sku_data in evaluation.groupby(_SKU_COL, observed=True):
        sku_mae = sku_data["abs_error"].mean()
        sku_mse = sku_data["sq_error"].mean()
        mae_scale = sku_data["mae_scale"].iloc[0]
        mse_scale = sku_data["mse_scale"].iloc[0]

        sku_rows.append(
            {
                "level": "sku",
                "segment": sku_id,
                "class": sku_data["class"].iloc[0],
                "MAE": sku_mae,
                "RMSE": np.sqrt(sku_mse),
                "WAPE": _wape(
                    sku_data[_TARGET_COL],
                    sku_data["forecast"],
                ),
                "Bias": sku_data["error"].mean(),
                "MASE": sku_mae / mae_scale,
                "RMSSE": np.sqrt(sku_mse / mse_scale),
                "n_obs": len(sku_data),
            }
        )

    sku_metrics = pd.DataFrame(sku_rows)

    # Portfolio and class rows use volume-weighted WAPE. MASE and RMSSE are
    # simple averages of the member-SKU scores.
    aggregate_rows = []
    reporting_groups = [("portfolio", "portfolio", evaluation)]
    for class_name, class_data in evaluation.groupby(
        "class",
        observed=True,
    ):
        reporting_groups.append(("class", class_name, class_data))

    for level, segment, group_data in reporting_groups:
        member_skus = group_data[_SKU_COL].unique()
        member_metrics = sku_metrics[sku_metrics["segment"].isin(member_skus)]

        if level == "class":
            class_label = segment
        else:
            class_label = "ALL"

        aggregate_rows.append(
            {
                "level": level,
                "segment": segment,
                "class": class_label,
                "MAE": group_data["abs_error"].mean(),
                "RMSE": np.sqrt(group_data["sq_error"].mean()),
                "WAPE": _wape(
                    group_data[_TARGET_COL],
                    group_data["forecast"],
                ),
                "Bias": group_data["error"].mean(),
                "MASE": member_metrics["MASE"].mean(),
                "RMSSE": member_metrics["RMSSE"].mean(),
                "n_obs": len(group_data),
            }
        )

    aggregate_metrics = pd.DataFrame(aggregate_rows)
    return pd.concat(
        [aggregate_metrics, sku_metrics],
        ignore_index=True,
    )


def save_forecast_metrics(metrics: pd.DataFrame) -> None:
    """Save the class, SKU and portfolio forecast metrics."""
    os.makedirs(_PRO_DIR, exist_ok=True)
    output_path = os.path.join(_PRO_DIR, "forecast_metrics.csv")
    metrics.to_csv(output_path, index=False)


def save_params(
    params: dict, feature_cols: list[str] | None = None, classes: list[str] | None = None
) -> None:
    """Save model settings and the three forecasting windows."""
    os.makedirs(_PARAMS_DIR, exist_ok=True)
    payload = {
        "protocol": _config["backtest"]["protocol"],
        "forecast_horizon_days": _config["backtest"]["forecast_horizon_days"],
        "selection_metric": _config["backtest"]["selection_metric"],
        "model_selection_start": str(_MODEL_SELECTION_START.date()),
        "model_selection_end": str(_MODEL_SELECTION_END.date()),
        "policy_calibration_start": str(_POLICY_CALIBRATION_START.date()),
        "policy_calibration_end": str(_POLICY_CALIBRATION_END.date()),
        "final_test_start": str(_FINAL_TEST_START.date()),
        "feature_schema": feature_cols or [],
        "training_classes": classes or [],
        "params": params,
    }
    output_path = os.path.join(_PARAMS_DIR, "best_params.json")
    with open(output_path, "w", encoding="utf-8") as params_file:
        json.dump(payload, params_file, indent=2)


def save_feature_importance(model: lgb.LGBMRegressor, feature_names: list[str]) -> None:
    """Save LightGBM feature importance from highest to lowest."""
    os.makedirs(_PARAMS_DIR, exist_ok=True)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    output_path = os.path.join(_PARAMS_DIR, "feature_importance.csv")
    importance.to_csv(output_path, index=False)