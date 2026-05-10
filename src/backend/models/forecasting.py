import json
import logging
import os

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsforecast import StatsForecast
from statsforecast.models import CrostonSBA, Naive, SeasonalNaive

from features.feature_engineering import create_features, get_feature_cols

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH) as f:
    _config = json.load(f)

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_MODEL_DIR   = os.path.join(_ROOT, _config["paths"]["model"])
_PARAMS_DIR  = os.path.join(_ROOT, _config["paths"]["params"])

_DATE_COL    = _config["data"]["date_col"]
_SKU_COL     = _config["data"]["sku_col"]
_TARGET_COL  = _config["data"]["target_col"]
_TRAIN_START = _config["data"]["train_start"]
_SPLIT_DATE  = _config["data"]["split_date"]

_FORECAST_CLASSES  = _config["forecasting"]["forecast_classes"]
_MEAN_CLASSES      = _config["forecasting"]["mean_classes"]
_SEASONAL_CLASSES  = _config["forecasting"]["seasonal_classes"]
_SEASON_LENGTH     = _config["forecasting"]["season_length"]
_N_TRIALS          = _config["forecasting"]["optuna_n_trials"]
_OPTUNA_SEED       = _config["forecasting"]["optuna_seed"]
_FIXED_PARAMS      = _config["forecasting"]["lgbm_fixed_params"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_test_split(df: pd.DataFrame):
    feature_cols = get_feature_cols(df)

    train_df = df[df[_DATE_COL] < _SPLIT_DATE].copy()
    test_df  = df[df[_DATE_COL] >= _SPLIT_DATE].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[_TARGET_COL]
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[_TARGET_COL]

    for X in [X_train, X_test]:
        if _SKU_COL in X.columns:
            X[_SKU_COL] = X[_SKU_COL].astype("category")

    return X_train, X_test, y_train, y_test

def _build_objective(X_train, y_train, X_test, y_test):
    def objective(trial):
        params = {
            "lambda_l1"        : trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2"        : trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves"       : trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction" : trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq"     : trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            **_FIXED_PARAMS,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        return root_mean_squared_error(y_test, model.predict(X_test))
    return objective

def tune_lgbm(X_train, y_train, X_test, y_test) -> dict:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_OPTUNA_SEED),
    )
    study.optimize(_build_objective(X_train, y_train, X_test, y_test), n_trials=_N_TRIALS)
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    return {**study.best_params, **_FIXED_PARAMS}


def train_lgbm(X_train, y_train, params: dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

def forecast_ax_ay(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df[df["class"].isin(_FORECAST_CLASSES)].copy()
    df_feat  = create_features(df_model[[_SKU_COL, _DATE_COL, _TARGET_COL]])

    X_train, X_test, y_train, y_test = train_test_split(df_feat)

    best_params = tune_lgbm(X_train, y_train, X_test, y_test)
    model       = train_lgbm(X_train, y_train, best_params)

    test_df = df_feat[df_feat[_DATE_COL] >= _SPLIT_DATE].copy()
    preds   = model.predict(X_test)

    result = test_df[[_SKU_COL, _DATE_COL, _TARGET_COL]].copy()
    result["forecast"] = preds

    save_model(model)
    save_params(best_params)
    save_feature_importance(model, X_train.columns.tolist())

    return result.reset_index(drop=True)

def _prep_sf(df: pd.DataFrame) -> pd.DataFrame:
    return df[[_SKU_COL, _DATE_COL, _TARGET_COL]].rename(
        columns={_SKU_COL: "unique_id", _DATE_COL: "ds", _TARGET_COL: "y"}
    )

def forecast_cx(df: pd.DataFrame) -> pd.DataFrame:
    df_cx   = df[df["class"].isin(_MEAN_CLASSES)].copy()
    sf_all  = _prep_sf(df_cx)
    sf_train = sf_all[(sf_all["ds"] >= _TRAIN_START) & (sf_all["ds"] < _SPLIT_DATE)].copy()
    sf_test  = sf_all[sf_all["ds"] >= _SPLIT_DATE].copy()

    mean_value = sf_train["y"].mean()

    result = sf_test.rename(columns={"unique_id": _SKU_COL, "ds": _DATE_COL, "y": _TARGET_COL}).copy()
    result["forecast"] = mean_value
    return result[[_SKU_COL, _DATE_COL, _TARGET_COL, "forecast"]].reset_index(drop=True)

def forecast_bz_cz(df: pd.DataFrame) -> pd.DataFrame:
    df_bz_cz = df[df["class"].isin(_SEASONAL_CLASSES)].copy()
    sf_all   = _prep_sf(df_bz_cz)
    sf_train = sf_all[(sf_all["ds"] >= _TRAIN_START) & (sf_all["ds"] < _SPLIT_DATE)].copy()
    sf_test  = sf_all[sf_all["ds"] >= _SPLIT_DATE].copy()
    horizon  = sf_test["ds"].nunique()

    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=_SEASON_LENGTH), CrostonSBA()],
        freq="D",
        n_jobs=-1,
    )
    preds = sf.forecast(df=sf_train, h=horizon).reset_index(drop=True)

    result = preds[["unique_id", "ds", "SeasonalNaive"]].merge(
        sf_test[["unique_id", "ds", "y"]], on=["unique_id", "ds"]
    ).rename(columns={
        "unique_id"    : _SKU_COL,
        "ds"           : _DATE_COL,
        "y"            : _TARGET_COL,
        "SeasonalNaive": "forecast",
    })
    return result[[_SKU_COL, _DATE_COL, _TARGET_COL, "forecast"]].reset_index(drop=True)

def combine_and_save(ax_ay: pd.DataFrame, cx: pd.DataFrame, bz_cz: pd.DataFrame) -> pd.DataFrame:
    df_forecast = pd.concat([ax_ay, cx, bz_cz], ignore_index=True)
    os.makedirs(_PRO_DIR, exist_ok=True)
    path = os.path.join(_PRO_DIR, "forecast.csv")
    df_forecast.to_csv(path, index=False)
    logger.info(f"Forecast saved → {path}")
    return df_forecast

def save_model(model: lgb.LGBMRegressor) -> None:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    path = os.path.join(_MODEL_DIR, "lgbm_model.pkl")
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")

def load_model() -> lgb.LGBMRegressor:
    return joblib.load(os.path.join(_MODEL_DIR, "lgbm_model.pkl"))

def save_params(params: dict) -> None:
    import json
    os.makedirs(_PARAMS_DIR, exist_ok=True)
    path = os.path.join(_PARAMS_DIR, "best_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Params saved → {path}")

def evaluate_models(y_true: pd.Series, predictions: dict) -> pd.DataFrame:
    mean_actual = y_true.mean()
    results = []
 
    for model_name, y_pred in predictions.items():
        y_pred = np.array(y_pred)
        mae    = mean_absolute_error(y_true, y_pred)
        rmse   = root_mean_squared_error(y_true, y_pred)
        bias   = np.mean(y_pred - y_true)
 
        results.append({
            "Model" : model_name,
            "MAE"   : round(mae,  4),
            "%MAE"  : round(mae  / mean_actual * 100, 2),
            "RMSE"  : round(rmse, 4),
            "%RMSE" : round(rmse / mean_actual * 100, 2),
            "Bias"  : round(bias, 4),
            "%Bias" : round(bias / mean_actual * 100, 2),
        })
 
    return pd.DataFrame(results).set_index("Model")

def save_feature_importance(model: lgb.LGBMRegressor, feature_names: list) -> None:
    import json
    df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    os.makedirs(_PARAMS_DIR, exist_ok=True)
    path = os.path.join(_PARAMS_DIR, "top_feature_importances.json")
    with open(path, "w") as f:
        json.dump(df["feature"].tolist(), f, indent=2)
    logger.info(f"Feature importance saved → {path}")