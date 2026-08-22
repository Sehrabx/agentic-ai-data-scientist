"""
tools.py — Deterministic, non-LLM tool layer.

This is the "hands" of the system. Nothing here reasons or guesses intent;
it just does exact computation. Agents call these functions and interpret
the results. Keeping this separate from agents.py means every numeric
result in the final report is reproducible and auditable, not hallucinated.
"""
from __future__ import annotations
import base64
import io
import json
import sqlite3
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- LOADING
def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


# ---------------------------------------------------------------- PROFILE
def profile_dataset(df: pd.DataFrame) -> dict:
    """Deterministic dataset understanding: shape, dtypes, missingness, cardinality."""
    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
    }
    for col in df.columns:
        s = df[col]
        col_info = {
            "name": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            col_info.update({
                "kind": "numeric",
                "mean": _r(desc.get("mean")), "std": _r(desc.get("std")),
                "min": _r(desc.get("min")), "max": _r(desc.get("max")),
                "median": _r(s.median()),
                "skew": _r(s.skew()),
            })
        else:
            top = s.value_counts(dropna=True).head(5)
            col_info.update({
                "kind": "categorical" if s.nunique(dropna=True) <= max(20, int(0.05 * len(s))) else "text/high-cardinality",
                "top_values": {str(k): int(v) for k, v in top.items()},
            })
        profile["columns"].append(col_info)
    profile["n_duplicate_rows"] = int(df.duplicated().sum())
    return profile


def _r(x, nd=4):
    try:
        return round(float(x), nd)
    except Exception:
        return None


# ---------------------------------------------------------------- CLEANING
def clean_dataset(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Rule-based cleaning pipeline. Returns cleaned df + human-readable log."""
    log = []
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.append(f"Dropped {before - len(df)} exact duplicate rows.")

    # drop identifier-like columns (e.g. customer_id, row_id) — these are
    # unique per row and carry no predictive signal, only noise if fed to a model
    for col in list(df.columns):
        if col == target:
            continue
        n_unique = df[col].nunique(dropna=True)
        looks_like_id = (
            n_unique >= 0.98 * len(df)
            and (col.lower() in ("id", "index") or col.lower().endswith("_id") or col.lower().endswith("id"))
        )
        if looks_like_id:
            df.drop(columns=col, inplace=True)
            log.append(f"Dropped column '{col}' — looks like a unique row identifier, not a predictive feature.")
            continue

        # drop high-cardinality free-text columns (e.g. Name, Ticket number) —
        # one-hot encoding these would create hundreds of near-useless dummy
        # features. These need real NLP feature extraction, out of scope here.
        # Exception: date-like text columns are kept — FeatureEngineeringAgent
        # decomposes those into year/month/day-of-week, which IS useful signal.
        if not pd.api.types.is_numeric_dtype(df[col]) and n_unique >= 0.5 * len(df) and n_unique > 20:
            parsed_dates = pd.to_datetime(df[col], errors="coerce")
            is_date_like = parsed_dates.notna().mean() > 0.9
            if not is_date_like:
                df.drop(columns=col, inplace=True)
                log.append(f"Dropped column '{col}' — high-cardinality free text ({n_unique} unique values), "
                           f"not usable as a categorical feature without dedicated NLP processing.")

    # drop columns that are entirely null or constant
    for col in list(df.columns):
        if df[col].isna().all():
            df.drop(columns=col, inplace=True)
            log.append(f"Dropped column '{col}' — entirely missing.")
        elif df[col].nunique(dropna=True) <= 1:
            df.drop(columns=col, inplace=True)
            log.append(f"Dropped column '{col}' — constant value, no signal.")

    for col in df.columns:
        if col == target:
            continue
        pct_missing = df[col].isna().mean()
        if pct_missing == 0:
            continue
        if pct_missing > 0.6:
            df.drop(columns=col, inplace=True)
            log.append(f"Dropped column '{col}' — {pct_missing:.0%} missing (above 60% threshold).")
        elif pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.append(f"Imputed {pct_missing:.1%} missing values in numeric column '{col}' with median ({_r(med)}).")
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            df[col] = df[col].fillna(fill)
            log.append(f"Imputed {pct_missing:.1%} missing values in categorical column '{col}' with mode ('{fill}').")

    # outlier flagging (IQR) on numeric cols, capped rather than dropped
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out > 0:
            df[col] = df[col].clip(lo, hi)
            log.append(f"Capped {n_out} extreme outliers in '{col}' to [{_r(lo,2)}, {_r(hi,2)}] (3×IQR rule).")

    if not log:
        log.append("Dataset was already clean — no changes needed.")
    return df, log


# ---------------------------------------------------------------- EDA
def eda_summary(df: pd.DataFrame, target: str | None) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    result = {"correlations": {}, "target_relationship": None}
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True).round(3)
        result["correlations"] = corr.to_dict()
    if target and target in numeric.columns:
        corr_with_target = numeric.corr(numeric_only=True)[target].drop(target, errors="ignore")
        result["target_relationship"] = corr_with_target.abs().sort_values(ascending=False).round(3).to_dict()
    return result


def make_plot_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_missingness(df: pd.DataFrame) -> str:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * len(miss) if len(miss) else 2)))
    if len(miss):
        ax.barh(miss.index[::-1], (miss.values[::-1] * 100), color="#c0392b")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values by column")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    return make_plot_b64(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> str | None:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(10, 1 + 0.6 * len(corr)), min(8, 1 + 0.6 * len(corr))))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap")
    return make_plot_b64(fig)


def plot_distributions(df: pd.DataFrame, max_cols: int = 6) -> str | None:
    numeric = df.select_dtypes(include=[np.number]).columns[:max_cols]
    if len(numeric) == 0:
        return None
    n = len(numeric)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric):
        axes[i].hist(df[col].dropna(), bins=25, color="#2980b9")
        axes[i].set_title(col, fontsize=10)
    for j in range(len(numeric), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return make_plot_b64(fig)


def plot_target_relationship(df: pd.DataFrame, target: str, task: str) -> str | None:
    if target not in df.columns:
        return None
    numeric = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")[:6]
    if len(numeric) == 0:
        return None
    n = len(numeric)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric):
        if task == "classification":
            for cls in df[target].dropna().unique()[:6]:
                axes[i].hist(df.loc[df[target] == cls, col].dropna(), bins=20, alpha=0.5, label=str(cls))
            axes[i].legend(fontsize=6)
        else:
            axes[i].scatter(df[col], df[target], s=8, alpha=0.4, color="#27ae60")
        axes[i].set_title(f"{col} vs {target}", fontsize=9)
    for j in range(len(numeric), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return make_plot_b64(fig)


# ---------------------------------------------------------------- SQL
def run_sql(df: pd.DataFrame, query: str, table_name: str = "data") -> pd.DataFrame:
    """Run arbitrary SQL against the dataframe via an in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        result = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------- FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Light, generic feature engineering: datetime decomposition, simple ratios skipped
    (kept generic/dataset-agnostic on purpose)."""
    df = df.copy()
    for col in df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df.drop(columns=col, inplace=True)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ---------------------------------------------------------------- MODELING
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "SVC": SVC(probability=True),
}
REGRESSORS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
}


def infer_task(df: pd.DataFrame, target: str) -> str:
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
        return "regression"
    return "classification"


def train_and_evaluate(df: pd.DataFrame, target: str, task: str) -> dict:
    X = df.drop(columns=[target])
    y = df[target]
    label_map = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        cat = y.astype("category")
        label_map = dict(enumerate(cat.cat.categories))  # int code -> original label
        y = cat.cat.codes

    stratify = y if (task == "classification" and y.nunique() > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pre = build_preprocessor(X)
    models = CLASSIFIERS if task == "classification" else REGRESSORS

    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            if task == "classification":
                metrics = {
                    "accuracy": _r(accuracy_score(y_test, preds)),
                    "f1": _r(f1_score(y_test, preds, average="weighted")),
                    "precision": _r(precision_score(y_test, preds, average="weighted", zero_division=0)),
                    "recall": _r(recall_score(y_test, preds, average="weighted")),
                }
                primary = metrics["f1"]
            else:
                metrics = {
                    "r2": _r(r2_score(y_test, preds)),
                    "mae": _r(mean_absolute_error(y_test, preds)),
                    "rmse": _r(mean_squared_error(y_test, preds) ** 0.5),
                }
                primary = metrics["r2"]
            results.append({"model": name, **metrics, "_primary": primary})
            fitted[name] = pipe
        except Exception as e:
            results.append({"model": name, "error": str(e), "_primary": -1e9})

    results.sort(key=lambda r: r.get("_primary", -1e9), reverse=True)
    best_name = results[0]["model"]
    best_pipe = fitted.get(best_name)

    cm_b64 = None
    if task == "classification" and best_pipe is not None:
        preds = best_pipe.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {best_name}")
        cm_b64 = make_plot_b64(fig)

    for r in results:
        r.pop("_primary", None)

    return {
        "task": task,
        "results": results,
        "best_model": best_name,
        "best_pipeline": best_pipe,          # fitted sklearn Pipeline — ready to call .predict()
        "label_map": label_map,               # int code -> original class label, or None
        "feature_columns": list(X.columns),   # exact column names/order the pipeline expects
        "confusion_matrix_plot": cm_b64,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_in": X.shape[1],
    }


def save_model(model_result: dict, path: str) -> str:
    """Persist the best fitted pipeline (+ metadata needed to use it) to a .joblib file."""
    import joblib
    bundle = {
        "pipeline": model_result["best_pipeline"],
        "label_map": model_result["label_map"],
        "feature_columns": model_result["feature_columns"],
        "task": model_result["task"],
        "model_name": model_result["best_model"],
    }
    joblib.dump(bundle, path)
    return path


def predict_new(bundle: dict, input_row: dict) -> dict:
    """Run a single new record through a saved model bundle and return a readable prediction."""
    X_new = pd.DataFrame([input_row])[bundle["feature_columns"]]
    pipe = bundle["pipeline"]
    pred = pipe.predict(X_new)[0]
    out = {"task": bundle["task"]}
    if bundle["task"] == "classification":
        label = bundle["label_map"][int(pred)] if bundle["label_map"] else pred
        out["prediction"] = label.item() if hasattr(label, "item") else label
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_new)[0]
            classes = pipe.named_steps["model"].classes_
            out["probabilities"] = {
                (bundle["label_map"][int(c)] if bundle["label_map"] else str(c)): _r(p)
                for c, p in zip(classes, proba)
            }
    else:
        out["prediction"] = _r(pred)
    return out