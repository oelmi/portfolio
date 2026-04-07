"""
featureEngineering.py — create_features()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def _to_bool_int(y: pd.Series) -> pd.Series:
    """Convert labels to 0/1."""
    if y.dtype == bool:
        return y.astype(int)
    return (
        y.astype(str)
         .str.strip()
         .str.lower()
         .map({"true": 1, "false": 0, "1": 1, "0": 0})
         .astype("Int64")
         .fillna(0)
         .astype(int)
    )


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast selected columns to numeric."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic derived features."""
    df = df.copy()

    df = _safe_numeric(df, ["startYear", "runtimeMinutes", "numVotes", "popularity", "vote_average", "vote_count"])

    df["decade"] = (df["startYear"] // 10) * 10 if "startYear" in df.columns else np.nan

    df["has_votes"] = df["numVotes"].notna().astype(int) if "numVotes" in df.columns else 0
    df["has_runtime"] = df["runtimeMinutes"].notna().astype(int) if "runtimeMinutes" in df.columns else 0
    df["has_tmdb_popularity"] = df["popularity"].notna().astype(int) if "popularity" in df.columns else 0
    df["has_tmdb_vote_average"] = df["vote_average"].notna().astype(int) if "vote_average" in df.columns else 0
    df["has_tmdb_vote_count"] = df["vote_count"].notna().astype(int) if "vote_count" in df.columns else 0

    if "primaryTitle" in df.columns and "originalTitle" in df.columns:
        pt = df["primaryTitle"].fillna("").astype(str).str.strip().str.lower()
        ot = df["originalTitle"].fillna("").astype(str).str.strip().str.lower()
        df["is_foreign"] = ((pt != "") & (ot != "") & (pt != ot)).astype(int)
    else:
        df["is_foreign"] = 0

    if "runtimeMinutes" in df.columns:
        df["is_short_movie"] = (df["runtimeMinutes"] < 80).astype("Int64")
        df["is_long_movie"] = (df["runtimeMinutes"] > 150).astype("Int64")
        df["runtime_log"] = np.log1p(df["runtimeMinutes"].clip(lower=0))
    else:
        df["is_short_movie"] = 0
        df["is_long_movie"] = 0
        df["runtime_log"] = np.nan

    df["numVotes_log"] = np.log1p(df["numVotes"].clip(lower=0)) if "numVotes" in df.columns else np.nan
    df["vote_count_log"] = np.log1p(df["vote_count"].clip(lower=0)) if "vote_count" in df.columns else np.nan

    return df


def _fit_count_encoding(train_col: pd.Series) -> pd.Series:
    """Compute frequency encoding."""
    return train_col.fillna("__MISSING__").astype(str).value_counts()


def _apply_count_encoding(series: pd.Series, counts: pd.Series) -> pd.Series:
    """Apply frequency encoding."""
    return (
        series.fillna("__MISSING__")
              .astype(str)
              .map(counts)
              .fillna(0)
              .astype(float)
    )


def _fit_target_stats(train_df: pd.DataFrame, key_col: str, y_col: str = "label") -> pd.DataFrame:
    """Compute mean and count per key."""
    tmp = train_df[[key_col, y_col]].copy()
    tmp[key_col] = tmp[key_col].fillna("__MISSING__").astype(str)
    tmp[y_col] = _to_bool_int(tmp[y_col])

    return (
        tmp.groupby(key_col)[y_col]
           .agg(["mean", "count"])
           .rename(columns={"mean": f"{key_col}_avg_label", "count": f"{key_col}_movie_count"})
           .reset_index()
    )


def _apply_target_stats(df: pd.DataFrame, stats: pd.DataFrame, key_col: str, global_mean: float) -> pd.DataFrame:
    """Join target stats."""
    df = df.copy()
    df[key_col] = df[key_col].fillna("__MISSING__").astype(str)
    df = df.merge(stats, on=key_col, how="left")

    df[f"{key_col}_avg_label"] = df[f"{key_col}_avg_label"].fillna(global_mean)
    df[f"{key_col}_movie_count"] = df[f"{key_col}_movie_count"].fillna(0)

    return df


def _fit_language_encoding(train_col: pd.Series) -> dict[str, int]:
    """Create label encoding map."""
    values = train_col.fillna("__MISSING__").astype(str).value_counts().index.tolist()
    return {v: i for i, v in enumerate(values)}


def _apply_language_encoding(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Apply label encoding."""
    return (
        series.fillna("__MISSING__")
              .astype(str)
              .map(mapping)
              .fillna(-1)
              .astype(int)
    )


def _median_impute_from_train(X_train: pd.DataFrame, X_other: pd.DataFrame):
    """Impute using train medians."""
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_other.fillna(medians)


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature creation."""
    return _add_basic_features(df.copy())


def create_features(cleaned: dict):
    """Generate train/val/test features."""
    df_train_full = cleaned["df_train"].copy()
    df_test_hidden = cleaned["df_test"].copy()

    if "label" not in df_train_full.columns:
        raise ValueError("cleaned['df_train'] must contain a label column.")

    df_train_full = _prepare_feature_frame(df_train_full)
    df_test_hidden = _prepare_feature_frame(df_test_hidden)

    y_full = _to_bool_int(df_train_full["label"])

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_full
    )

    y_train = _to_bool_int(train_df["label"])
    y_val = _to_bool_int(val_df["label"])
    global_mean = float(y_train.mean())

    for col in ["director_nm", "writer_nm"]:
        if col in train_df.columns:
            counts = _fit_count_encoding(train_df[col])
            train_df[f"{col}_freq"] = _apply_count_encoding(train_df[col], counts)
            val_df[f"{col}_freq"] = _apply_count_encoding(val_df[col], counts)
            df_test_hidden[f"{col}_freq"] = _apply_count_encoding(df_test_hidden[col], counts)

    for col in ["director_nm", "writer_nm"]:
        if col in train_df.columns:
            stats = _fit_target_stats(
                pd.concat([train_df[[col]], y_train.rename("label")], axis=1),
                key_col=col
            )
            train_df = _apply_target_stats(train_df, stats, col, global_mean)
            val_df = _apply_target_stats(val_df, stats, col, global_mean)
            df_test_hidden = _apply_target_stats(df_test_hidden, stats, col, global_mean)

    if "original_language" in train_df.columns:
        lang_map = _fit_language_encoding(train_df["original_language"])
        train_df["original_language_enc"] = _apply_language_encoding(train_df["original_language"], lang_map)
        val_df["original_language_enc"] = _apply_language_encoding(val_df["original_language"], lang_map)
        df_test_hidden["original_language_enc"] = _apply_language_encoding(df_test_hidden["original_language"], lang_map)

    candidate_features = [
        "startYear","runtimeMinutes","numVotes","decade","has_votes","has_runtime",
        "is_foreign","is_short_movie","is_long_movie","runtime_log","numVotes_log",
        "popularity","vote_average","vote_count","vote_count_log",
        "has_tmdb_popularity","has_tmdb_vote_average","has_tmdb_vote_count",
        "director_nm_freq","writer_nm_freq",
        "director_nm_avg_label","director_nm_movie_count",
        "writer_nm_avg_label","writer_nm_movie_count",
        "original_language_enc"
    ]

    genre_cols = sorted([c for c in train_df.columns if c.startswith("genre_")])
    feature_cols = [c for c in candidate_features + genre_cols if c in train_df.columns]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = df_test_hidden[feature_cols].copy()

    for df in [X_train, X_val, X_test]:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    X_train, X_val = _median_impute_from_train(X_train, X_val)
    X_train, X_test = _median_impute_from_train(X_train, X_test)

    print(f"Feature engineering complete: {len(feature_cols)} features")

    return X_train, X_val, X_test, y_train, y_val