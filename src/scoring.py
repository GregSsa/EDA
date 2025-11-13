"""Scoring utilities: normalization and composite score.

Extracted from notebook logic.
"""
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Normalize provided columns in-place to [0,1] using MinMaxScaler.
    Returns the DataFrame.
    """
    scaler = MinMaxScaler()
    cols = [c for c in columns if c in df.columns]
    if cols:
        df[cols] = scaler.fit_transform(df[cols])
    return df


def composite_score_row(row: pd.Series, w_support: float = 0.2, w_lift: float = 0.2, w_conf: float = 0.2,
                        w_surprise: float = 0.2, w_redundancy_penalty: float = 0.2) -> float:
    """Compute composite score for a single rule (row).

    Formula mirrors the notebook: linear combination with redundancy penalty.
    """
    support = float(row.get('support', 0.0))
    lift = float(row.get('lift', 0.0))
    conf = float(row.get('confidence', 0.0))
    length = float(row.get('length', 0.0))
    surprise = 1.0 - support
    score = (w_lift * lift) + (w_conf * conf) + (w_support * support) + (w_surprise * surprise) - (w_redundancy_penalty * (length / 10.0))
    return float(score)


def compute_composite_scores(df: pd.DataFrame, norm_columns: Iterable[str] = ('lift', 'confidence')) -> pd.DataFrame:
    """Normalize metrics and compute `composite_score` column.
    Modifies and returns the DataFrame.
    """
    df_copy = df.copy()
    df_copy = normalize_columns(df_copy, norm_columns)
    df_copy['composite_score'] = df_copy.apply(composite_score_row, axis=1)
    df_copy['feedback_weight'] = 1.0
    df_copy['final_sampling_weight'] = df_copy['composite_score'] * df_copy['feedback_weight']
    return df_copy
