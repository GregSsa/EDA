"""Utilitaires de feedback pour mettre à jour les poids d'échantillonnage.

Extrait et adapté des notebooks.
"""
from typing import Optional, Iterable
import pandas as pd

def _make_key(row: pd.Series) -> str:
    """Génère une clé string unique pour une règle.

    IMPORTANT : On trie les antécédents et les conséquents pour garantir
    la stabilité de la clé (ex: {A, B} doit donner la même clé que {B, A}).
    """
    # Conversion en liste et tri alphabétique pour la stabilité
    ant = sorted(list(row['antecedents']))
    cons = sorted(list(row['consequents']))
    return str(ant) + '||' + str(cons)


def apply_like(P_df: pd.DataFrame, idx: Optional[int] = None, key: Optional[str] = None, factor: float = 1.25) -> pd.DataFrame:
    """Applique un feedback 'J'aime' : multiplie feedback_weight par le facteur.
    Retourne le DataFrame modifié.
    """
    if 'feedback_weight' not in P_df.columns:
        P_df['feedback_weight'] = 1.0

    if idx is not None:
        # Mise à jour par index direct
        current_val = P_df.at[int(idx), 'feedback_weight']
        P_df.at[int(idx), 'feedback_weight'] = float(current_val * factor)
    elif key is not None:
        # Mise à jour par clé (plus lent mais robuste si l'index change)
        mask = (P_df.apply(_make_key, axis=1) == key)
        P_df.loc[mask, 'feedback_weight'] = P_df.loc[mask, 'feedback_weight'] * factor

    # Recalcul impératif du poids final
    P_df['final_sampling_weight'] = P_df['composite_score'] * P_df['feedback_weight']
    return P_df


def apply_dislike(P_df: pd.DataFrame, idx: Optional[int] = None, key: Optional[str] = None, factor: float = 0.8) -> pd.DataFrame:
    """Applique un feedback 'Je n'aime pas' (wrapper utilisant un facteur < 1).
    """
    return apply_like(P_df, idx=idx, key=key, factor=factor)


def reset_feedback(P_df: pd.DataFrame) -> pd.DataFrame:
    """Réinitialise tous les feedbacks utilisateurs."""
    P_df['feedback_weight'] = 1.0
    P_df['final_sampling_weight'] = P_df['composite_score'].copy()
    return P_df