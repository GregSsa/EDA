"""Mining utilities: frequent itemsets and association rules.

Extracted from notebooks; thin wrappers around mlxtend functions.
"""
from typing import Optional
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


def run_fpgrowth(df_onehot: pd.DataFrame, min_support: float = 0.33) -> pd.DataFrame:
    """Run fpgrowth on a one-hot encoded DataFrame.

    Returns DataFrame with columns `itemsets` and `support`.
    """
    # Ensure boolean
    df_bool = df_onehot.astype(bool)
    if 'Transaction' in df_bool.columns:
        df_bool = df_bool.drop(columns=['Transaction'])
    frequent_itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    return frequent_itemsets


def generate_rules(frequent_itemsets: pd.DataFrame, metric: str = 'confidence', min_threshold: float = 0.5) -> pd.DataFrame:
    """Generate association rules and compute standard metrics.

    Returns DataFrame containing antecedents, consequents, support, confidence, lift, coverage, length.
    """
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    # length as antecedent + consequent size
    rules['length'] = rules['antecedents'].apply(lambda x: len(x)) + rules['consequents'].apply(lambda x: len(x))
    # coverage: antecedent support (mlxtend provides 'antecedent support')
    if 'antecedent support' in rules.columns:
        rules.rename(columns={'antecedent support': 'coverage'}, inplace=True)

    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    if 'coverage' in rules.columns:
        cols.append('coverage')
    cols.append('length')
    return rules[cols]
