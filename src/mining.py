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


def mcmc_sampler_rules(df_onehot: pd.DataFrame, n_iterations: int = 5000, min_support: float = 0.01) -> pd.DataFrame:
    """Génère des règles par échantillonnage MCMC sans extraction exhaustive.

    Args:
        df_onehot: DataFrame binarisé des transactions.
        n_iterations: Nombre d'itérations pour la chaîne de Markov.
        min_support: Support minimum pour qu'une règle soit considérée.

    Returns:
        Un DataFrame de règles échantillonnées.
    """
    import random
    all_items = list(df_onehot.columns)
    collected_rules = {}

    if not all_items:
        return pd.DataFrame()

    # Démarrage avec un itemset aléatoire de taille 2
    current_itemset = frozenset(random.sample(all_items, min(2, len(all_items))))

    for _ in range(n_iterations):
        # Proposition d'un nouvel itemset (ajout/retrait d'un item)
        new_itemset = set(current_itemset)
        if random.random() < 0.5 and len(new_itemset) > 1: # Retrait
            item_to_remove = random.choice(list(new_itemset))
            new_itemset.remove(item_to_remove)
        elif len(new_itemset) < len(all_items): # Ajout
            item_to_add = random.choice([item for item in all_items if item not in new_itemset])
            new_itemset.add(item_to_add)
        
        new_itemset = frozenset(new_itemset)

        # Calcul du support de l'itemset
        support = df_onehot[list(new_itemset)].all(axis=1).mean()

        # Critère d'acceptation (simplifié) : on accepte si le support est suffisant
        if support >= min_support:
            current_itemset = new_itemset
            
            # Si l'itemset a au moins 2 items, on génère une règle
            if len(current_itemset) >= 2:
                # On choisit un conséquent aléatoire
                consequent = frozenset(random.sample(list(current_itemset), 1))
                antecedent = current_itemset - consequent
                
                # Calcul des métriques pour la règle (A -> C)
                antecedent_support = df_onehot[list(antecedent)].all(axis=1).mean()
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    consequent_support = df_onehot[list(consequent)].all(axis=1).mean()
                    if consequent_support > 0:
                        lift = confidence / consequent_support
                        
                        # On stocke la règle si elle est intéressante (lift > 1)
                        if lift > 1:
                            rule_id = (antecedent, consequent)
                            if rule_id not in collected_rules:
                                collected_rules[rule_id] = {
                                    'antecedents': antecedent,
                                    'consequents': consequent,
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'length': len(current_itemset)
                                }

    if not collected_rules:
        return pd.DataFrame()

    rules_df = pd.DataFrame.from_records(list(collected_rules.values()))
    return rules_df