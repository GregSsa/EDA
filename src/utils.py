import pandas as pd
import numpy as np
import re
import time
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MinMaxScaler

class TransactionDf():
    def __init__(self, file_path=None, dataframe=None, header=False, target_column=None, separator=",", formatting="Auto"):
        self.dfs = []
        self.last_mining_time = 0 # Pour métrique temps

        raw_df = None
        if dataframe is not None:
            raw_df = dataframe
        elif file_path:
            if header:
                raw_df = pd.read_csv(file_path)
            else:
                raw_df = pd.read_csv(file_path, header=None)

        if raw_df is not None:
            if formatting == "Auto":
                formatting = self._detect_format(raw_df)
                # print(f"Format détecté automatiquement : {formatting}")

            self._process_transactions(raw_df, target_column, separator, formatting)

    def _detect_format(self, df):
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            sample_text = str(df[str_cols[0]].iloc[0]) + (str(df[str_cols[-1]].iloc[0]) if len(str_cols)>1 else "")
            if "(" in sample_text and ")" in sample_text:
                return "Sequential"

        numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
        if len(numeric_cols) >= 2:
            sample = df[numeric_cols].sample(min(len(df), 50)).fillna(0)
            unique_vals = np.unique(sample.values)
            is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
            if is_binary and len(numeric_cols) >= (len(df.columns) - 1):
                return "Wide_OneHot"

        if df.shape[1] > df.shape[0] and df.shape[1] > 2:
            try:
                if "," in str(df.iloc[0, 1]):
                    return "Wide_Transposed"
            except: pass

        last_col = df.columns[-1]
        if df[last_col].dtype == object:
            sample_items = df[last_col].head(5).astype(str)
            has_separator = sample_items.str.contains(",").any()
            if has_separator:
                return "Basic"
            first_col = df.columns[0]
            if df[first_col].duplicated().any():
                return "Long"
        return "Basic"

    def _process_transactions(self, transactions, target_column, sep, formatting):
        if formatting == "Long":
            col_id = transactions.columns[0]
            col_item = target_column if target_column in transactions.columns else transactions.columns[1]
            df_cross = pd.crosstab(transactions[col_id], transactions[col_item])
            df_cross = df_cross.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(df_cross)
            return

        elif formatting == "Sequential":
            if not target_column or target_column not in transactions.columns:
                for col in transactions.columns:
                    if transactions[col].dtype == object and "(" in str(transactions[col].iloc[0]):
                        target_column = col
                        break
            transactions[target_column] = transactions[target_column].astype(str).str.replace(r'[()]', '', regex=True)
            return self._process_transactions(transactions, target_column, ",", "Basic")

        elif formatting == "Wide_Transposed":
            transactions = transactions.T
            transactions.columns = ["items"]
            return self._process_transactions(transactions, "items", sep, "Basic")

        elif formatting == "Wide_OneHot" or formatting == "Wide":
            numeric_cols = transactions.select_dtypes(include=[np.number, bool]).columns.tolist()
            str_cols = transactions.select_dtypes(include=['object']).columns.tolist()
            if len(str_cols) > 0:
                transactions = transactions.set_index(str_cols[0])
            transactions = transactions[numeric_cols]
            transactions = transactions.fillna(0).applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions)
            return

        else:
            if not target_column or target_column not in transactions.columns:
                target_column = transactions.columns[-1]
            transactions = transactions[transactions[target_column].apply(lambda x: isinstance(x, str))]
            transactions[target_column] = transactions[target_column].str.replace('"', '').str.replace("'", "")
            transactions[target_column] = transactions[target_column].apply(lambda x: [i.strip() for i in x.split(sep)])
            transactions_exploded = transactions[target_column].explode()
            transactions_dummies = pd.get_dummies(transactions_exploded)
            transactions_dummies = transactions_dummies.groupby(transactions_dummies.index).sum()
            transactions_dummies = transactions_dummies.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions_dummies)

    def get_df(self):
        return self.dfs[0] if self.dfs else None

    # NOUVEAU : Méthode pour le coverage global de l'échantillon
    def calculate_global_coverage(self, sample_rules):
        """Calcule le % de transactions couvertes par AU MOINS une règle de l'échantillon."""
        if self.dfs and not sample_rules.empty:
            df = self.dfs[0]
            covered_indices = set()

            # Pour chaque règle de l'échantillon
            for _, rule in sample_rules.iterrows():
                # On récupère les items de l'antécédent
                antecedents = list(rule['antecedents'])
                # On cherche les lignes où tous les items de l'antécédent sont présents (1)
                # Note: c'est une opération lourde, on l'optimise un peu
                if all(item in df.columns for item in antecedents):
                    # Masque booléen pour cette règle
                    mask = (df[antecedents] == 1).all(axis=1)
                    covered_indices.update(df[mask].index.tolist())

            return len(covered_indices) / len(df)
        return 0.0

# Métriques d'évaluation
def calculate_diversity(sample_df):
    """Calcul simple de diversité : (Nombre d'items uniques utilisés) / (Total items possibles dans l'échantillon)"""
    if sample_df.empty: return 0.0

    unique_items = set()
    total_slots = 0

    for _, row in sample_df.iterrows():
        items = set(row['antecedents']).union(set(row['consequents']))
        unique_items.update(items)
        total_slots += len(items)

    # Jaccard moyen entre paires serait mieux mais lourd, ici une heuristique simple
    # Diversité = ratio d'items uniques par rapport à la "place" utilisée
    if total_slots == 0: return 0
    return len(unique_items) / total_slots

def calculate_composite_score(df, w_support, w_lift, w_conf, w_surprise, w_redundancy):
    scaler = MinMaxScaler()
    metrics = ['lift', 'confidence', 'support']
    df_norm = df.copy()
    try:
        df_norm[metrics] = scaler.fit_transform(df[metrics])
    except:
        df_norm[metrics] = 0.5

    scores = (w_lift * df_norm['lift'] +
              w_conf * df_norm['confidence'] +
              w_support * df_norm['support'] +
              w_surprise * (1 - df_norm['support']) -
              w_redundancy * (df['length'] / 10))
    return scores

def light_mcmc(P_df, k=10, replace=True, max_iters=10000, random_seed=None):
    weights = P_df['final_sampling_weight'].values.copy()
    weights = np.maximum(weights, 0.0)
    if weights.sum() == 0: weights = np.ones_like(weights)

    rng = np.random.default_rng(random_seed)
    n = len(weights)
    if n == 0: return P_df.copy()

    cur = int(rng.integers(n))
    samples = []
    visited = set()
    iters = 0

    while (len(samples) < k) and (iters < max_iters):
        prop = int(rng.integers(n))
        w_cur = weights[cur] if weights[cur] > 0 else 1e-12
        w_prop = weights[prop] if weights[prop] > 0 else 1e-12
        alpha = min(1.0, (w_prop / w_cur))

        if rng.random() < alpha:
            cur = prop

        if replace:
            samples.append(cur)
        else:
            if cur not in visited:
                samples.append(cur)
                visited.add(cur)
        iters += 1

    if not replace and len(samples) < k:
        remaining = list(set(range(n)) - visited)
        if remaining:
            needed = k - len(samples)
            fill = rng.choice(remaining, size=min(len(remaining), needed), replace=False)
            samples.extend(fill)

    return P_df.iloc[samples].reset_index(drop=True)


# Fonctions utilitaires pour MCMC
def support(itemset: frozenset, vertical: dict, N: int) -> float:
    """Calcul rapide du support via l'index vertical. Plus rapide que fpgrowth pour petits itemsets."""
    if not itemset:
        return 0.0
    # intersection des ensembles d'indices
    it = iter(itemset)
    inter = set(vertical[next(it)])
    for item in it:
        inter &= vertical[item]
        if not inter:
            return 0.0
    return len(inter) / float(N)

def interest_weight(itemset: frozenset, interest: str, rng, vertical: dict, N: int) -> float:
    """Poids d'intérêt utilisé par Metropolis-Hastings."""
    if len(itemset) < 2:
        return 0.0
    # Choix d'un conséquent aléatoire
    cons = frozenset([rng.choice(list(itemset))])
    ant = itemset - cons
    sup_cur = support(itemset, vertical, N)
    sup_ant = support(ant, vertical, N)
    sup_cons = support(cons, vertical, N)
    if sup_ant <= 0 or sup_cons <= 0:
        return 0.0
    conf = sup_cur / sup_ant
    lift = conf / sup_cons if sup_cons > 0 else 0.0
    if interest == "composite":
        return max(0.0, lift * conf)
    return max(0.0, lift)


# Pattern Sampling (Output Sampling) MCMC
def pattern_sample_mcmc(df_onehot: pd.DataFrame,
                        iterations: int = 5000,
                        min_support: float = 0.005,
                        max_rules: int = 500,
                        random_seed: int = None,
                        burn_in: int = 0,
                        thinning: int = 1,
                        interest: str = "lift",
                        return_stats: bool = False):

    rng = np.random.default_rng(random_seed)

    # Préparation: columns as items; ensure int/bool
    df_bin = df_onehot.fillna(0).astype(int)
    items = list(df_bin.columns)
    n_items = len(items)
    if n_items == 0:
        empty = pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift','coverage','length'])
        return (empty, {'accept_rate': 0.0}) if return_stats else empty

    # Index vertical pour support rapide
    vertical = {col: set(df_bin.index[df_bin[col] == 1]) for col in items}
    N = len(df_bin)

    # État initial: itemset aléatoire de taille 1 ou 2
    cur_size = 2 if n_items >= 2 else 1
    current = frozenset(rng.choice(items, size=cur_size, replace=False))

    rules_store: dict[tuple[frozenset,frozenset], dict] = {}
    accepts = 0
    proposals = 0

    for t in range(iterations):
        if len(rules_store) >= max_rules:
            break

        proposal = set(current)
        # Choix move: add ou remove
        if rng.random() < 0.5 and len(proposal) > 1:
            # remove
            to_remove = rng.choice(list(proposal))
            proposal.remove(to_remove)
        else:
            # add (si possible)
            remaining = [it for it in items if it not in proposal]
            if remaining:
                to_add = rng.choice(remaining)
                proposal.add(to_add)

        proposal = frozenset(proposal)
        proposals += 1
        sup_prop = support(proposal, vertical, N)
        sup_cur = support(current, vertical, N)

        # Contrainte dure: min_support
        if sup_prop < min_support:
            # rejet direct
            pass
        else:
            # Metropolis–Hastings selon poids d'intérêt
            w_prop = interest_weight(proposal, interest, rng, vertical, N)
            w_cur = interest_weight(current, interest, rng, vertical, N)
            alpha = 1.0
            if w_cur > 0:
                alpha = min(1.0, (w_prop / w_cur))
            elif w_prop > 0:
                alpha = 1.0
            else:
                alpha = 0.0
            if rng.random() < alpha:
                current = proposal
                accepts += 1

            if t >= burn_in and (t - burn_in) % max(1, thinning) == 0 and len(current) >= 2:
                # Génération d'une règle: item consequent aléatoire
                consequent = frozenset([rng.choice(list(current))])
                antecedent = current - consequent
                sup_ant = support(antecedent, vertical, N)
                sup_cons = support(consequent, vertical, N)
                if sup_ant > 0 and sup_cons > 0:
                    confidence = sup_cur / sup_ant
                    lift = confidence / sup_cons if sup_cons > 0 else 0
                    coverage = sup_ant
                    if lift > 1:  # règle intéressante
                        key = (antecedent, consequent)
                        if key not in rules_store:
                            rules_store[key] = {
                                'antecedents': antecedent,
                                'consequents': consequent,
                                'support': sup_cur,
                                'confidence': confidence,
                                'lift': lift,
                                'coverage': coverage,
                                'length': len(current)
                            }

    accept_rate = (accepts / proposals) if proposals > 0 else 0.0
    stats = {'accept_rate': accept_rate, 'iterations': iterations, 'burn_in': burn_in, 'thinning': thinning}

    if not rules_store:
        empty = pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift','coverage','length'])
        return (empty, stats) if return_stats else empty

    df_rules = pd.DataFrame.from_records(list(rules_store.values()))
    return (df_rules, stats) if return_stats else df_rules