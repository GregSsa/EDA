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

    # --- NOUVEAU : Méthode pour le coverage global de l'échantillon ---
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

# --- Métriques d'évaluation ---
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