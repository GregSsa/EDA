import pandas as pd
import numpy as np
import re
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MinMaxScaler

class TransactionDf():
    def __init__(self, file_path=None, dataframe=None, header=False, target_column=None, separator=",", formatting="Auto"):
        self.dfs = []
        
        raw_df = None
        if dataframe is not None:
            raw_df = dataframe
        elif file_path:
            # Lecture avec header par défaut car la plupart des formats en ont un
            # Si header=False est forcé, pandas générera des int pour les colonnes
            if header:
                raw_df = pd.read_csv(file_path)
            else:
                raw_df = pd.read_csv(file_path, header=None)
        
        if raw_df is not None:
            if formatting == "Auto":
                formatting = self._detect_format(raw_df)
                print(f"Format détecté automatiquement : {formatting}")
            
            self._process_transactions(raw_df, target_column, separator, formatting)

    def _detect_format(self, df):
        """
        Détection intelligente des formats supportés.
        """
        # 1. Sequential : Présence de parenthèses ( )
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            sample_text = str(df[str_cols[0]].iloc[0]) + (str(df[str_cols[-1]].iloc[0]) if len(str_cols)>1 else "")
            if "(" in sample_text and ")" in sample_text:
                return "Sequential"

        # 2. Wide Checks
        # A. One-Hot (Votre format cible : Transaction,pain,lait...)
        # Critère : Majorité de colonnes numériques contenant des valeurs binaires (0/1)
        numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
        
        if len(numeric_cols) >= 2: # Au moins 2 colonnes d'items
            # On vérifie un échantillon pour voir si c'est binaire (0, 1)
            sample = df[numeric_cols].sample(min(len(df), 50)).fillna(0)
            unique_vals = np.unique(sample.values)
            # On accepte 0, 1, 0.0, 1.0
            is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
            
            # Si c'est binaire et que ça couvre presque tout le fichier (sauf l'ID)
            if is_binary and len(numeric_cols) >= (len(df.columns) - 1):
                return "Wide_OneHot"

        # B. Transposed (Items en header mais structure transposée)
        if df.shape[1] > df.shape[0] and df.shape[1] > 2:
            try:
                # Vérifie présence de séparateur dans les cellules
                if "," in str(df.iloc[0, 1]):
                    return "Wide_Transposed"
            except: pass

        # 3. Distinction Long vs Basic
        # Basic : Cellules avec séparateurs ("pain, lait")
        # Long : Cellules uniques, souvent avec ID répété
        last_col = df.columns[-1]
        if df[last_col].dtype == object:
            sample_items = df[last_col].head(5).astype(str)
            has_separator = sample_items.str.contains(",").any()
            
            if has_separator:
                return "Basic"
            
            # Check duplication ID colonne 0
            first_col = df.columns[0]
            if df[first_col].duplicated().any():
                return "Long"

        return "Basic" # Fallback par défaut

    def _process_transactions(self, transactions, target_column, sep, formatting):
        
        # --- CAS 1 : Long Format (Transaction, Item) ---
        if formatting == "Long":
            col_id = transactions.columns[0]
            col_item = target_column if target_column in transactions.columns else transactions.columns[1]
            # Pivot rapide
            df_cross = pd.crosstab(transactions[col_id], transactions[col_item])
            df_cross = df_cross.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(df_cross)
            return

        # --- CAS 2 : Sequential (Aplatissement) ---
        elif formatting == "Sequential":
            if not target_column or target_column not in transactions.columns:
                for col in transactions.columns:
                    if transactions[col].dtype == object and "(" in str(transactions[col].iloc[0]):
                        target_column = col
                        break
            
            # On retire toutes les parenthèses pour traiter comme une liste simple
            transactions[target_column] = transactions[target_column].astype(str).str.replace(r'[()]', '', regex=True)
            return self._process_transactions(transactions, target_column, ",", "Basic")

        # --- CAS 3 : Wide Transposed ---
        elif formatting == "Wide_Transposed":
            transactions = transactions.T
            transactions.columns = ["items"] 
            return self._process_transactions(transactions, "items", sep, "Basic")

        # --- CAS 4 : Wide One-Hot (Le format demandé) ---
        elif formatting == "Wide_OneHot" or formatting == "Wide":
            # On identifie les colonnes numériques (les items)
            numeric_cols = transactions.select_dtypes(include=[np.number, bool]).columns.tolist()
            
            # Gestion de la colonne ID (si elle existe en texte/objet)
            # On essaie de préserver "Transaction" ou "T1", "T2" comme index
            str_cols = transactions.select_dtypes(include=['object']).columns.tolist()
            if len(str_cols) > 0:
                # On prend la première colonne texte comme index (ex: 'Transaction')
                transactions = transactions.set_index(str_cols[0])
            
            # On garde uniquement la partie matrice binaire
            transactions = transactions[numeric_cols]
            
            # Nettoyage et Binarisation
            transactions = transactions.fillna(0).applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions)
            return

        # --- CAS 5 : Basic (Liste d'items) ---
        else:
            if not target_column or target_column not in transactions.columns:
                target_column = transactions.columns[-1]

            transactions = transactions[transactions[target_column].apply(lambda x: isinstance(x, str))]
            
            # Nettoyage quotes et split
            transactions[target_column] = transactions[target_column].str.replace('"', '').str.replace("'", "")
            transactions[target_column] = transactions[target_column].apply(lambda x: [i.strip() for i in x.split(sep)])
            
            transactions_exploded = transactions[target_column].explode()
            transactions_dummies = pd.get_dummies(transactions_exploded)
            transactions_dummies = transactions_dummies.groupby(transactions_dummies.index).sum()
            
            transactions_dummies = transactions_dummies.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions_dummies)

    def get_df(self):
        return self.dfs[0] if self.dfs else None

# --- Fonctions Helpers (Reste inchangé) ---
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