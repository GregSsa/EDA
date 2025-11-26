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
            # Lecture standard, on laisse Pandas gérer le header
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
        Détection intelligente des 5 formats possibles.
        """
        # 1. Sequential : Présence de parenthèses ( ) dans les données textuelles
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            # On concatène un échantillon des colonnes texte pour tester
            
            sample_text = str(df[str_cols[0]].iloc[0]) + str(df[str_cols[-1]].iloc[0])
            if "(" in sample_text and ")" in sample_text:
                return "Sequential"

        # 2. Wide Checks (One-Hot ou Transposed)
        # S'il y a une majorité de colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
        # Si colonnes numériques > 3 et représentent +50% des colonnes totales -> OneHot
        if len(numeric_cols) > 3 and len(numeric_cols) >= (len(df.columns) / 2):
            return "Wide_OneHot"

        # Si le tableau est plus large que haut (forme transposée typique) ET contient des virgules
        if df.shape[1] > df.shape[0] and df.shape[1] > 2:
            try:
                if "," in str(df.iloc[0, 1]):
                    return "Wide_Transposed"
            except: pass

        # 3. Distinction Long vs Basic
        # "Long" et "Basic" ont souvent peu de colonnes (2 ou 3).
        # Basic : La colonne item contient des séparateurs (ex: "eggs, milk")
        # Long : La colonne item contient un seul mot (ex: "eggs")
        
        # On prend la dernière colonne (supposée être les items)
        last_col = df.columns[-1]
        if df[last_col].dtype == object:
            # On vérifie si on trouve des virgules dans les 5 premières lignes
            sample_items = df[last_col].head(5).astype(str)
            has_separator = sample_items.str.contains(",").any()
            
            if has_separator:
                return "Basic" # Liste d'items : "a,b,c"
            
            # S'il n'y a pas de séparateur, c'est probablement "Long" (une ligne par item)
            # Vérifions s'il y a des IDs dupliqués dans la première colonne (typique du format Long)
            first_col = df.columns[0]
            if df[first_col].duplicated().any():
                return "Long"

        return "Basic" # Fallback

    def _process_transactions(self, transactions, target_column, sep, formatting):
        
        # --- CAS 1 : Long Format (Transaction, Item) ---
        if formatting == "Long":
            # Structure: [ID, Item]
            # On utilise crosstab pour pivoter directement
            
            # Identification des colonnes si pas fournies
            col_id = transactions.columns[0]
            col_item = target_column if target_column in transactions.columns else transactions.columns[1]
            
            # Pivot table (Crosstab est très rapide pour ça)
            # Cela crée une matrice avec ID en index et Items en colonnes
            df_cross = pd.crosstab(transactions[col_id], transactions[col_item])
            
            # Conversion binaire stricte
            df_cross = df_cross.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(df_cross)
            return

        # --- CAS 2 : Sequential (Nettoyage agressif) ---
        elif formatting == "Sequential":
            # On cible la colonne contenant les parenthèses
            if not target_column or target_column not in transactions.columns:
                # Cherche la première colonne avec '('
                for col in transactions.columns:
                    if transactions[col].dtype == object and "(" in str(transactions[col].iloc[0]):
                        target_column = col
                        break
            
            # Stratégie : Aplatir. On enlève ( et ) partout.
            # Ex: "(pain, lait), (beurre)" -> "pain, lait, beurre"
            transactions[target_column] = transactions[target_column].astype(str).str.replace(r'[()]', '', regex=True)
            
            # Maintenant on traite exactement comme du Basic propre
            # Récursion vers Basic
            return self._process_transactions(transactions, target_column, ",", "Basic")

        # --- CAS 3 : Wide Transposed ---
        elif formatting == "Wide_Transposed":
            transactions = transactions.T
            transactions.columns = ["items"] 
            return self._process_transactions(transactions, "items", sep, "Basic")

        # --- CAS 4 : Wide One-Hot ---
        elif formatting == "Wide_OneHot":
            numeric_cols = transactions.select_dtypes(include=[np.number, bool]).columns.tolist()
            if "Transaction" in transactions.columns:
                transactions = transactions.set_index("Transaction")
            elif "client_id" in transactions.columns:
                transactions = transactions.set_index("client_id")
            
            transactions = transactions[numeric_cols]
            transactions = transactions.fillna(0).applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions)
            return

        # --- CAS 5 : Basic ---
        else:
            if not target_column or target_column not in transactions.columns:
                target_column = transactions.columns[-1]

            transactions = transactions[transactions[target_column].apply(lambda x: isinstance(x, str))]
            
            # Nettoyage des quotes parasites avant le split
            transactions[target_column] = transactions[target_column].str.replace('"', '').str.replace("'", "")
            
            # Split
            transactions[target_column] = transactions[target_column].apply(lambda x: [i.strip() for i in x.split(sep)])
            
            transactions_exploded = transactions[target_column].explode()
            transactions_dummies = pd.get_dummies(transactions_exploded)
            transactions_dummies = transactions_dummies.groupby(transactions_dummies.index).sum()
            
            transactions_dummies = transactions_dummies.applymap(lambda x: 1 if x > 0 else 0)
            self.dfs.append(transactions_dummies)

    def get_df(self):
        return self.dfs[0] if self.dfs else None

# --- Fonctions Helpers (Inchangées) ---
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