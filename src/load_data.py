"""Utilities to load and convert transaction datasets.
Updated to handle Long (Transactional) format.
"""
from typing import Optional, List
import pandas as pd
import re

class TransactionDf:
    """Light wrapper to load transaction CSVs in various formats."""

    def __init__(self, file_path: str, header: bool = False, target_column: Optional[str] = None,
                 id_column: Optional[str] = None, separator: str = ",", formatting: str = "Basic"):
        self.file_paths = [file_path]
        self.headers = [header]
        self.target_columns = [target_column]
        self.id_columns = [id_column] # Nouvelle propriété
        self.separators = [separator]
        self.dfs: List[pd.DataFrame] = []
        self.load_transactions(file_path, header, target_column, id_column, separator, formatting)

    def load_transaction_csv(self, file_path: str, header: bool, target_column: Optional[str],
                           id_column: Optional[str], sep: str, formatting: str) -> pd.DataFrame:
        if header:
            transactions = pd.read_csv(file_path)
        else:
            transactions = pd.read_csv(file_path, header=None)

        if formatting == "Wide":
            transactions = transactions.T
            formatting = "Basic"

        if formatting == "Long":
            # Nouveau format : Transaction ID | Item
            if target_column is None or id_column is None:
                raise ValueError("Format Long requires both target_column (Items) and id_column (Transaction IDs)")

            # On utilise crosstab pour pivoter : Lignes=ID, Colonnes=Items
            # Cela fait automatiquement le One-Hot Encoding et le groupement
            transactions = pd.crosstab(transactions[id_column], transactions[target_column])
            # On s'assure que c'est binaire (0 ou 1)
            transactions = (transactions > 0).astype(int)
            return transactions

        if formatting == "Basic":
            if target_column is None:
                if len(transactions.columns) == 1:
                    target_column = transactions.columns[0]
                else:
                    raise ValueError("target_column must be provided for Basic formatting")

            if sep:
                transactions = transactions[transactions[target_column].apply(lambda x: isinstance(x, str))]
                transactions[target_column] = transactions[target_column].apply(lambda x: [i.strip() for i in x.split(sep)])

            transactions_exploded = transactions[target_column].explode()
            transactions_dummies = pd.get_dummies(transactions_exploded)
            transactions_dummies.columns = [f'{col}' for col in transactions_dummies.columns] # Removed 0_ prefix for consistency
            transactions_dummies = transactions_dummies.groupby(transactions_dummies.index).sum()
            transactions = transactions.drop(columns=[target_column]).join(transactions_dummies)

        elif formatting == "Sequential":
            # ... (Code existant inchangé pour sequential) ...
             if target_column is None:
                raise ValueError("target_column required for Sequential formatting")

             all_items = set()
             parsed_sequences = []

             for seq in transactions[target_column]:
                all_transactions = re.findall(r"\((.*?)\)", str(seq))
                steps = [t.split(",") for t in all_transactions]
                steps = [[item.strip() for item in step if item.strip()] for step in steps]
                parsed_sequences.append(steps)
                for step in steps:
                    all_items.update(step)

             all_items = sorted(all_items)
             output_rows = []
             for i in range(len(transactions)):
                client_id = transactions.iloc[i].get('client_id', i)
                steps = parsed_sequences[i]
                row = {"client_id": client_id}
                max_steps = len(steps)
                for step_i in range(max_steps):
                    for item in all_items:
                        row[f"{step_i}_{item}"] = 1 if item in steps[step_i] else 0
                output_rows.append(row)
             transactions = pd.DataFrame(output_rows).fillna(0)

        return transactions

    def load_transactions(self, file_path: str, header: bool = True, target_column: Optional[str] = None,
                          id_column: Optional[str] = None, sep: str = ",", formatting: str = "Basic") -> None:
        res = None
        if file_path.lower().endswith('.csv'):
            try:
                res = self.load_transaction_csv(file_path, header, target_column, id_column, sep, formatting)
            except ValueError as e:
                print(f"Error: {e}")

        if res is not None:
            self.dfs.append(res)

    # ... (Le reste des méthodes displays, combine, size reste inchangé) ...
    def displays(self) -> None:
        for df in self.dfs:
            try:
                from IPython.display import display
                display(df)
            except Exception:
                print(df.head())

    def combine(self, indexes: List[int] = [], column_to_check: Optional[str] = None) -> Optional[pd.DataFrame]:
        if len(self.dfs) == 2:
            indexes = [0, 1]
        elif (indexes == [] or max(indexes) >= len(self.dfs)):
            return None

        res = None
        if column_to_check is not None:
            df_merged = pd.merge(self.dfs[indexes[0]], self.dfs[indexes[1]], on=column_to_check, how='outer', suffixes=('_1', '_2'))
            df_merged = df_merged.fillna(0)
            numeric_cols = [c for c in df_merged.columns if c != column_to_check]
            df_merged[numeric_cols] = df_merged[numeric_cols].astype(int)

            all_columns = set(self.dfs[indexes[0]].columns).union(self.dfs[indexes[1]].columns) - {column_to_check}
            for col in all_columns:
                cols_to_sum = [c for c in df_merged.columns if c == col or c.startswith(col + "_")]
                if cols_to_sum:
                    df_merged[col] = df_merged[cols_to_sum].sum(axis=1)
                    if len(cols_to_sum) == 2:
                        df_merged.drop(columns=cols_to_sum, inplace=True, errors='ignore')
            res = df_merged
            res = res[sorted(res.columns)]

        return res

    def size(self) -> int:
        return len(self.dfs)

def load_transactions_simple(file_path: str) -> List[List[str]]:
    transactions = []
    with open(file_path, 'r') as file:
        for line in file:
            transaction = line.strip().split()
            transactions.append(transaction)
    return transactions