import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# Importation des modules locaux
from load_data import TransactionDf, load_transactions_simple
from mining import run_fpgrowth, generate_rules
from scoring import compute_composite_scores
from sampling import weighted_sampling, light_mcmc
from feedback import apply_like, apply_dislike, reset_feedback

st.set_page_config(page_title="Fouille Interactive de Motifs", layout="wide")


# --- 1. Gestion du Cache ---
@st.cache_resource
def load_and_process_data(file_path, formatting, min_support, target_col=None, id_col=None):
    """Charge les données et extrait les règles initiales.
    Cette fonction n'est exécutée que si les arguments changent.
    """
    # Chargement des transactions
    if file_path.lower().endswith('.txt'):
        transactions = load_transactions_simple(file_path)
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    else:
        # Utilisation de la classe TransactionDf pour CSV
        T = TransactionDf(file_path, header=True, target_column=target_col, id_column=id_col, formatting=formatting)
        if T.size() == 0:
            return None
        df_transactions = T.dfs[0]

    # Extraction des motifs (Partie coûteuse en temps)
    frequent_itemsets = run_fpgrowth(df_transactions, min_support=min_support)

    # Seuil de confiance très bas pour voir un maximum de règles au début
    rules = generate_rules(frequent_itemsets, metric='confidence', min_threshold=0.01)

    if rules.empty:
        return pd.DataFrame()  # Retourne vide proprement

    # Calcul initial des scores
    rules_scored = compute_composite_scores(rules)

    # Ajout d'une colonne ID unique pour le suivi dans l'interface
    rules_scored['rule_id'] = rules_scored.index
    return rules_scored


# --- 2. Interface Utilisateur ---
st.title("Exploration Interactive de Motifs")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("1. Chargement")
    uploaded_file = st.file_uploader("Fichier de transactions", type=['csv', 'txt'])

    # Ajout du format Long pour supporter le fichier sample_transactions.csv
    format_option = st.selectbox("Format", ["Basic", "Long", "Wide", "Sequential"])

    # Sélecteurs dynamiques de colonnes
    target_col = None
    id_col = None

    if uploaded_file is not None and uploaded_file.name.endswith('.csv'):
        # Lecture de l'en-tête pour les options
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file, nrows=0)
        cols = df_preview.columns.tolist()

        if format_option == "Basic":
            target_col = st.selectbox("Colonne des articles (ex: 'pain, lait')", cols)

        elif format_option == "Long":
            st.info("Format Long : 1 ligne = 1 article")
            id_col = st.selectbox("Colonne ID Transaction", cols, index=0)
            target_col = st.selectbox("Colonne Article", cols, index=1 if len(cols) > 1 else 0)

        uploaded_file.seek(0)  # Important : Rembobiner le fichier après lecture

    st.header("2. Extraction")
    min_sup = st.slider("Support Minimum", 0.01, 0.5, 0.05)  # Valeur par défaut 5%

    st.header("3. Scoring")
    w_lift = st.slider("Poids Lift", 0.0, 1.0, 0.5)
    w_conf = st.slider("Poids Confiance", 0.0, 1.0, 0.3)
    w_supp = st.slider("Poids Support", 0.0, 1.0, 0.2)

    st.header("4. Echantillonnage")
    k_samples = st.number_input("Nombre de motifs (k)", 1, 50, 5)

    btn_reset = st.button("Réinitialiser Feedback")

# Initialisation de l'état de session si nécessaire
if 'pool_rules' not in st.session_state:
    st.session_state['pool_rules'] = None

# Logique principale
if uploaded_file is not None:
    # Sauvegarde temporaire du fichier uploadé pour le traiter
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Vérification que les colonnes nécessaires sont sélectionnées pour le format Long
    ready_to_load = True
    if format_option == "Long" and (not id_col or not target_col):
        ready_to_load = False

    if ready_to_load:
        try:
            # Chargement (mis en cache)
            rules_initial = load_and_process_data(f"temp_{uploaded_file.name}", format_option, min_sup, target_col,
                                                  id_col)

            if rules_initial is not None and not rules_initial.empty:
                # Initialisation ou mise à jour du pool dans l'état de session
                if st.session_state['pool_rules'] is None:
                    st.session_state['pool_rules'] = rules_initial.copy()

                # Récupération du pool courant
                current_pool = st.session_state['pool_rules']

                # Mise à jour dynamique des scores composites selon les sliders
                current_pool['composite_score'] = (
                        w_lift * current_pool['lift'] +
                        w_conf * current_pool['confidence'] +
                        w_supp * current_pool['support']
                )
                current_pool['final_sampling_weight'] = current_pool['composite_score'] * current_pool[
                    'feedback_weight']

                if btn_reset:
                    current_pool = reset_feedback(current_pool)
                    st.success("Feedback réinitialisé.")

                st.success(f"Pool chargé : {len(current_pool)} règles trouvées.")

                # Echantillonnage
                try:
                    sampled_df = weighted_sampling(current_pool, k=k_samples, with_replacement=False)

                    st.subheader("Motifs Suggérés")

                    # Affichage interactif avec boutons Like/Dislike
                    for idx, row in sampled_df.iterrows():
                        col1, col2, col3 = st.columns([4, 1, 1])

                        with col1:
                            # Formatage textuel propre des règles
                            ant_str = ", ".join(list(row['antecedents']))
                            con_str = ", ".join(list(row['consequents']))
                            st.markdown(f"**{ant_str}** --> **{con_str}**")
                            st.caption(
                                f"Lift: {row['lift']:.2f} | Conf: {row['confidence']:.2f} | Supp: {row['support']:.2f}")

                        # Boutons de feedback textuels
                        with col2:
                            if st.button("Like", key=f"like_{idx}"):
                                st.session_state['pool_rules'] = apply_like(current_pool, idx=idx, factor=1.5)
                                st.rerun()  # Recharger pour afficher le nouvel état

                        with col3:
                            if st.button("Dislike", key=f"dislike_{idx}"):
                                st.session_state['pool_rules'] = apply_dislike(current_pool, idx=idx, factor=0.5)
                                st.rerun()

                except Exception as e:
                    st.warning(f"Impossible d'échantillonner : {e}")

            else:
                st.warning("Aucune règle trouvée. Essayez de baisser le 'Support Minimum'.")
        except Exception as e:
            st.error(f"Erreur technique : {e}")
    else:
        st.info("Veuillez sélectionner les colonnes dans le menu latéral.")
else:
    st.info("Veuillez charger un fichier pour commencer.")