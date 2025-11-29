import streamlit as st
import pandas as pd
import time
import plotly.express as px
from mlxtend.frequent_patterns import fpgrowth, association_rules
# Assurez-vous que utils.py est bien dans le dossier
from utils import TransactionDf, calculate_composite_score, light_mcmc, calculate_diversity, pattern_sample_mcmc

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="EDA Project - Pattern Mining",
    layout="wide",
    page_icon="⛏️",
    initial_sidebar_state="expanded"
)

st.title("⛏️ Projet EDA : Fouille Interactive de Motifs")

# --- GESTION ÉTAT & NETTOYAGE AUTOMATIQUE ---
# Cette section doit être tout en haut pour intercepter le changement de fichier
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None


def clear_cache():
    """Nettoie toute la mémoire de session liée aux données"""
    keys_to_clear = ['pool_rules', 'feedback_weights', 'last_sample', 'exec_time', 'processor']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# --- SIDEBAR (PARAMÈTRES) ---
with st.sidebar:
    st.header("1. Configuration")

    # Widget d'upload
    uploaded_file = st.file_uploader("Charger un CSV", type=["csv"])

    # DÉTECTION DE CHANGEMENT DE FICHIER
    # Si le fichier change (ou devient None), on reset tout.
    if uploaded_file != st.session_state['last_uploaded_file']:
        clear_cache()
        st.session_state['last_uploaded_file'] = uploaded_file
        # On force un rerun immédiat pour que l'interface se mette à jour à vide
        st.rerun()

    format_option = st.selectbox("Format", ["Auto", "Basic", "Long", "Wide", "Sequential"])
    sep_option = ','
    target_col = ""

    st.markdown("---")
    st.header("2. Paramètres Extraction")
    algo_choice = st.radio("Méthode", ["Exhaustive (FP-Growth)", "Output Sampling (MCMC)"])

    with st.expander("🛠️ Paramètres avancés", expanded=False):
        if algo_choice == "Exhaustive (FP-Growth)":
            min_support = st.slider("Support Min", 0.001, 0.5, 0.05, 0.001, format="%.3f")
            min_confidence = st.slider("Confiance Min", 0.0, 1.0, 0.4, 0.05)
        else:
            sampling_iterations = st.number_input("Itérations MCMC", 1000, 100000, 5000, 1000)
            sampling_min_support = st.slider("Support Min (sampling)", 0.0, 0.5, 0.005, 0.01)
            sampling_max_rules = st.number_input("Nb max règles", 50, 5000, 500, 50)
            sampling_interest = st.selectbox("Mesure d'intérêt",
                                             ["lift", "confidence", "support", "composite (lift*confidence)"])

    st.markdown("---")
    st.header("3. Scoring & Poids")
    w_support = st.slider("Poids Support", 0.0, 1.0, 0.2)
    w_lift = st.slider("Poids Lift", 0.0, 1.0, 0.2)
    w_conf = st.slider("Poids Confiance", 0.0, 1.0, 0.2)
    w_surprise = st.slider("Poids Surprise", 0.0, 1.0, 0.2)

    st.markdown("---")
    st.header("4. Échantillonnage")
    k_samples = st.number_input("Taille échantillon (k)", 1, 50, 5)
    replace_strategy = st.checkbox("Avec Remise", value=True)
    random_seed = st.number_input("Seed", value=42)

# --- INIT VARIABLES SESSION (si elles n'existent pas après le clear) ---
if 'pool_rules' not in st.session_state: st.session_state['pool_rules'] = None
if 'feedback_weights' not in st.session_state: st.session_state['feedback_weights'] = {}
if 'last_sample' not in st.session_state: st.session_state['last_sample'] = None
if 'exec_time' not in st.session_state: st.session_state['exec_time'] = 0.0
if 'processor' not in st.session_state: st.session_state['processor'] = None

# --- MAIN LOGIC ---
if uploaded_file:
    try:
        # Chargement et Preprocessing
        # On utilise une clé de cache basée sur le file_id pour éviter de recharger pandas à chaque clic
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)


        raw_df = load_data(uploaded_file)
        tgt = target_col if target_col.strip() != "" else None

        # On ne réinstancie le processor que s'il est vide (optimisation)
        if st.session_state['processor'] is None:
            processor = TransactionDf(dataframe=raw_df, target_column=tgt, separator=sep_option,
                                      formatting=format_option)
            st.session_state['processor'] = processor

        df_encoded = st.session_state['processor'].get_df()

        if df_encoded is None or df_encoded.empty:
            st.error("Erreur : Impossible de traiter le fichier. Vérifiez le format.")
        else:
            # --- STRUCTURE EN ONGLETS ---
            tab_data, tab_mining, tab_interactive = st.tabs(
                ["📊 Données", "⚙️ Extraction & Pool", "🎮 Échantillonnage Interactif"])

            # ==========================
            # TAB 1: DONNÉES
            # ==========================
            with tab_data:
                st.subheader("Aperçu du Dataset")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Transactions:** `{df_encoded.shape[0]}`")
                    st.markdown(f"**Items uniques:** `{df_encoded.shape[1]}`")
                    st.dataframe(df_encoded.head(10), use_container_width=True)
                with col2:
                    st.markdown("**Top 20 Items les plus fréquents**")
                    item_counts = df_encoded.sum().sort_values(ascending=False).head(20).reset_index()
                    item_counts.columns = ['Item', 'Frequence']
                    fig = px.bar(item_counts, x='Item', y='Frequence', color='Frequence',
                                 color_continuous_scale='Viridis', title="Distribution des Items")
                    st.plotly_chart(fig, use_container_width=True)

            # ==========================
            # TAB 2: EXTRACTION
            # ==========================
            with tab_mining:
                c_head, c_btn = st.columns([3, 1])
                with c_head:
                    st.subheader("Génération du Pool de Règles")
                with c_btn:
                    launch_btn = st.button("🚀 Lancer l'Extraction", type="primary", use_container_width=True)

                # Logique d'extraction
                if launch_btn:
                    start_time = time.time()
                    with st.spinner("Algorithme en cours d'exécution..."):
                        if algo_choice == "Exhaustive (FP-Growth)":
                            df_for_mining = df_encoded
                            frequent = fpgrowth(df_for_mining.astype(bool), min_support=min_support, use_colnames=True)
                            if frequent.empty:
                                st.warning("Aucun motif trouvé. Essayez de baisser le support minimum.")
                            else:
                                rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
                                if rules.empty:
                                    st.warning("Aucune règle trouvée. Essayez de baisser la confiance.")
                                else:
                                    end_time = time.time()
                                    st.session_state['exec_time'] = end_time - start_time
                                    rules['length'] = rules['antecedents'].apply(len) + rules['consequents'].apply(len)
                                    if 'antecedent support' in rules.columns:
                                        rules.rename(columns={'antecedent support': 'coverage'}, inplace=True)

                                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                                    rules['rule_id'] = rules.index

                                    st.session_state['pool_rules'] = rules
                                    st.session_state['feedback_weights'] = {i: 1.0 for i in rules.index}
                                    st.success(
                                        f"Extraction terminée en {st.session_state['exec_time']:.3f} sec : {len(rules)} règles.")
                        else:
                            st.info("Mode Sampling MCMC activé.")
                            sampled_rules = pattern_sample_mcmc(
                                df_encoded,
                                iterations=int(sampling_iterations),
                                min_support=float(sampling_min_support),
                                max_rules=int(sampling_max_rules),
                                random_seed=int(random_seed)
                            )
                            if sampled_rules.empty:
                                st.warning("Aucune règle trouvée via Sampling.")
                            else:
                                end_time = time.time()
                                st.session_state['exec_time'] = end_time - start_time
                                if 'coverage' not in sampled_rules.columns and 'support' in sampled_rules.columns:
                                    sampled_rules['coverage'] = sampled_rules['support']
                                sampled_rules['antecedents_str'] = sampled_rules['antecedents'].apply(
                                    lambda x: ', '.join(list(x)))
                                sampled_rules['consequents_str'] = sampled_rules['consequents'].apply(
                                    lambda x: ', '.join(list(x)))
                                sampled_rules['rule_id'] = sampled_rules.index
                                st.session_state['pool_rules'] = sampled_rules
                                st.session_state['feedback_weights'] = {i: 1.0 for i in sampled_rules.index}
                                st.success(
                                    f"Sampling terminé en {st.session_state['exec_time']:.3f} sec : {len(sampled_rules)} règles.")

                # Visualisation du Pool (Uniquement si pool existe)
                if st.session_state['pool_rules'] is not None:
                    rules_df = st.session_state['pool_rules'].copy()

                    st.divider()
                    st.markdown("#### 🔍 Explorer les règles")

                    # Filtre produits
                    all_items_in_rules = sorted(list(set(
                        [item for sublist in rules_df['antecedents'] for item in sublist] +
                        [item for sublist in rules_df['consequents'] for item in sublist]
                    )))

                    col_filter, col_dl = st.columns([3, 1])
                    with col_filter:
                        selected_items = st.multiselect("Filtrer par produit (contient...)", all_items_in_rules)

                    if selected_items:
                        mask = rules_df.apply(lambda r: any(i in r['antecedents'] for i in selected_items) or
                                                        any(i in r['consequents'] for i in selected_items), axis=1)
                        rules_filtered = rules_df[mask]
                    else:
                        rules_filtered = rules_df

                    with col_dl:
                        csv = rules_filtered.to_csv(index=False).encode('utf-8')
                        st.download_button("💾 Télécharger CSV", data=csv, file_name='regles_association.csv',
                                           mime='text/csv')

                    st.caption(f"Affichage de **{len(rules_filtered)}** règles sur {len(rules_df)}.")

                    col_viz1, col_viz2 = st.columns([1.5, 1])
                    with col_viz1:
                        if not rules_filtered.empty:
                            fig_scatter = px.scatter(
                                rules_filtered, x="support", y="confidence", size="lift", color="lift",
                                hover_data=["antecedents_str", "consequents_str"],
                                title="Support vs Confiance (Taille=Lift)", color_continuous_scale="RdBu_r"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Aucune règle ne correspond aux filtres.")

                    with col_viz2:
                        st.dataframe(
                            rules_filtered[['antecedents_str', 'consequents_str', 'lift', 'confidence', 'support']],
                            height=400, hide_index=True, use_container_width=True
                        )

                    # Heatmap
                    if not rules_filtered.empty and len(rules_filtered) < 200:
                        st.divider()
                        st.markdown("#### 🔥 Heatmap des Associations (Top 50)")
                        top_rules = rules_filtered.sort_values('lift', ascending=False).head(50)
                        top_rules['ant_short'] = top_rules['antecedents_str'].apply(
                            lambda x: x[:15] + '..' if len(x) > 15 else x)
                        top_rules['cons_short'] = top_rules['consequents_str'].apply(
                            lambda x: x[:15] + '..' if len(x) > 15 else x)
                        try:
                            pivot_df = top_rules.pivot_table(index='ant_short', columns='cons_short', values='lift')
                            fig_heat = px.imshow(pivot_df, text_auto=".1f", aspect="auto",
                                                 color_continuous_scale="Viridis",
                                                 title="Matrice Antécédents (Y) vs Conséquents (X) - Valeur = Lift")
                            st.plotly_chart(fig_heat, use_container_width=True)
                        except:
                            pass

            # ==========================
            # TAB 3: INTERACTIF
            # ==========================
            with tab_interactive:
                if st.session_state['pool_rules'] is None:
                    st.info("⚠️ Veuillez d'abord lancer l'extraction dans l'onglet précédent.")
                else:
                    rules_df = st.session_state['pool_rules']

                    # Calculs
                    rules_df['composite_score'] = calculate_composite_score(rules_df, w_support, w_lift, w_conf,
                                                                            w_surprise, 0.2)
                    rules_df['feedback_weight'] = rules_df.index.map(st.session_state['feedback_weights']).fillna(1.0)
                    rules_df['final_sampling_weight'] = rules_df['composite_score'] * rules_df['feedback_weight']

                    col_control, col_results = st.columns([1, 2.5])

                    with col_control:
                        st.markdown("### 🎮 Contrôles")
                        if st.button("🎲 Générer un nouvel échantillon", type="primary", use_container_width=True):
                            sample = light_mcmc(rules_df, k=int(k_samples), replace=replace_strategy,
                                                random_seed=int(random_seed))
                            st.session_state['last_sample'] = sample

                        # --- FIX BUG 1 : RESET FEEDBACK ---
                        if st.button("♻️ Reset Feedback", use_container_width=True):
                            st.session_state['feedback_weights'] = {}
                            st.toast("Feedback réinitialisé !", icon="↺")
                            time.sleep(0.5)  # Petit délai pour l'UX
                            st.rerun()

                        st.markdown("---")
                        st.markdown("**📊 Distribution des Scores**")
                        fig_dist = px.histogram(rules_df, x="final_sampling_weight", nbins=20,
                                                title="Pool (Gris) vs Sample (Bleu)",
                                                color_discrete_sequence=['lightgray'], opacity=0.6)
                        fig_dist.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))

                        if st.session_state['last_sample'] is not None:
                            sample_hist = px.histogram(st.session_state['last_sample'], x="final_sampling_weight",
                                                       nbins=20,
                                                       color_discrete_sequence=['blue'])
                            fig_dist.add_trace(sample_hist.data[0])
                        st.plotly_chart(fig_dist, use_container_width=True)

                    with col_results:
                        st.markdown("### 🔎 Échantillon & Feedback")

                        if st.session_state['last_sample'] is not None:
                            sample = st.session_state['last_sample']

                            total_feedback = sum(1 for w in st.session_state['feedback_weights'].values() if w != 1.0)
                            positive_feedback = sum(1 for w in st.session_state['feedback_weights'].values() if w > 1.0)
                            acceptance_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
                            diversity = calculate_diversity(sample)
                            coverage_global = st.session_state['processor'].calculate_global_coverage(sample)

                            kpi1, kpi2, kpi3 = st.columns(3)
                            kpi1.metric("Taux Like", f"{acceptance_rate:.0f}%")
                            kpi2.metric("Diversité", f"{diversity:.2f}")
                            kpi3.metric("Couverture", f"{coverage_global * 100:.1f}%")

                            st.divider()

                            # --- FIX BUG 2 : COULEUR DU TEXTE ---
                            # On force color: #1f2937 (gris foncé presque noir) pour la lisibilité
                            for i, row in sample.iterrows():
                                rid = row['rule_id'] if 'rule_id' in row else row.name
                                cw = st.session_state['feedback_weights'].get(rid, 1.0)

                                # Style CSS conditionnel
                                base_style = "padding: 10px; border-radius: 5px; color: #1f2937;"
                                if cw > 1.0:
                                    bg_style = f"background-color: #dcfce7; border-left: 5px solid #22c55e; {base_style}"  # Vert
                                elif cw < 1.0:
                                    bg_style = f"background-color: #fee2e2; border-left: 5px solid #ef4444; {base_style}"  # Rouge
                                else:
                                    bg_style = f"background-color: #f3f4f6; border-left: 5px solid #9ca3af; {base_style}"  # Gris neutre

                                with st.container():
                                    c_txt, c_vals, c_act = st.columns([3, 1.5, 1.5])

                                    with c_txt:
                                        # Injection HTML avec couleur forcée
                                        st.markdown(
                                            f"<div style='{bg_style}'><b>{row['antecedents_str']} ➝ {row['consequents_str']}</b></div>",
                                            unsafe_allow_html=True)

                                    with c_vals:
                                        st.caption(f"Lift: {row['lift']:.2f} | Conf: {row['confidence']:.2f}")

                                    with c_act:
                                        b_col1, b_col2, b_col3 = st.columns(3)
                                        if cw > 1.0:
                                            b_col1.button("👍", key=f"l_{i}", disabled=True)
                                            if b_col2.button("↺", key=f"r_{i}"):
                                                st.session_state['feedback_weights'][rid] = 1.0
                                                st.rerun()
                                        elif cw < 1.0:
                                            if b_col2.button("↺", key=f"r_{i}"):
                                                st.session_state['feedback_weights'][rid] = 1.0
                                                st.rerun()
                                            b_col3.button("👎", key=f"d_{i}", disabled=True)
                                        else:
                                            if b_col1.button("👍", key=f"l_{i}"):
                                                st.session_state['feedback_weights'][rid] = 1.5
                                                st.rerun()
                                            if b_col3.button("👎", key=f"d_{i}"):
                                                st.session_state['feedback_weights'][rid] = 0.5
                                                st.rerun()
                                    st.markdown("---")
                        else:
                            st.info("👈 Cliquez sur 'Générer' pour commencer.")

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")