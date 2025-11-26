import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from utils import TransactionDf, calculate_composite_score, light_mcmc, calculate_diversity

st.set_page_config(page_title="EDA Project - Pattern Mining", layout="wide")

st.title("Projet EDA : Fouille Interactive de Motifs")

# --- SIDEBAR ---
st.sidebar.header("1. Configuration")
uploaded_file = st.sidebar.file_uploader("Charger un CSV", type=["csv"])
format_option = st.sidebar.selectbox("Format", ["Auto", "Basic", "Long", "Wide", "Sequential"])
# sep_option = st.sidebar.text_input("Séparateur (Basic)", value=",")
sep_option = ','
# target_col = st.sidebar.text_input("Colonne cible / Item (optionnel)")
target_col = ""

st.sidebar.markdown("---")
st.sidebar.header("2. Paramètres Extraction")
# Choix de l'algorithme (Pour répondre à la consigne 5, même si simulé)
algo_choice = st.sidebar.radio("Algorithme", ["Exhaustive (FP-Growth)", "Output Sampling (Simulé)"])
min_support = st.sidebar.slider("Support Min", 0.01, 1.0, 0.1, 0.01)
min_confidence = st.sidebar.slider("Confiance Min", 0.0, 1.0, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("3. Scoring & Poids")
w_support = st.sidebar.slider("Poids Support", 0.0, 1.0, 0.2)
w_lift = st.sidebar.slider("Poids Lift", 0.0, 1.0, 0.2)
w_conf = st.sidebar.slider("Poids Confiance", 0.0, 1.0, 0.2)
w_surprise = st.sidebar.slider("Poids Surprise", 0.0, 1.0, 0.2)

st.sidebar.markdown("---")
st.sidebar.header("4. Échantillonnage")
k_samples = st.sidebar.number_input("Taille échantillon (k)", 1, 50, 5)
replace_strategy = st.sidebar.checkbox("Avec Remise", value=True)
random_seed = st.sidebar.number_input("Seed", value=42)

# --- STATE MANAGEMENT ---
if 'pool_rules' not in st.session_state: st.session_state['pool_rules'] = None
if 'feedback_weights' not in st.session_state: st.session_state['feedback_weights'] = {}
if 'last_sample' not in st.session_state: st.session_state['last_sample'] = None
if 'exec_time' not in st.session_state: st.session_state['exec_time'] = 0.0
if 'processor' not in st.session_state: st.session_state['processor'] = None

def reset_feedback():
    st.session_state['feedback_weights'] = {}
    st.success("Feedback reset !")

# --- MAIN ---
if uploaded_file:
    try:
        # On ne recharge pas le DF à chaque interaction si possible, mais Streamlit rerun tout le script.
        # Optimisation simple : lecture
        raw_df = pd.read_csv(uploaded_file)
        tgt = target_col if target_col.strip() != "" else None
        
        # Instanciation processor
        processor = TransactionDf(dataframe=raw_df, target_column=tgt, separator=sep_option, formatting=format_option)
        st.session_state['processor'] = processor # Stockage pour calcul coverage
        df_encoded = processor.get_df()

        if df_encoded is None or df_encoded.empty:
            st.error("Erreur : Impossible de traiter le fichier.")
        else:
            st.subheader("1. Aperçu des Données")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"Transac: {df_encoded.shape[0]} | Items: {df_encoded.shape[1]}")
                st.dataframe(df_encoded.head(5), height=150)
            with col2:
                item_counts = df_encoded.sum().sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                sns.barplot(x=item_counts.index, y=item_counts.values, ax=ax, palette="viridis")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("2. Extraction (Pool P)")
            
            if st.button("Lancer l'Extraction"):
                start_time = time.time()
                with st.spinner("Extraction..."):
                    # Simulation consigne 5 : Si Output Sampling, on échantillonne les données AVANT extraction
                    if algo_choice == "Output Sampling (Simulé)":
                        df_for_mining = df_encoded.sample(frac=0.5, random_state=int(random_seed)) # 50% des données
                        st.info("Mode Échantillonnage : Extraction sur 50% de la base.")
                    else:
                        df_for_mining = df_encoded

                    frequent = fpgrowth(df_for_mining.astype(bool), min_support=min_support, use_colnames=True)
                    if frequent.empty:
                        st.warning("Aucun motif trouvé.")
                    else:
                        rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
                        if rules.empty:
                            st.warning("Aucune règle trouvée.")
                        else:
                            end_time = time.time()
                            st.session_state['exec_time'] = end_time - start_time
                            
                            rules['length'] = rules['antecedents'].apply(len) + rules['consequents'].apply(len)
                            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules['rule_id'] = rules.index
                            st.session_state['pool_rules'] = rules
                            st.session_state['feedback_weights'] = {i: 1.0 for i in rules.index}
                            st.success(f"Extraction terminée en {st.session_state['exec_time']:.3f} sec : {len(rules)} règles.")

            if st.session_state['pool_rules'] is not None:
                rules_df = st.session_state['pool_rules']
                
                with st.expander(f"Voir le pool complet"):
                    st.dataframe(rules_df[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']])

                st.markdown("---")
                st.subheader("3. Échantillonnage Interactif")

                rules_df['composite_score'] = calculate_composite_score(rules_df, w_support, w_lift, w_conf, w_surprise, 0.2)
                rules_df['feedback_weight'] = rules_df.index.map(st.session_state['feedback_weights']).fillna(1.0)
                rules_df['final_sampling_weight'] = rules_df['composite_score'] * rules_df['feedback_weight']

                c_sample, c_viz = st.columns([0.6, 0.4])

                with c_sample:
                    if st.button("🎲 Générer Échantillon"):
                        sample = light_mcmc(rules_df, k=int(k_samples), replace=replace_strategy, random_seed=int(random_seed))
                        st.session_state['last_sample'] = sample
                    
                    if st.session_state['last_sample'] is not None:
                        sample = st.session_state['last_sample']
                        
                        # --- SECTION 4 : AFFICHAGE METRIQUES ---
                        st.markdown("#### 📊 Métriques d'Évaluation (Consigne 4)")
                        
                        # 1. Taux d'Acceptation
                        total_feedback = 0
                        positive_feedback = 0
                        for w in st.session_state['feedback_weights'].values():
                            if w != 1.0: total_feedback += 1
                            if w > 1.0: positive_feedback += 1
                        acceptance_rate = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
                        
                        # 2. Diversité & Coverage
                        diversity = calculate_diversity(sample)
                        coverage_global = st.session_state['processor'].calculate_global_coverage(sample)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Temps Réponse", f"{st.session_state['exec_time']:.3f}s")
                        m2.metric("Taux Like", f"{acceptance_rate:.0f}%")
                        m3.metric("Diversité", f"{diversity:.2f}")
                        m4.metric("Couverture", f"{coverage_global*100:.1f}%")
                        
                        st.divider()
                        st.write("### Votre Sélection :")
                        
                        for i, row in sample.iterrows():
                            rid = row['rule_id'] if 'rule_id' in row else row.name
                            cw = st.session_state['feedback_weights'].get(rid, 1.0)
                            
                            icon = "⚪"
                            if cw > 1.0: icon = "✅"
                            elif cw < 1.0: icon = "❌"

                            with st.container():
                                cs = st.columns([0.7, 0.15, 0.15])
                                cs[0].markdown(f"**{icon} {row['antecedents_str']} $\\rightarrow$ {row['consequents_str']}**")
                                cs[0].caption(f"Score: {row['composite_score']:.2f}")

                                if cw > 1.0:
                                    cs[1].button("👍", key=f"d_{i}", disabled=True)
                                    if cs[2].button("Rst", key=f"r_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.0
                                        st.rerun()
                                elif cw < 1.0:
                                    if cs[1].button("Rst", key=f"r_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.0
                                        st.rerun()
                                    cs[2].button("👎", key=f"d2_{i}", disabled=True)
                                else:
                                    if cs[1].button("👍", key=f"lk_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.5
                                        st.rerun()
                                    if cs[2].button("👎", key=f"dl_{i}"):
                                        st.session_state['feedback_weights'][rid] = 0.5
                                        st.rerun()
                                st.divider()
                        
                        if st.button("Reset Avis"): reset_feedback(); st.rerun()

                with c_viz:
                    st.write("### Distribution")
                    fig2, ax2 = plt.subplots()
                    sns.histplot(rules_df['final_sampling_weight'], kde=True, ax=ax2, color="gray", alpha=0.3, label="Pool")
                    if st.session_state['last_sample'] is not None:
                        sns.histplot(st.session_state['last_sample']['final_sampling_weight'], kde=True, ax=ax2, color="blue", label="Sample")
                    ax2.legend()
                    st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erreur : {e}")