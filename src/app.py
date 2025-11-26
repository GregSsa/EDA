import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from utils import TransactionDf, calculate_composite_score, light_mcmc

st.set_page_config(page_title="EDA Project - Pattern Mining", layout="wide")

st.title("Projet EDA : Fouille Interactive de Motifs")

# --- SIDEBAR ---
st.sidebar.header("1. Configuration")
uploaded_file = st.sidebar.file_uploader("Charger un CSV", type=["csv"])

# Ajout de "Long" dans la liste
format_option = st.sidebar.selectbox("Format", ["Auto", "Basic", "Long", "Wide", "Sequential"])
sep_option = st.sidebar.text_input("Séparateur (Basic)", value=",")
target_col = st.sidebar.text_input("Colonne cible / Item (optionnel)")

st.sidebar.markdown("---")
st.sidebar.header("2. Paramètres Extraction")
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

def reset_feedback():
    st.session_state['feedback_weights'] = {}
    st.success("Feedback reset !")

# --- MAIN ---
if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        tgt = target_col if target_col.strip() != "" else None
        
        processor = TransactionDf(dataframe=raw_df, target_column=tgt, separator=sep_option, formatting=format_option)
        df_encoded = processor.get_df()

        if df_encoded is None or df_encoded.empty:
            st.error("Erreur : Impossible de traiter le fichier. Vérifiez le format.")
        else:
            st.subheader("1. Aperçu des Données Traitées")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"Transactions : {df_encoded.shape[0]} | Items : {df_encoded.shape[1]}")
                st.dataframe(df_encoded.head(5), height=200)
            with col2:
                item_counts = df_encoded.sum().sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.barplot(x=item_counts.index, y=item_counts.values, ax=ax, palette="viridis")
                plt.xticks(rotation=45, ha='right')
                ax.set_title("Top 15 Items")
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("2. Extraction (Pool P)")
            
            if st.button("Lancer l'Extraction"):
                with st.spinner("Extraction..."):
                    frequent = fpgrowth(df_encoded.astype(bool), min_support=min_support, use_colnames=True)
                    if frequent.empty:
                        st.warning("Aucun motif trouvé. Baissez le support.")
                    else:
                        rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
                        if rules.empty:
                            st.warning("Aucune règle trouvée. Baissez la confiance.")
                        else:
                            rules['length'] = rules['antecedents'].apply(len) + rules['consequents'].apply(len)
                            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                            rules['rule_id'] = rules.index
                            st.session_state['pool_rules'] = rules
                            st.session_state['feedback_weights'] = {i: 1.0 for i in rules.index}
                            st.success(f"Terminé : {len(rules)} règles trouvées.")

            if st.session_state['pool_rules'] is not None:
                rules_df = st.session_state['pool_rules']
                
                with st.expander(f"Voir le pool ({len(rules_df)} règles)"):
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
                        st.write("### Sélection :")
                        
                        for i, row in sample.iterrows():
                            rid = row['rule_id'] if 'rule_id' in row else row.name
                            cw = st.session_state['feedback_weights'].get(rid, 1.0)

                            box_col = "transparent"
                            icon = "⚪"
                            if cw > 1.0: 
                                box_col = "#e6ffe6"
                                icon = "✅"
                            elif cw < 1.0: 
                                box_col = "#ffe6e6"
                                icon = "❌"

                            with st.container():
                                cs = st.columns([0.7, 0.15, 0.15])
                                cs[0].markdown(f"**{icon} {row['antecedents_str']} $\\rightarrow$ {row['consequents_str']}**")
                                cs[0].caption(f"Score: {row['composite_score']:.2f} | Poids: {cw:.2f}")

                                if cw > 1.0:
                                    cs[1].button("👍", key=f"lk_d_{i}", disabled=True)
                                    if cs[2].button("Rst", key=f"rs_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.0
                                        st.rerun()
                                elif cw < 1.0:
                                    if cs[1].button("Rst", key=f"rs_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.0
                                        st.rerun()
                                    cs[2].button("👎", key=f"dl_d_{i}", disabled=True)
                                else:
                                    if cs[1].button("👍", key=f"lk_{i}"):
                                        st.session_state['feedback_weights'][rid] = 1.5
                                        st.rerun()
                                    if cs[2].button("👎", key=f"dl_{i}"):
                                        st.session_state['feedback_weights'][rid] = 0.5
                                        st.rerun()
                                st.divider()
                        
                        if st.button("Reset Avis"):
                            reset_feedback()
                            st.rerun()

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