import streamlit as st
import pandas as pd
from utils import TransactionDf
from ui_sections import render_data_section, render_mining_section, render_interactive_section

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="EDA Project - Pattern Mining",
    layout="wide",
    page_icon="⛏️",
    initial_sidebar_state="expanded"
)

st.title("Projet EDA : Fouille Interactive de Motifs")

# GESTION ÉTAT & NETTOYAGE AUTOMATIQUE
# Cette section doit être tout en haut pour intercepter le changement de fichier
if 'last_uploaded_file_key' not in st.session_state:
    st.session_state['last_uploaded_file_key'] = None


def clear_cache():
    """Nettoie toute la mémoire de session liée aux données"""
    keys_to_clear = ['pool_rules', 'feedback_weights', 'last_sample', 'exec_time', 'processor']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# SIDEBAR (PARAMÈTRES)
with st.sidebar:
    st.header("1. Configuration")

    # Widget d'upload
    uploaded_file = st.file_uploader("Charger un CSV", type=["csv"])

    # DÉTECTION DE CHANGEMENT DE FICHIER (par clé stable)
    current_file_key = uploaded_file.name if uploaded_file is not None else None
    if current_file_key != st.session_state['last_uploaded_file_key']:
        clear_cache()
        st.session_state['last_uploaded_file_key'] = current_file_key
        st.experimental_set_query_params(file=current_file_key or "")
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

# INIT VARIABLES SESSION (si elles n'existent pas après le clear)
if 'pool_rules' not in st.session_state: st.session_state['pool_rules'] = None
if 'feedback_weights' not in st.session_state: st.session_state['feedback_weights'] = {}
if 'last_sample' not in st.session_state: st.session_state['last_sample'] = None
if 'exec_time' not in st.session_state: st.session_state['exec_time'] = 0.0
if 'processor' not in st.session_state: st.session_state['processor'] = None

# MAIN LOGIC
if uploaded_file:
    try:
        # Chargement et Preprocessing
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
            # STRUCTURE EN ONGLETS
            tab_data, tab_mining, tab_interactive = st.tabs([
                "📊 Données", "⚙️ Extraction & Pool", "🎮 Échantillonnage Interactif"
            ])

            # ==========================
            # TAB 1: DONNÉES
            # ==========================
            with tab_data:
                render_data_section(df_encoded)

            # ==========================
            # TAB 2: EXTRACTION
            # ==========================
            with tab_mining:
                render_mining_section(
                    df_encoded=df_encoded,
                    algo_choice=algo_choice,
                    min_support=min_support if 'min_support' in locals() else None,
                    min_confidence=min_confidence if 'min_confidence' in locals() else None,
                    sampling_iterations=sampling_iterations if 'sampling_iterations' in locals() else None,
                    sampling_min_support=sampling_min_support if 'sampling_min_support' in locals() else None,
                    sampling_max_rules=sampling_max_rules if 'sampling_max_rules' in locals() else None,
                    random_seed=int(random_seed),
                )

            # ==========================
            # TAB 3: INTERACTIF
            # ==========================
            with tab_interactive:
                render_interactive_section(
                    processor=st.session_state['processor'],
                    w_support=w_support,
                    w_lift=w_lift,
                    w_conf=w_conf,
                    w_surprise=w_surprise,
                    k_samples=int(k_samples),
                    replace_strategy=bool(replace_strategy),
                    random_seed=int(random_seed),
                )

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")