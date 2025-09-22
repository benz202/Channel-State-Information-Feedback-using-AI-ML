# from wordcloud import WordCloud # type: ignore 
# from dash import Dash, Input, Output, State, dcc, html # type: ignore
# from collections import defaultdict
# import base64
# from io import BytesIO 

# # Initialisation de l'application Dash
# app = Dash(__name__)

# # Mise en page
# app.layout = html.Div([
#     html.H1("Système de recherche textuelle"),
#     dcc.Input(id='search-input', type='text', placeholder='Entrez un mot ou une expression', style={'width': '60%'}),
#     html.Button('Rechercher', id='search-button'),
#     html.Div(id='results-output', style={'margin-top': '20px'})
# ])

# # Fonction pour générer un Word Cloud
# def generate_wordcloud(context_words):
#     if not context_words:
#         return ""

#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(context_words))
#     buffer = BytesIO()
#     wordcloud.to_image().save(buffer, format='PNG')
#     buffer.seek(0)
#     encoded_image = base64.b64encode(buffer.read()).decode()
#     return f"data:image/png;base64,{encoded_image}"

# # Fonction pour extraire des extraits de texte
# def extract_contexts(expression, data, inverted_index):
#     tokens = expression.lower().split()
#     contexts = []
#     context_words = []
#     doc_indices = defaultdict(list)

#     for doc_id, position in inverted_index.get(tokens[0], []):
#         if all((doc_id, position + i) in inverted_index.get(tokens[i], []) for i in range(len(tokens))):
#             token_list = data.iloc[doc_id]['tokens']
#             start = max(0, position - 5)  # 5 mots avant l'expression
#             end = position + len(tokens) + 5  # 5 mots après l'expression
#             contexts.append(' '.join(token_list[start:end]))
#             context_words += token_list[start:position] + token_list[position + len(tokens):end]
#             doc_indices[doc_id].append(position)

#     return contexts, context_words, doc_indices

# # Fonction pour calculer les statistiques
# def calculate_statistics(expression, doc_indices):
#     total_appearances = sum(len(positions) for positions in doc_indices.values())
#     mean_appearances = {doc_id: sum(positions) / len(positions) for doc_id, positions in doc_indices.items()}
#     distribution = {doc_id: positions for doc_id, positions in doc_indices.items()}

#     return total_appearances, mean_appearances, distribution

# # Callback pour traiter la recherche
# @app.callback(
#     Output('results-output', 'children'),
#     Input('search-button', 'n_clicks'),
#     State('search-input', 'value')
# )
# def update_results(n_clicks, search_query):
#     if n_clicks is None or not search_query:
#         return "Entrez un mot ou une expression pour commencer la recherche."

#     # Extraction des extraits et mots de contexte
#     contexts, context_words, doc_indices = extract_contexts(search_query, data, inverted_index) # type: ignore

#     if contexts:
#         # Génération du Word Cloud
#         wordcloud_src = generate_wordcloud(context_words)

#         # Calcul des statistiques
#         total_appearances, mean_appearances, distribution = calculate_statistics(search_query, doc_indices)

#         # Affichage des extraits, Word Cloud et statistiques
#         return html.Div([
#             html.H3("Extraits de texte contenant l'expression :"),
#             html.Ul([html.Li(context) for context in contexts]),
#             html.H3("Word Cloud des mots de contexte :"),
#             html.Img(src=wordcloud_src, style={'width': '80%', 'margin-top': '20px'}),
#             html.H3("Statistiques :"),
#             html.P(f"Nombre total d'apparitions : {total_appearances}"),
#             html.H4("Distribution des indices dans les documents :"),
#             html.Ul([html.Li(f"Document {doc_id} : {positions}") for doc_id, positions in distribution.items()]),
#             html.H4("Moyenne d'apparition dans chaque document :"),
#             html.Ul([html.Li(f"Document {doc_id} : {mean:.2f}") for doc_id, mean in mean_appearances.items()])
#         ])
#     else:
#         return "Aucun résultat trouvé."

# # Lancer l'application
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import streamlit as st
# import pandas as pd
# import numpy as np

# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(page_title="CSI Enhancement | Hackathon", layout="wide")

# # -----------------------------
# # Logo Nokia en haut de page
# # -----------------------------
# st.image("nokia_logo.png", width=180)

# st.markdown("""
# <style>
#     .main {
#         background-color: #f4f8fb;
#     }
#     h1 {
#         color: #0a3c6b;
#     }
#     .block-container {
#         padding-top: 1rem;
#     }
#     .css-1aumxhk {
#         font-family: 'Segoe UI', sans-serif;
#     }
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # Sidebar
# # -----------------------------
# st.sidebar.image("nokia_logo.png", width=100)

# st.sidebar.title("🔍 Menu")
# page = st.sidebar.radio("Navigation", [
#     "🏠 Accueil",
#     "📂 Données",
#     "🔁 Autoencoder",
#     "📈 Channel Predictor",
#     "📊 Comparaison"
# ])


# # -----------------------------
# # Fonctions modèles (à remplacer par mes collègues)
# # -----------------------------
# def train_autoencoder(data):
#     st.info("⏳ Entraînement en cours...")
#     return {"loss": np.random.rand(100).tolist(), "nmse": round(np.random.uniform(0.01, 0.1), 4)}

# def train_channel_predictor(data):
#     st.info("⏳ Entraînement en cours...")
#     return {"loss": np.random.rand(100).tolist(), "nmse": round(np.random.uniform(0.01, 0.1), 4)}

# # -----------------------------
# # 🏠 Accueil
# # -----------------------------
# if page == "🏠 Accueil":
#     st.title("📡 CSI Feedback Enhancement avec Machine Learning")
#     st.subheader("Amélioration de l’efficacité du CSI Feedback dans les réseaux 5G/6G")

#     st.markdown("""
#     Ce projet explore deux approches pour compresser ou prédire les informations CSI :
    
#     - 🔁 **Autoencoder** : encode et reconstruit les informations de canal
#     - 📈 **Channel Predictor** : prédit les futures CSI à partir des historiques

#     🧠 Objectif : réduire la bande passante et améliorer la performance globale des réseaux.
#     """)

# # -----------------------------
# # 📂 Données
# # -----------------------------
# elif page == "📂 Données":
#     st.title("📂 Chargement de données CSI")
#     uploaded_file = st.file_uploader("📄 Importer un fichier CSV contenant les données CSI", type=["csv"])

#     if uploaded_file:
#         data = pd.read_csv(uploaded_file)
#         st.session_state["csi_data"] = data
#         st.success("✅ Données chargées avec succès !")
#         st.write("👀 Aperçu des données :")
#         st.dataframe(data.head())
#         st.caption(f"Dimensions du dataset : {data.shape}")
#     else:
#         st.warning("🕐 En attente de chargement du fichier CSV.")

# # -----------------------------
# # 🔁 Autoencoder
# # -----------------------------
# elif page == "🔁 Autoencoder":
#     st.title("🔁 Modèle Autoencoder")
#     st.markdown("Compression de CSI via un réseau de neurones autoencoder.")

#     if "csi_data" in st.session_state:
#         if st.button("🚀 Lancer l'entraînement"):
#             result = train_autoencoder(st.session_state["csi_data"])
#             st.success(f"📉 NMSE obtenu : **{result['nmse']}**")
#             st.line_chart(result["loss"])
#             st.caption("🔍 Courbe d'apprentissage du modèle")
#     else:
#         st.warning("📂 Veuillez d'abord charger un dataset dans l'onglet Données.")

# # -----------------------------
# # 📈 Channel Predictor
# # -----------------------------
# elif page == "📈 Channel Predictor":
#     st.title("📈 Modèle de Prédiction CSI")
#     st.markdown("Utilisation d'un modèle prédictif pour estimer la prochaine valeur CSI.")

#     if "csi_data" in st.session_state:
#         if st.button("🚀 Lancer la prédiction"):
#             result = train_channel_predictor(st.session_state["csi_data"])
#             st.success(f"📉 NMSE obtenu : **{result['nmse']}**")
#             st.line_chart(result["loss"])
#             st.caption("📉 Évolution de la perte durant l'entraînement")
#     else:
#         st.warning("📂 Veuillez charger un fichier CSI pour lancer le modèle.")

# # -----------------------------
# # 📊 Comparaison
# # -----------------------------
# elif page == "📊 Comparaison":
#     st.title("📊 Comparaison des résultats")
#     st.markdown("Visualisation comparative des deux approches sur la base du NMSE.")

#     # Exemples fictifs (à remplacer par les vrais)
#     ae_nmse = 0.045
#     pred_nmse = 0.032

#     comp_df = pd.DataFrame({
#         "Méthode": ["Autoencoder", "Channel Predictor"],
#         "NMSE": [ae_nmse, pred_nmse]
#     })

#     st.bar_chart(comp_df.set_index("Méthode"))
#     st.caption("📝 Ces résultats sont simulés. Remplacez-les avec les sorties des modèles réels.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os

# # --------------------------------------------
# # 0. Configuration de la page (DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT)
# # --------------------------------------------
# st.set_page_config(page_title="CSI Feedback - Hackathon Nokia", layout="wide")

# try:
#     st.sidebar.image("nokia_logo.png", width=100)
# except Exception as e:
#     st.sidebar.warning("⚠️ Logo non disponible.")
#     st.sidebar.text(f"Erreur : {e}")

# # --------------------------------------------
# # 1. Configuration de l'interface
# # --------------------------------------------


# st.sidebar.title("🔍 Navigation")
# page = st.sidebar.radio("Aller à :", ["🏠 Accueil", "📂 Données", "🔁 Autoencoder", "📈 Channel Predictor", "📊 Comparaison", "📡 Simulation UE-BS"])

# if page == "🏠 Accueil":
#     st.title("📡 CSI Feedback Enhancement with Machine Learning")
#     st.subheader("Hackathon Nokia x Université")
#     st.markdown("""
#     Bienvenue sur l'interface Streamlit du projet !  
#     Cette application permet d'explorer deux approches pour améliorer le feedback CSI :
    
#     - 🔁 **Autoencoder** : compresse et reconstruit les données CSI
#     - 📈 **Channel Predictor** : prédit le CSI futur sans feedback direct
    
#     L'objectif est de minimiser la différence entre les CSI originaux et reconstruits.
#     """)

# elif page == "📂 Données":
#     st.title("📂 Chargement des données CSI")
#     uploaded_file = st.file_uploader("📄 Téléverser un fichier CSV contenant les données CSI", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         st.session_state["csi_data"] = df
#         st.success("✅ Données chargées avec succès !")
#         st.dataframe(df.head())
#         st.caption(f"Dimensions : {df.shape}")

# elif page == "🔁 Autoencoder":
#     st.title("🔁 Compression CSI avec Autoencoder")
#     if "csi_data" in st.session_state:
#         if st.button("🚀 Lancer l'entraînement"):
#             result = train_autoencoder(st.session_state["csi_data"])
#             st.success(f"NMSE : {result['nmse']}")
#             st.line_chart(result["loss"])
#     else:
#         st.warning("Veuillez charger des données d'abord")

# elif page == "📈 Channel Predictor":
#     st.title("📈 Prédiction CSI")
#     if "csi_data" in st.session_state:
#         if st.button("🚀 Lancer la prédiction"):
#             result = train_channel_predictor(st.session_state["csi_data"])
#             st.success(f"NMSE : {result['nmse']}")
#             st.line_chart(result["loss"])
#     else:
#         st.warning("Veuillez charger des données d'abord")

# elif page == "📊 Comparaison":
#     st.title("📊 Comparaison des modèles")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Autoencoder NMSE", st.session_state.get('ae_nmse', 0.045))
#     with col2:
#         st.metric("Predictor NMSE", st.session_state.get('pred_nmse', 0.032))

# elif page == "📡 Simulation UE-BS":
#     try:
#         from simulation import simulation_page
#         simulation_page()
#     except Exception as e:
#         st.error(f"Erreur de chargement : {str(e)}")
#         st.info("Assurez-vous que le fichier simulation.py existe")


# # --------------------------------------------
# # 2. Fonctions de simulation améliorées
# # --------------------------------------------
# def train_autoencoder(data):
#     st.info("⏳ Simulation de l'entraînement Autoencoder...")
#     loss = [1/(x+1) for x in range(100)]  # Courbe de loss plus réaliste
#     nmse = round(0.02 + np.random.uniform(0.01, 0.05), 4)
#     st.session_state.ae_nmse = nmse  # Stocke le résultat pour la comparaison
#     return {"loss": loss, "nmse": nmse}

# def train_channel_predictor(data):
#     st.info("⏳ Simulation de l'entraînement Channel Predictor...")
#     loss = [1/(x+1.5) for x in range(100)]  # Courbe de loss différente
#     nmse = round(0.01 + np.random.uniform(0.01, 0.03), 4)
#     st.session_state.pred_nmse = nmse  # Stocke le résultat pour la comparaison
#     return {"loss": loss, "nmse": nmse}


# import streamlit as st
# import pandas as pd
# import numpy as np
# from utils.models import train_autoencoder, train_channel_predictor
# from utils.visualization import plot_comparison
# from utils.data_loader import load_csi_from_upload

# # Configuration de la page (DOIT RESTER LA PREMIÈRE COMMANDE STREAMLIT)
# st.set_page_config(page_title="CSI Feedback - Hackathon Nokia", layout="wide")

# # --------------------------------------------
# # Barre latérale - Navigation
# # --------------------------------------------
# try:
#     st.sidebar.image("nokia_logo.png", width=100)
# except FileNotFoundError:
#     st.sidebar.warning("Logo non trouvé")

# st.sidebar.title("🔍 Navigation")
# menu_options = [
#     "🏠 Accueil", 
#     "📂 Données", 
#     "🔁 Autoencoder",
#     "📈 Channel Predictor", 
#     "📊 Comparaison",
#     "📡 Simulation UE-BS"
# ]
# page = st.sidebar.radio("Aller à :", menu_options)

# # --------------------------------------------
# # Corps principal de l'application
# # --------------------------------------------

# # Page Accueil
# if page == "🏠 Accueil":
#     st.title("📡 CSI Feedback Enhancement with Machine Learning")
#     st.markdown("""
#     **Bienvenue à l'interface du projet !**  
#     Fonctionnalités disponibles :
#     - Compression CSI avec Autoencoder
#     - Prédiction de canal
#     - Simulation complète UE → Station de Base
#     """)

# # Page Données
# elif page == "📂 Données":
#     st.title("📂 Chargement des données")
#     uploaded_file = st.file_uploader("Téléversez vos données CSI (CSV/MAT)", type=["csv", "mat"])
    
#     if uploaded_file:
#         try:
#             st.session_state.csi_data = load_csi_from_upload(uploaded_file)
#             st.success("Données chargées avec succès !")
#             st.dataframe(st.session_state.csi_data.head())
#         except Exception as e:
#             st.error(f"Erreur : {str(e)}")

# # Page Autoencoder
# elif page == "🔁 Autoencoder":
#     st.title("🔁 Entraînement Autoencoder")
    
#     if "csi_data" not in st.session_state:
#         st.warning("Veuillez d'abord charger des données")
#     else:
#         if st.button("Lancer l'entraînement"):
#             with st.spinner("Entraînement en cours..."):
#                 # Conversion explicite si nécessaire
#                 if isinstance(st.session_state.csi_data, dict):
#                     st.info("Conversion des données .mat en cours...")
                
#                 results = train_autoencoder(st.session_state.csi_data)
                
#                 if results:
#                     st.session_state.ae_results = results
#                     st.success("Entraînement terminé!")
#                     st.line_chart(results["loss"])

# # Page Channel Predictor  
# elif page == "📈 Channel Predictor":
#     st.title("📈 Prédicteur de Canal")
    
#     if "csi_data" not in st.session_state:
#         st.warning("Veuillez d'abord charger des données")
#     else:
#         if st.button("Lancer la prédiction"):
#             with st.spinner("Calcul des prédictions..."):
#                 results = train_channel_predictor(st.session_state.csi_data)
#                 st.session_state.pred_results = results
#                 st.success("Prédiction terminée !")
#                 st.line_chart(results["loss"])

# # Page Comparaison
# elif page == "📊 Comparaison":
#     st.title("📊 Comparaison des Modèles")
    
#     if "ae_results" in st.session_state and "pred_results" in st.session_state:
#         plot_comparison(
#             st.session_state.ae_results["nmse"],
#             st.session_state.pred_results["nmse"]
#         )
#     else:
#         st.warning("Veuillez exécuter les deux modèles d'abord")

# # Page Simulation UE-BS (NOUVELLE INTÉGRATION)
# elif page == "📡 Simulation UE-BS":
#     st.title("📡 Simulation Complète UE → Station de Base")
    
#     # Chargement conditionnel pour éviter les imports circulaires
#     if "simulation" not in st.session_state:
#         from simulation import run_simulation
#         st.session_state.simulation = run_simulation
    
#     # Bouton de lancement
#     if st.button("🚀 Démarrer la Simulation"):
#         with st.spinner("Simulation en cours (cela peut prendre quelques minutes)..."):
#             try:
#                 st.session_state.simulation()
#             except Exception as e:
#                 st.error(f"Erreur lors de la simulation : {str(e)}")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from models import CSI_AutoencoderUE  # Supposons que votre modèle est dans ce fichier

# Titre de l'application
st.title("Simulation CSI UE→BS")
st.markdown("""
**Fonctionnement:**
1. Chargez/selectionnez des données CSI
2. Le modèle UE encode et quantifie les données
3. Le BS tente de reconstruire le signal original
""")

# Section 1: Chargement des données
st.header("1. Données CSI d'entrée")
data_option = st.radio("Source des données:", 
                      ("Exemple prédéfini", "Upload manuel"))

if data_option == "Exemple prédéfini":
    # Charger un exemple prédéfini
    sample_data = np.random.randn(1024, 2) * 0.5  # Données factices
    st.success("Données d'exemple chargées!")
else:
    uploaded_file = st.file_uploader("Uploader un fichier .mat ou .npy", type=["mat", "npy"])
    if uploaded_file:
        if uploaded_file.name.endswith('.mat'):
            from scipy.io import loadmat
            sample_data = loadmat(uploaded_file)['CSI_data']  # Adaptez selon votre structure
        else:
            sample_data = np.load(uploaded_file)
        st.success("Données chargées avec succès!")

# Aperçu des données
if 'sample_data' in locals():
    st.subheader("Aperçu des données")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(sample_data[:, 0], label='Magnitude')
    ax[0].set_title('Magnitude')
    ax[1].plot(sample_data[:, 1], label='Phase', color='orange')
    ax[1].set_title('Phase')
    st.pyplot(fig)

# Section 2: Simulation UE
st.header("2. Encodage UE")
if st.button("Lancer l'encodage UE") and 'sample_data' in locals():
    # Initialisation du modèle UE
    ue_model = CSI_AutoencoderUE().eval()
    if os.path.exists("best_model_ue.pth"):
        ue_model.load_state_dict(torch.load("best_model_ue.pth"))
    
    # Conversion des données en tenseur
    input_tensor = torch.FloatTensor(sample_data[np.newaxis, ...])
    
    # Encodage
    with torch.no_grad():
        _, _, quantized = ue_model(input_tensor)
    
    st.session_state['quantized'] = quantized.numpy()
    st.success(f"Données quantifiées générées! Taille: {st.session_state['quantized'].shape}")

# Section 3: Simulation BS
st.header("3. Décodage BS")
if st.button("Lancer l'estimation BS") and 'quantized' in st.session_state:
    # Ici vous intégrerez votre modèle BS (TimeLlama ou autre)
    # Pour l'exemple, nous allons simplement utiliser le décodeur UE
    
    bs_model = CSI_AutoencoderUE().eval()
    if os.path.exists("best_model_ue.pth"):
        bs_model.load_state_dict(torch.load("best_model_ue.pth"))
    
    # Décodage
    with torch.no_grad():
        reconstructed, _, _ = bs_model.decoder(torch.FloatTensor(st.session_state['quantized']))
    
    st.session_state['reconstructed'] = reconstructed.numpy()
    
    # Calcul de l'erreur
    original = sample_data
    mse = np.mean((original - st.session_state['reconstructed'][0])**2)
    
    st.subheader("Résultats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Erreur MSE", f"{mse:.4f}")
    
    with col2:
        similarity = np.corrcoef(original.flatten(), st.session_state['reconstructed'][0].flatten())[0,1]
        st.metric("Similarité", f"{similarity:.2%}")
    
    # Visualisation
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    # Magnitude
    ax[0,0].plot(original[:, 0], label='Original')
    ax[0,0].set_title('Magnitude (Original)')
    
    ax[0,1].plot(st.session_state['reconstructed'][0, :, 0], label='Reconstruit', color='orange')
    ax[0,1].set_title('Magnitude (Reconstruit)')
    
    # Phase
    ax[1,0].plot(original[:, 1], label='Original')
    ax[1,0].set_title('Phase (Original)')
    
    ax[1,1].plot(st.session_state['reconstructed'][0, :, 1], label='Reconstruit', color='orange')
    ax[1,1].set_title('Phase (Reconstruit)')
    
    plt.tight_layout()
    st.pyplot(fig)

# Section pour les experts (optionnelle)
with st.expander("Paramètres avancés"):
    st.write("""
    **Paramètres du modèle UE:**
    - Taille latente: 32 dimensions
    - Quantification: 8 bits
    """)
    
    if 'reconstructed' in st.session_state:
        st.download_button(
            label="Télécharger les données reconstruites",
            data=np.save("reconstructed.npy", st.session_state['reconstructed']),
            file_name="csi_reconstructed.npy",
            mime="application/octet-stream"
        )