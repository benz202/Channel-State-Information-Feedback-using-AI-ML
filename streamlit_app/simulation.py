import streamlit as st  # type: ignore
import torch  # type: ignore
from utils.models import CSI_AutoencoderUE, ChannelPredictorBS
from utils.data_loader import generate_real_csi
from utils.visualization import plot_simulation_results

def run_simulation():
    """Fonction principale de simulation"""
    # 1. Génération des données
    csi_data = generate_real_csi()
    
    # 2. Initialisation   des modèles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ue_model = CSI_AutoencoderUE().to(device).eval()
    bs_model = ChannelPredictorBS().to(device).eval()
    
    # 3. Simulation complète
    with torch.no_grad():
        # Encodage UE
        _, _, quantized = ue_model(torch.tensor(csi_data).float().to(device))
        
        # Décodage BS
        reconstructed = bs_model(quantized)
    
    # 4. Visualisation
    results = {
        "real": csi_data,
        "quantized": quantized.cpu().numpy(),
        "reconstructed": reconstructed.cpu().numpy()
    }
    plot_simulation_results(results)
    
    st.success("Simulation terminée avec succès !")