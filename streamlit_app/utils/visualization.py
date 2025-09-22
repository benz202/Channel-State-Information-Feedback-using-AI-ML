import matplotlib.pyplot as plt # type: ignore
import streamlit as st # type: ignore

def plot_comparison(ae_nmse, pred_nmse):
    """Affiche la comparaison des deux modèles"""
    fig, ax = plt.subplots()
    models = ['Autoencoder', 'Channel Predictor']
    values = [ae_nmse, pred_nmse]
    ax.bar(models, values)
    ax.set_ylabel('NMSE')
    st.pyplot(fig)

def plot_simulation_results(results):
    """Visualise les résultats de simulation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CSI réel
    axes[0].imshow(np.abs(results["real_csi"][0].reshape(8,8)), cmap='hot')
    axes[0].set_title("CSI Réel")
    
    # Feedback quantifié
    axes[1].plot(results["quantized"][0].cpu().numpy())
    axes[1].set_title("Feedback UE")
    
    # CSI reconstruit
    rec_magnitude = results["reconstructed"][0,:,0].cpu().numpy()
    axes[2].imshow(rec_magnitude.reshape(8,8), cmap='hot')
    axes[2].set_title(f"Reconstruction BS\nNMSE: {results['nmse']:.4f}")
    
    st.pyplot(fig)