import numpy as np # type: ignore

def generate_real_csi(num_samples=1000, num_antennas=64):
    """Génère des données CSI réalistes avec corrélation temporelle"""
    time_corr = 0.95
    csi = np.random.randn(num_samples, num_antennas) + 1j*np.random.randn(num_samples, num_antennas)
    
    for t in range(1, num_samples):
        csi[t] = time_corr * csi[t-1] + np.sqrt(1-time_corr**2) * csi[t]
    
    return csi

def preprocess_csi(csi_data):
    """Prétraitement des données CSI pour les modèles"""
    magnitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    
    # Normalisation
    norm_mag = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
    norm_phase = phase / np.pi
    
    return np.stack([norm_mag, norm_phase], axis=-1)