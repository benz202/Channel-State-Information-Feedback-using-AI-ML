import os
import numpy as np  # type: ignore
import pandas as pd # type: ignore
from scipy.io import loadmat # type: ignore
import zipfile
import tempfile
import streamlit as st # type: ignore

def load_mat_file(uploaded_file):
    """Charge un fichier .mat en gérant mieux les structures complexes"""
    try:
        mat_data = loadmat(uploaded_file)
        
        # Nettoyage des clés MATLAB
        mat_data = {k:v for k,v in mat_data.items() 
                   if not k.startswith('__') and not isinstance(v, (str, bytes))}
        
        # Cas particulier des données CSI
        if 'H' in mat_data:  # Format commun pour les matrices de canal
            csi_data = mat_data['H']
            if csi_data.ndim == 3:
                # Conversion en 2D [samples, antennas*subcarriers]
                return csi_data.reshape(csi_data.shape[0], -1)
        
        return mat_data
        
    except Exception as e:
        st.error(f"Erreur de chargement .mat: {str(e)}")
        return None

def load_csi_from_upload(uploaded_file):
    """Charge les données CSI depuis un fichier uploadé"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.mat'):
        return loadmat(uploaded_file)
    else:
        raise ValueError("Format de fichier non supporté")

def load_drone_dataset(session=None):
    """
    Charge le dataset drone depuis Snowflake ou en local
    Args:
        session: Session Snowpark (si None, utilise le mode local)
    """
    if session:  # Mode Snowflake
        return _load_from_snowflake(session)
    else:  # Mode local
        return _load_local_sample()

def _load_from_snowflake(session):
    """Charge les données depuis un stage Snowflake"""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Télécharge depuis le stage
            zip_path = os.path.join(tmp_dir, "O1_drone_200.zip")
            session.file.get("@CSI/O1_drone_200.zip", tmp_dir)
            
            # Extraction et traitement
            return _process_zip_file(zip_path)
            
    except Exception as e:
        st.error(f"Erreur de chargement Snowflake : {str(e)}")
        return None

def _load_local_sample():
    """Charge un échantillon local pour le développement"""
    sample_data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'csi_amplitude': np.random.rand(100, 64),
        'csi_phase': np.random.uniform(-np.pi, np.pi, (100, 64))
    }
    return pd.DataFrame(sample_data)

def _process_zip_file(zip_path):
    """Traite le fichier zip téléchargé"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as extract_dir:
            zip_ref.extractall(extract_dir)
            
            # Charge les fichiers .mat
            cir_path = os.path.join(extract_dir, "O1_drone_200", "O1_drone_200.1.CIR.mat")
            cir_data = loadmat(cir_path)
            
            # Conversion en DataFrame
            csi_key = [k for k in cir_data if not k.startswith('__')][0]
            csi_matrix = cir_data[csi_key]
            
            return pd.DataFrame({
                'amplitude': np.abs(csi_matrix).mean(axis=1),
                'phase': np.angle(csi_matrix).mean(axis=1)
            })

def preprocess_csi_data(df):
    """Prétraitement standard des données CSI"""
    if isinstance(df, pd.DataFrame):
        # Normalisation
        df['amplitude_norm'] = (df['amplitude'] - df['amplitude'].mean()) / df['amplitude'].std()
        df['phase_norm'] = df['phase'] / np.pi
        return df
    else:
        raise ValueError("Input doit être un DataFrame pandas")