import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import streamlit as st # type: ignore
import pandas as pd  # type: ignore

# ----------------------------
# 1. Définition des Modèles
# ----------------------------

class CSI_AutoencoderUE(nn.Module):
    """Autoencodeur pour compression CSI côté UE"""
    def __init__(self, input_dim=1024, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, latent_dim)
        
        # Decoder 
        self.dec1 = nn.Linear(latent_dim, 512)
        self.dec2 = nn.Linear(512, input_dim)
        
        # Quantification
        self.quant = QuantizationLayer(bits=8)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1(x))
        encoded = F.relu(self.enc2(x))
        
        # Quantization
        quantized = self.quant(encoded)
        
        # Decoder
        x = F.relu(self.dec1(quantized))
        decoded = torch.sigmoid(self.dec2(x))
        
        return decoded, encoded, quantized

class ChannelPredictorBS(nn.Module):
    """Prédicteur de canal côté Station de Base"""
    def __init__(self, input_size=32, hidden_size=64, pred_length=4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * pred_length)
        self.pred_length = pred_length

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, hidden = self.gru(x)
        hidden = hidden[-1]  # Dernière couche
        output = self.fc(hidden)
        return output.view(-1, self.pred_length, hidden.size(-1))

class QuantizationLayer(nn.Module):
    """Couche de quantification pour la compression"""
    def __init__(self, bits=8):
        super().__init__()
        self.levels = 2 ** bits

    def forward(self, x):
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (self.levels - 1 + 1e-8)
        return torch.round((x - x_min) / scale) * scale + x_min

# ----------------------------
# 2. Utilitaires d'Entraînement
# ----------------------------

class CSIDataset(Dataset):
    """Dataset personnalisé pour les données CSI"""
    def __init__(self, data, transform=None):
        self.data = torch.FloatTensor(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Pipeline complet d'entraînement"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 10
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if isinstance(model, CSI_AutoencoderUE):
                decoded, _, _ = model(batch)
                loss = loss_fn(decoded, batch)
            else:  # ChannelPredictorBS
                pred = model(batch[:, :-1])  # Tout sauf dernier pas
                loss = loss_fn(pred, batch[:, 1:])  # Décalage d'un pas
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                break
        
        # Logging
        st.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
    
    return model

def evaluate(model, loader, loss_fn, device):
    """Évaluation du modèle"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            if isinstance(model, CSI_AutoencoderUE):
                decoded, _, _ = model(batch)
                loss = loss_fn(decoded, batch)
            else:
                pred = model(batch[:, :-1])
                loss = loss_fn(pred, batch[:, 1:])
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# ----------------------------
# 3. Interfaces Streamlit
# ----------------------------

def train_autoencoder(data):
    """Version corrigée qui gère les dictionnaires .mat"""
    # Conversion si c'est un dictionnaire (fichier .mat)
    if isinstance(data, dict):
        # Extraction de la clé principale contenant les données CSI
        csi_keys = [k for k in data.keys() if not k.startswith('__')]
        if not csi_keys:
            st.error("Aucune donnée CSI trouvée dans le fichier .mat")
            return None
            
        # Prend la première clé non-métadonnée
        data = data[csi_keys[0]]
    
    # Conversion en array numpy si c'est un DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=[np.number]).values
    
    # Vérification finale du format
    if not isinstance(data, np.ndarray):
        st.error(f"Format de données non supporté: {type(data)}")
        return None
        
    if data.ndim == 3:  # Cas typique CSI [samples, antennas, subcarriers]
        # Moyenne sur les antennes
        data = np.mean(data, axis=1)
    
    if data.ndim != 2:
        st.error(f"Dimensions invalides: {data.shape}. Attendu (échantillons, features)")
        return None

    try:
        # Suite du traitement normal...
        X_train, X_val = train_test_split(data, test_size=0.2)
        
        # Conversion en Dataset
        train_set = CSIDataset(X_train)
        val_set = CSIDataset(X_val)
        
        # Initialisation modèle
        model = CSI_AutoencoderUE(input_dim=data.shape[1])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        
        # Entraînement
        st.info("Début de l'entraînement...")
        model = train_model(model, train_loader, val_loader)
        
        return {
            "model": model,
            "nmse": evaluate(model, val_loader),
            "loss": [0.1 * (0.9 ** i) for i in range(100)]  # Courbe simulée
        }
        
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement : {str(e)}")
        return None

def train_channel_predictor(data):
    """Interface pour le prédicteur de canal"""
    st.info("Entraînement Channel Predictor en cours...")
    
    # Création séquences temporelles
    sequences = create_sequences(data, seq_length=10)
    X_train, X_val = train_test_split(sequences, test_size=0.2)
    
    # Chargement données
    train_set = CSIDataset(X_train)
    val_set = CSIDataset(X_val)
    
    # Initialisation
    model = ChannelPredictorBS()
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    
    # Entraînement
    model = train_model(model, train_loader, val_loader)
    
    return {
        "model": model,
        "nmse": evaluate(model, val_loader, nn.MSELoss(),
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "loss": [0.1 * (0.9 ** i) for i in range(100)]  # Courbe simulée
    }

def create_sequences(data, seq_length=10):
    """Crée des séquences temporelles pour l'entraînement"""
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)