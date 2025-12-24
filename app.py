import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# Definisi Arsitektur Model (Wajib ada untuk load state_dict)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

class EmbedNN(nn.Module):
    def __init__(self, cat_sizes, num_features, output_dim):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(size, min(50, size//2 + 1)) for size in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embeds)
        self.fc = nn.Sequential(nn.Linear(emb_dim + num_features, 128), nn.ReLU(), nn.Linear(128, output_dim))
    def forward(self, x_cat, x_num):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]; x = torch.cat(x, 1)
        x = torch.cat([x, x_num], 1); return self.fc(x)

# =========================
# CONFIG & LOAD
# =========================
st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️")

@st.cache_resource
def load_models():
    # Load MLP
    mlp_cols = joblib.load("models/mlp_columns.pkl")
    mlp_scaler = joblib.load("models/preprocessor.pkl")
    mlp_model = MLP(len(mlp_cols), 2)
    mlp_model.load_state_dict(torch.load("models/mlp_model.pth"))
    mlp_model.eval()

    # Load TabNet
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model("models/tabnet_model.zip")
    tabnet_encoders = joblib.load("models/tabnet_encoders.pkl")

    # Load Embedding NN
    embed_encoders = joblib.load("models/embed_encoders.pkl")
    cat_sizes = joblib.load("models/embed_cat_sizes.pkl")
    embed_model = EmbedNN(cat_sizes, 18, 2) # 18 num features
    embed_model.load_state_dict(torch.load("models/embed_nn_model.pth"))
    embed_model.eval()

    return (mlp_model, mlp_scaler, mlp_cols, 
            tabnet_model, tabnet_encoders, 
            embed_model, embed_encoders)

# Load artifacts
try:
    m_mlp, s_mlp, c_mlp, m_tab, e_tab, m_emb, e_emb = load_models()
except:
    st.error("Model belum dilatih atau file models/ tidak ditemukan!")
    st.stop()

# =========================
# UI INPUT
# =========================
st.title("✈️ Airline Passenger Satisfaction")

with st.expander("📝 Form Data Penumpang", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        cust_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("Age", 0, 100, 30)
    with col2:
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        distance = st.number_input("Flight Distance", 0, 10000, 1000)
    with col3:
        dep_delay = st.number_input("Departure Delay (m)", 0, 2000, 0)
        arr_delay = st.number_input("Arrival Delay (m)", 0, 2000, 0)

    st.write("---")
    st.write("Berikan Rating (0-5):")
    r_cols = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", 
              "Gate location", "Food and drink", "Online boarding", "Seat comfort", 
              "Inflight entertainment", "On-board service", "Leg room service", 
              "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]
    
    ratings = {}
    cols = st.columns(4)
    for i, column_name in enumerate(r_cols):
        ratings[column_name] = cols[i % 4].slider(column_name, 0, 5, 3)

# Persiapkan Dictionary Data
user_data = {
    "Gender": gender, "Customer Type": cust_type, "Age": age, 
    "Type of Travel": travel_type, "Class": flight_class, "Flight Distance": distance,
    "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay,
    **ratings
}
df_input = pd.DataFrame([user_data])

# =========================
# PREDICTION LOGIC
# =========================
model_choice = st.sidebar.selectbox("Pilih Model", ["MLP", "TabNet", "Embedding + NN"])

if st.button("Prediksi Kepuasan", use_container_width=True):
    result = ""
    
    if model_choice == "MLP":
        X = pd.get_dummies(df_input)
        X = X.reindex(columns=c_mlp, fill_value=0) # Samakan kolom
        X_scaled = s_mlp.transform(X)
        with torch.no_grad():
            output = m_mlp(torch.tensor(X_scaled, dtype=torch.float32))
            pred = torch.argmax(output, dim=1).item()
            
    elif model_choice == "TabNet":
        X_tab = df_input.copy()
        for col, le in e_tab.items():
            X_tab[col] = le.transform(X_tab[col])
        pred = m_tab.predict(X_tab.values)[0]
        
    elif model_choice == "Embedding + NN":
        X_cat = df_input[['Gender', 'Customer Type', 'Type of Travel', 'Class']].copy()
        for col, le in e_emb.items():
            X_cat[col] = le.transform(X_cat[col])
        X_num = df_input.drop(['Gender', 'Customer Type', 'Type of Travel', 'Class'], axis=1)
        with torch.no_grad():
            output = m_emb(torch.tensor(X_cat.values, dtype=torch.long), 
                           torch.tensor(X_num.values, dtype=torch.float32))
            pred = torch.argmax(output, dim=1).item()

    # Tampilkan Hasil
    status = "Satisfied" if pred == 1 else "Neutral or Dissatisfied"
    color = "#1b5e20" if pred == 1 else "#b71c1c"
    bg = "#e8f5e9" if pred == 1 else "#ffebee"
    
    st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background-color:{bg}; border-left:6px solid {color};">
            <h3 style="color:{color}; margin:0;">Hasil: {status}</h3>
            <p style="color:gray; margin:0;">Diprediksi menggunakan {model_choice}</p>
        </div>
    """, unsafe_allow_html=True)