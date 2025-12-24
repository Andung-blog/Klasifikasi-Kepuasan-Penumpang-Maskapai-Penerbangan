import streamlit as st

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Kepuasan Penumpang Maskapai",
    page_icon="✈️",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center;'>✈️ Klasifikasi Kepuasan Penumpang</h1>
    <p style='text-align: center; color: gray;'>
        Sistem Prediksi Kepuasan Penumpang Maskapai Penerbangan
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan Model")

model = st.sidebar.selectbox(
    "Pilih Model Machine Learning",
    ["MLP (Base)", "TabNet (Pretrained)", "Embedding + NN (Pretrained)"]
)

# =========================
# INPUT USER
# =========================
st.subheader("🧾 Input Data Penumpang")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Umur Penumpang", 0, 100, 30)

with col2:
    flight_distance = st.number_input("🛫 Jarak Penerbangan (km)", 0, 10000, 1000)

st.divider()

# =========================
# SESSION STATE INIT
# =========================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.model_used = None

# =========================
# BUTTON
# =========================
if st.button("🔍 Prediksi Kepuasan", use_container_width=True):
    # dummy hasil (nanti ganti model)
    st.session_state.prediction = "Satisfied"
    st.session_state.model_used = model

if st.session_state.prediction is not None:
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: #e8f5e9;
            border-left: 6px solid #2e7d32;
            color: #1b5e20;
        ">
            <h3 style="color: #1b5e20;">✅ Hasil Prediksi</h3>
            <p><b>Model:</b> {st.session_state.model_used}</p>
            <p><b>Status Kepuasan:</b> 
               <span style="font-size: 18px; font-weight: bold; color: #0d47a1;">
                   {st.session_state.prediction}
               </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
