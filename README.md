# ✈️ Klasifikasi Kepuasan Penumpang Maskapai Penerbangan

Model yang digunakan:

* **MLP (Multi-Layer Perceptron)**
* **TabNet (Pretrained)**
* **Embedding + Neural Network**

---

## 📌 Fitur Utama

* 🔍 Prediksi kepuasan penumpang (*Satisfied / Neutral / Dissatisfied*)
* ⚙️ Pemilihan model secara interaktif
* 🧠 Training model tabular dengan berbagai pendekatan
* 📊 Antarmuka web menggunakan **Streamlit**
* 💾 Penyimpanan model & preprocessor

---

## 🗂️ Struktur Folder

```
├── data/
│   └── train.csv
│
├── models/
│   ├── mlp_model.pth
│   ├── tabnet_model.zip
│   ├── embed_nn_model.pth
│   ├── preprocessor.pkl
│   └── encoders.pkl
│
├── app.py                 # Streamlit App
├── train_mlp.py           # Training MLP
├── train_tabnet.py        # Training TabNet
├── train_embedding_nn.py  # Training Embedding + NN
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset

Dataset yang digunakan adalah **Passenger Satisfaction Dataset**, dengan target:

* `satisfaction` → label klasifikasi

Tipe fitur:

* **Numerik**: Age, Flight Distance, dll
* **Kategorikal**: Gender, Class, Type of Travel, dll

---

## 🧠 Model yang Digunakan

### 1️⃣ MLP (Multi-Layer Perceptron)

* One-hot encoding
* StandardScaler
* Arsitektur:

  ```
  Input → 128 → 64 → Output
  ```
* Loss: CrossEntropyLoss
* Optimizer: Adam

### 2️⃣ TabNet

* Native tabular deep learning
* Handling fitur numerik & kategorikal
* Konfigurasi utama:

  * `n_d = 16`
  * `n_steps = 5`
  * `gamma = 1.5`
* Optimizer: Adam

### 3️⃣ Embedding + Neural Network

* Label Encoding untuk fitur kategorikal
* Embedding layer untuk tiap fitur kategorikal
* Digabung dengan fitur numerik
* Cocok untuk data tabular campuran

---

## 🚀 Cara Menjalankan Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Contoh `requirements.txt`:

```txt
streamlit
torch
pandas
scikit-learn
joblib
pytorch-tabnet
```

---

### 2️⃣ Training Model (Opsional)

```bash
python train_mlp.py
python train_tabnet.py
python train_embedding_nn.py
```

Model akan tersimpan di folder `models/`.

---

### 3️⃣ Jalankan Streamlit App

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser:

```
http://localhost:8501
```

---

## 🖥️ Tampilan Aplikasi

Fitur UI:

* Sidebar pemilihan model
* Input data penumpang
* Tombol prediksi
* Hasil prediksi ditampilkan secara visual
