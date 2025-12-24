import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier

# =========================
# Load Data
# =========================
df = pd.read_csv("data/train.csv")

# =========================
# Pisahkan fitur & target
# =========================
X = df.drop("satisfaction", axis=1)
y = LabelEncoder().fit_transform(df["satisfaction"])

# =========================
# Handle Missing Values
# =========================
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Kategorikal → isi "Unknown"
X[cat_cols] = X[cat_cols].fillna("Unknown")

# Numerik → isi median
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# =========================
# Label Encoding (WAJIB untuk TabNet)
# =========================
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# =========================
# Final Check (WAJIB)
# =========================
assert not X.isna().any().any(), "Masih ada NaN di data!"

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Train TabNet
# =========================
model = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="entmax"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=50,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# =========================
# Save Model
# =========================
model.save_model("models/tabnet_model")

print("✅ TabNet selesai tanpa error")
