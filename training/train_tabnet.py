import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier

df = pd.read_csv("data/train.csv")
df = df.drop(["Unnamed: 0", "id"], axis=1)

X = df.drop("satisfaction", axis=1)
y = LabelEncoder().fit_transform(df["satisfaction"])

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Handle Missing Values
X[cat_cols] = X[cat_cols].fillna("Unknown")
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Save Encoders
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
joblib.dump(encoders, "models/tabnet_encoders.pkl")

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, stratify=y, random_state=42)

model = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2), mask_type="entmax")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=20, patience=5, batch_size=1024, virtual_batch_size=128)

model.save_model("models/tabnet_model")
print("✅ TabNet selesai")