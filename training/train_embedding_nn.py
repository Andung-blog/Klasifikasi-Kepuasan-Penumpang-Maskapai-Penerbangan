import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/train.csv")
df = df.drop(["Unnamed: 0", "id"], axis=1)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median())

cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
num_cols = [c for c in df.columns if c not in cat_cols and c != 'satisfaction']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
joblib.dump(encoders, "models/embed_encoders.pkl")

X_cat = df[cat_cols].values
X_num = df[num_cols].values
y = LabelEncoder().fit_transform(df["satisfaction"])

cat_sizes = [df[col].nunique() for col in cat_cols]
joblib.dump(cat_sizes, "models/embed_cat_sizes.pkl")

class EmbedNN(nn.Module):
    def __init__(self, cat_sizes, num_features, output_dim):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(size, min(50, size//2 + 1)) for size in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embeds)
        self.fc = nn.Sequential(nn.Linear(emb_dim + num_features, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x_cat, x_num):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = torch.cat([x, x_num], 1)
        return self.fc(x)

model = EmbedNN(cat_sizes, X_num.shape[1], 2)
# ... (proses training sama seperti sebelumnya, gunakan 50 epoch)
torch.save(model.state_dict(), "models/embed_nn_model.pth")