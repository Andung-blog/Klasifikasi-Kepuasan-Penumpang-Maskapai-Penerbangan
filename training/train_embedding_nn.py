import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/train.csv")

cat_cols = df.select_dtypes(include="object").columns.drop("satisfaction")
num_cols = df.select_dtypes(exclude="object").columns

encoders = {}
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

joblib.dump(encoders, "models/encoders.pkl")

X_cat = df[cat_cols].values
X_num = df[num_cols].values
y = LabelEncoder().fit_transform(df["satisfaction"])

Xc_tr, _, Xn_tr, _, y_tr, _ = train_test_split(
    X_cat, X_num, y, test_size=0.2, stratify=y, random_state=42
)

class EmbedNN(nn.Module):
    def __init__(self, cat_sizes, num_features, output_dim):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(size, min(50, size//2 + 1)) for size in cat_sizes
        ])
        emb_dim = sum(e.embedding_dim for e in self.embeds)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x_cat, x_num):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = torch.cat([x, x_num], 1)
        return self.fc(x)

cat_sizes = [df[col].nunique() for col in cat_cols]
model = EmbedNN(cat_sizes, X_num.shape[1], len(set(y)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

Xc_tr = torch.tensor(Xc_tr, dtype=torch.long)
Xn_tr = torch.tensor(Xn_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr)

for _ in range(20):
    optimizer.zero_grad()
    loss = loss_fn(model(Xc_tr, Xn_tr), y_tr)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/embed_nn_model.pth")
print("Embedding NN selesai")
