import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/train.csv")

X = df.drop("satisfaction", axis=1)
y = LabelEncoder().fit_transform(df["satisfaction"])

X = pd.get_dummies(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "models/preprocessor.pkl")

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1], len(set(y)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)

for _ in range(20):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/mlp_model.pth")
print("MLP selesai")
