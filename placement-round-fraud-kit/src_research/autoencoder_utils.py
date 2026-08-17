"""
Reusable PyTorch autoencoder for the fraud-detection feature set.

Architecture: input -> Dense(16) -> Dense(8) -> bottleneck(4) -> Dense(8) ->
Dense(16) -> output, trained on the RobustScaler-scaled numeric+engineered
feature matrix (features_v2.csv, ID columns excluded). Built here as an
importable class/function pair (not inline notebook code) specifically so
the next modeling phase can load `artifacts_research/autoencoder.pt` and
reuse this exact architecture as "Model 9" (reconstruction-error-based
anomaly score) without retraining logic living in two places.
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, bottleneck_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def train_autoencoder(X_train, X_val, bottleneck_dim=4, epochs=200, lr=1e-3,
                       batch_size=64, random_state=42, weight_decay=1e-5, verbose=True):
    """X_train / X_val: 2D numpy arrays, already scaled. Returns (model, history)."""
    torch.manual_seed(random_state)
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, bottleneck_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    n = Xtr.shape[0]

    history = {"train_mse": [], "val_mse": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx]
            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.shape[0]
        train_mse = running_loss / n

        model.eval()
        with torch.no_grad():
            out_va, _ = model(Xva)
            val_mse = loss_fn(out_va, Xva).item()

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:>4}/{epochs}  train_mse={train_mse:.5f}  val_mse={val_mse:.5f}")

    return model, history


def reconstruction_errors(model, X):
    """Returns (per-row MSE, per-row MAE, bottleneck codes, reconstructions) as numpy arrays."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        out, z = model(Xt)
        mse = ((out - Xt) ** 2).mean(dim=1).numpy()
        mae = (out - Xt).abs().mean(dim=1).numpy()
    return mse, mae, z.numpy(), out.numpy()


def load_autoencoder(path, input_dim, bottleneck_dim=4):
    model = Autoencoder(input_dim, bottleneck_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
