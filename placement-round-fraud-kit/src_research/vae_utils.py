"""
Reusable PyTorch Variational Autoencoder for the fraud-detection feature set.

Mirrors the structure of autoencoder_utils.py (Model 9's plain autoencoder) so
Model 10 (VAE, 08_models_deep.py) and the Phase 9 hyperparameter search
(09_hyperparameter_optimization.py) share one implementation instead of two
copies that could drift apart.

Architecture: input -> Dense(hidden1) -> Dense(hidden2) -> [mu, logvar]
(both latent_dim) -> reparameterize -> Dense(hidden2) -> Dense(hidden1) ->
output. Loss = reconstruction MSE + beta * KL(q(z|x) || N(0, I)).
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden1=16, hidden2=8, latent_dim=4):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden1)
        self.enc2 = nn.Linear(hidden1, hidden2)
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden2)
        self.dec2 = nn.Linear(hidden2, hidden1)
        self.dec_out = nn.Linear(hidden1, input_dim)
        self.act = nn.ReLU()

    def encode(self, x):
        h = self.act(self.enc1(x))
        h = self.act(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.act(self.dec1(z))
        h = self.act(self.dec2(h))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar, z


def train_vae(X_train, X_val, hidden1=16, hidden2=8, latent_dim=4, epochs=200, lr=1e-3,
              batch_size=64, beta=0.1, random_state=42, verbose=True):
    torch.manual_seed(random_state)
    input_dim = X_train.shape[1]
    model = VAE(input_dim, hidden1=hidden1, hidden2=hidden2, latent_dim=latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    n = Xtr.shape[0]

    history = {"train_recon_mse": [], "train_kl": [], "val_recon_mse": [], "val_kl": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running_recon, running_kl = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx]
            opt.zero_grad()
            out, mu, logvar, _ = model(xb)
            recon_loss = nn.functional.mse_loss(out, xb)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            loss.backward()
            opt.step()
            running_recon += recon_loss.item() * xb.shape[0]
            running_kl += kl_loss.item() * xb.shape[0]
        train_recon = running_recon / n
        train_kl = running_kl / n

        model.eval()
        with torch.no_grad():
            out_va, mu_va, logvar_va, _ = model(Xva)
            val_recon = nn.functional.mse_loss(out_va, Xva).item()
            val_kl = (-0.5 * torch.mean(1 + logvar_va - mu_va.pow(2) - logvar_va.exp())).item()

        history["train_recon_mse"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_recon_mse"].append(val_recon)
        history["val_kl"].append(val_kl)
        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:>4}/{epochs}  train_recon={train_recon:.5f}  "
                  f"train_kl={train_kl:.5f}  val_recon={val_recon:.5f}  val_kl={val_kl:.5f}")

    return model, history


def vae_reconstruction_errors(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        out, mu, logvar, z = model(Xt)
        mse = ((out - Xt) ** 2).mean(dim=1).numpy()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).numpy()
    return mse, kl
