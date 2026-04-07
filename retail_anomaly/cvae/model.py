"""
retail_anomaly.cvae.model
==========================
Conditional Variational Autoencoder for synthetic retail data generation.

Architecture
------------
* Encoder  q_φ(z | x, city)  →  μ_φ(x,y),  logσ²_φ(x,y)
* Decoder  p_θ(x | z, city)  →  x̂

Design choices
--------------
* Decoder first layer initialised from LFM loading matrix Λ  (warm start,
  speeds convergence, reduces posterior collapse risk on small datasets)
* KL annealing  (β: 0 → 1 over `beta_warmup_epochs`)
* Free bits     (per-dim KL floor = `free_bits` nats)
* City label as one-hot vector fed into both encoder and decoder

Reference: Slide 41-44 (Conditional VAE) + Slide 45 (LFM connection)
in the STAT 32100 VAE lecture notes.
"""
from __future__ import annotations

import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    warnings.warn("PyTorch not installed; RetailCVAE unavailable. "
                  "Install with: pip install torch")

import numpy as np
import pandas as pd
from retail_anomaly.utils.config import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Stubs when torch is absent
# ──────────────────────────────────────────────────────────────────────────────

if not _TORCH_OK:
    import types
    nn = types.SimpleNamespace(
        Module=object,
        Sequential=lambda *a: None,
        Linear=lambda *a, **k: None,
        SiLU=lambda: None,
    )
    F  = types.SimpleNamespace(
        mse_loss=lambda *a, **k: None,
        one_hot=lambda *a, **k: None,
    )
    class torch:  # type: ignore
        Tensor = None
        class no_grad:
        def __enter__(self): pass
        def __exit__(self, *a): pass
        def __call__(self, fn): return fn


# ──────────────────────────────────────────────────────────────────────────────
# Network blocks  (functional only when torch is available)
# ──────────────────────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),        nn.SiLU(),
        )
        self.mu_layer     = nn.Linear(hidden, z_dim)
        self.logvar_layer = nn.Linear(hidden, z_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, y], dim=-1))
        return self.mu_layer(h), self.logvar_layer(h)


class _Decoder(nn.Module):
    def __init__(self, z_dim: int, y_dim: int, x_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),        nn.SiLU(),
            nn.Linear(hidden, x_dim),
            nn.Sigmoid(),   # KDE scores ∈ [0, 1] after /100 normalisation
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, y], dim=-1))

    def init_from_loadings(self, loadings: np.ndarray) -> None:
        """Warm-start first Linear weight from LFM loading matrix Λ."""
        with torch.no_grad():
            L = torch.tensor(loadings.T, dtype=torch.float32)   # (z_dim, x_dim)
            first_linear: nn.Linear = self.net[0]
            # Only copy the z-slice of the first layer (y part stays random)
            z_dim = L.shape[0]
            x_dim = L.shape[1]
            first_linear.weight.data[:z_dim, :x_dim] = L


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class RetailCVAE(nn.Module):
    """
    Conditional VAE for retail indicator data.

    Parameters
    ----------
    config : dict, optional
    city_map : dict[str, int], optional
        e.g. {"A": 0, "B": 1, "C": 2}.  Auto-built on first fit() call.
    """

    def __init__(
        self,
        config:   dict | None       = None,
        city_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        cfg      = config or load_config()
        vc       = cfg["cvae"]
        dc       = cfg["data"]

        self.score_cols   = dc["score_cols"]
        self.city_col     = dc["city_col"]
        self.x_dim        = len(self.score_cols)
        self.z_dim        = vc["z_dim"]
        self.hidden       = vc["hidden_dim"]
        self.n_cities     = vc["n_cities"]
        self.y_dim        = self.n_cities

        self.epochs           = vc["epochs"]
        self.batch_size       = vc["batch_size"]
        self.lr               = vc["lr"]
        self.beta_start       = vc["beta_start"]
        self.beta_end         = vc["beta_end"]
        self.beta_warmup      = vc["beta_warmup_epochs"]
        self.free_bits        = vc["free_bits"]
        self.n_synthetic      = vc["n_synthetic"]
        self.city_weights_cfg = vc["city_weights"]
        self.seed             = vc["random_seed"]

        self.city_map: dict[str, int] = city_map or {}
        self._fitted  = False

        self.encoder = _Encoder(self.x_dim, self.y_dim, self.z_dim, self.hidden)
        self.decoder = _Decoder(self.z_dim, self.y_dim, self.x_dim, self.hidden)

    # ── ELBO helpers ──────────────────────────────────────────────────────

    def _reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterisation trick: z = μ + σ ⊙ ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, y)
        z          = self._reparameterise(mu, logvar)
        x_hat      = self.decoder(z, y)

        recon = F.mse_loss(x_hat, x, reduction="sum")

        # Per-dim KL with free-bits floor
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.sum(torch.clamp(kl_per_dim, min=self.free_bits))

        loss = recon + beta * kl
        return loss, recon, kl

    # ── public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        score_df: pd.DataFrame,          # KDE scores (0-100) + city column
        loadings: np.ndarray | None = None,
        verbose:  bool = True,
    ) -> "RetailCVAE":
        """
        Train the CVAE.

        Parameters
        ----------
        score_df  : DataFrame with KDE indicator scores + city_col
        loadings  : LFM loading matrix Λ for decoder warm-start (optional)
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build city map from data
        cities = sorted(score_df[self.city_col].unique())
        if not self.city_map:
            self.city_map = {c: i for i, c in enumerate(cities)}
        self.n_cities = max(len(self.city_map), self.n_cities)

        # Warm-start decoder from LFM loadings
        if loadings is not None:
            self.decoder.init_from_loadings(loadings)

        # Tensors
        kde_cols = [f"{c}_kde" for c in self.score_cols
                    if f"{c}_kde" in score_df.columns]
        if not kde_cols:
            kde_cols = [c for c in score_df.columns
                        if c in self.score_cols]
        X = torch.tensor(
            score_df[kde_cols].values / 100.0, dtype=torch.float32
        )
        Y_idx = torch.tensor(
            [self.city_map[c] for c in score_df[self.city_col]],
            dtype=torch.long,
        )
        Y = F.one_hot(Y_idx, num_classes=self.n_cities).float()

        dataset = TensorDataset(X, Y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt     = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            beta = self.beta_start + (self.beta_end - self.beta_start) * min(
                epoch / self.beta_warmup, 1.0
            )
            self.train()
            total_loss = recon_sum = kl_sum = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss, recon, kl = self._elbo(xb, yb, beta)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                recon_sum  += recon.item()
                kl_sum     += kl.item()

            if verbose and (epoch % 50 == 0 or epoch == 1):
                n = len(dataset)
                print(
                    f"[CVAE] epoch {epoch:>4}/{self.epochs}  "
                    f"loss={total_loss/n:.2f}  "
                    f"recon={recon_sum/n:.2f}  "
                    f"KL={kl_sum/n:.2f}  β={beta:.2f}"
                )

        self._fitted = True
        return self

    @torch.no_grad()
    def generate(
        self,
        n: int | None = None,
        city_weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Sample *n* synthetic rows from the prior.

        Returns a DataFrame with the same KDE-score columns as training data
        plus a city_col.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")
        self.eval()
        n = n or self.n_synthetic
        cw = city_weights or self.city_weights_cfg

        # Sample city labels according to weights
        cities  = list(cw.keys())
        weights = np.array([cw[c] for c in cities])
        weights /= weights.sum()
        city_labels = np.random.choice(cities, size=n, p=weights)

        Y_idx = torch.tensor(
            [self.city_map[c] for c in city_labels], dtype=torch.long
        )
        Y = F.one_hot(Y_idx, num_classes=self.n_cities).float()
        Z = torch.randn(n, self.z_dim)

        X_hat = self.decoder(Z, Y).numpy() * 100.0   # back to [0, 100]

        kde_cols = [f"{c}_kde" for c in self.score_cols]
        df_out   = pd.DataFrame(X_hat, columns=kde_cols)
        df_out[self.city_col] = city_labels
        return df_out

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard VAE forward (for nn.Module compatibility)."""
        mu, logvar = self.encoder(x, y)
        z          = self._reparameterise(mu, logvar)
        x_hat      = self.decoder(z, y)
        return x_hat, mu, logvar
