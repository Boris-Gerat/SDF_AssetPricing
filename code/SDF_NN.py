from Utils import stock_retrieve, Fred_MD
from Utils_Stat import tickers_20

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def align_dataframes(Z: pd.DataFrame, R: pd.DataFrame):
    Z = Z.sort_index()
    R = R.sort_index()

    R_next = R.shift(-1)  # next-period returns
    idx = Z.index.intersection(R_next.index)

    X = Z.loc[idx]
    Y = R_next.loc[idx]

    # drop NaNs introduced by shift and any missing values
    mask = (~X.isna().any(axis=1)) & (~Y.isna().any(axis=1))
    return X.loc[mask], Y.loc[mask]


def time_split(
    X: pd.DataFrame, Y: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
):
    assert len(X) == len(Y)
    T = len(X)

    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    n_test = T - n_train - n_val
    assert n_test > 0

    X_train, Y_train = X.iloc[:n_train], Y.iloc[:n_train]
    X_val, Y_val = X.iloc[n_train : n_train + n_val], Y.iloc[n_train : n_train + n_val]
    X_test, Y_test = X.iloc[n_train + n_val :], Y.iloc[n_train + n_val :]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def standardize(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0).replace(0.0, 1.0)

    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma


def to_torch(X: pd.DataFrame, Y: pd.DataFrame, device: torch.device):
    X_t = torch.tensor(X.values, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y.values, dtype=torch.float32, device=device)
    return X_t, Y_t


class NN_SDF(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z).squeeze(-1)  # (T,)
        m = torch.exp(y)  # positive SDF
        return m


def sdf_gmm_loss(
    m: torch.Tensor,
    R_next: torch.Tensor,
    normalize_mean: bool = True,
    lambda_var_m: float = 0.0,
):
    if normalize_mean:
        m = m / (m.mean() + 1e-8)

    g = (m.unsqueeze(1) * R_next).mean(dim=0)  # (N,)
    loss = (g**2).mean()

    if lambda_var_m > 0:
        loss = loss + lambda_var_m * m.var(unbiased=False)

    diagnostics = {
        "loss": loss.detach().item(),
        "g_norm": torch.norm(g).detach().item(),
        "m_mean": m.mean().detach().item(),
        "m_std": m.std(unbiased=False).detach().item(),
        "max_abs_g": g.abs().max().detach().item(),
    }
    return loss, diagnostics


def train_sdf(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    epochs: int = 500,
    normalize_mean: bool = True,
    lambda_var_m: float = 1e-3,
    grad_clip: float = 1.0,
    print_every: int = 50,
    device: torch.device = torch.device("cpu"),
):
    best_state = None
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        optimizer.zero_grad()

        m_train = model(X_train)
        loss_train, diag_train = sdf_gmm_loss(
            m_train,
            Y_train,
            normalize_mean=normalize_mean,
            lambda_var_m=lambda_var_m,
        )
        loss_train.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            m_val = model(X_val)
            loss_val, diag_val = sdf_gmm_loss(
                m_val,
                Y_val,
                normalize_mean=normalize_mean,
                lambda_var_m=0.0,
            )

        # select best by moment norm
        val_metric = diag_val["g_norm"]
        if val_metric < best_val:
            best_val = val_metric
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if ep == 1 or ep % print_every == 0:
            print(
                f"ep {ep:4d} | "
                f"train loss {diag_train['loss']:.3e} g_norm {diag_train['g_norm']:.3e} "
                f"m_mean {diag_train['m_mean']:.3f} m_std {diag_train['m_std']:.3f} | "
                f"val loss {diag_val['loss']:.3e} g_norm {diag_val['g_norm']:.3e} "
                f"m_mean {diag_val['m_mean']:.3f} m_std {diag_val['m_std']:.3f}"
            )

    # restore best model AFTER training
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_val


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # safer device selection for Mac
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print("DEVICE:", DEVICE)

    econ_state = Fred_MD(start="2011-01-01", factors=10, factor_verbose=True)
    stock_data = stock_retrieve(tickers=tickers_20, start="2011-01-01", returns=True)

    X, Y = align_dataframes(econ_state, stock_data)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = time_split(X, Y)
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    X_train_t, Y_train_t = to_torch(X_train, Y_train, DEVICE)
    X_val_t, Y_val_t = to_torch(X_val, Y_val, DEVICE)
    X_test_t, Y_test_t = to_torch(X_test, Y_test, DEVICE)

    model = NN_SDF(d_in=X_train_t.shape[1], hidden_dim=16).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model, best_val_metric = train_sdf(
        model=model,
        optimizer=optimizer,
        X_train=X_train_t,
        Y_train=Y_train_t,
        X_val=X_val_t,
        Y_val=Y_val_t,
        epochs=500,
        normalize_mean=True,
        lambda_var_m=1e-3,
        grad_clip=1.0,
        print_every=50,
        device=DEVICE,
    )

    print("Best val g_norm:", best_val_metric)


if __name__ == "__main__":
    main()
