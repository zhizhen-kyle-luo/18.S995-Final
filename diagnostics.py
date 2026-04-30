#!/usr/bin/env python3
"""
DistilBERT numerical matrix diagnostics: core LayerNorm vs core DyT.

Extracts hidden states from distilbert-base-uncased, applies the core
non-affine LayerNorm and DyT maps post-hoc to the same feature matrices,
and compares using SVD-based diagnostics.

Interpretation guide
--------------------
* Similar spectra/ranks across transforms => DyT is numerically LN-like.
* Differing spectra => task-level replacement does not imply matrix-level
  equivalence.
* Differences concentrated in G => map changes representation geometry.
* Differences concentrated in S => QK projection amplifies the map difference.
* Strong dependence on alpha => DyT behavior is tunable, not inherently LN-like.

Outputs (in --output_dir)
-------------------------
  metrics_summary.csv        spectral metrics for X, G, S per layer/transform
  low_rank_errors.csv        relative Frobenius best-rank-k errors
  gram_diagonal_summary.csv  Gram diagonal statistics per layer/transform
  attention_perturbation.csv perturbation sensitivity of S (if not --skip_attention)
  fig_*.png                  diagnostic plots

Install
-------
  uv sync

Run (fast, feature/Gram only)
------------------------------
  uv run python diagnostics.py --num_texts 40 --max_tokens 256 --skip_attention

Run (full, ~2 min on CPU)
--------------------------
  uv run python diagnostics.py --num_texts 120 --max_tokens 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

_BASE_TEXTS: List[str] = [
    "Transformers use attention mechanisms to compare token representations.",
    "Layer normalization rescales each token vector using row statistics.",
    "Dynamic Tanh replaces normalization with a coordinatewise saturating nonlinearity.",
    "Numerical linear algebra studies conditioning, stability, rank, and low-rank approximation.",
    "The Gram matrix stores pairwise inner products between token representations.",
    "Attention scores are formed from query and key matrices before the softmax operation.",
    "A matrix with rapidly decaying singular values admits efficient low-rank approximation.",
    "Condition numbers measure how sensitive a numerical problem is to input perturbations.",
    "The singular value decomposition reveals the geometry of a linear transformation.",
    "This experiment compares matrices produced by LayerNorm and Dynamic Tanh post-hoc.",
    "Different neural network components can have similar accuracy but different matrix structure.",
    "A controlled post-hoc comparison isolates the effect of the normalization map itself.",
    "DistilBERT is a compact pretrained transformer model for language representations.",
    "The hidden states of a transformer form matrices whose rows correspond to tokens.",
    "The goal is not benchmark accuracy but numerical similarity of feature matrices.",
    "Effective rank measures how many singular values contribute meaningfully.",
    "Stable rank is the ratio of squared Frobenius norm to squared spectral norm.",
    "The effective condition number ignores singular values below a relative threshold.",
    "Low-rank approximation error decays faster when singular values decay rapidly.",
    "Alpha controls the saturation regime of the DyT nonlinearity.",
    "LayerNorm normalizes each token independently across the feature dimension.",
    "Per-head attention allows different heads to capture different relational patterns.",
    "The embedding layer maps discrete tokens to continuous vector representations.",
    "Self-attention computes all pairwise interactions between tokens in a sequence.",
    "Pretrained language models encode rich syntactic and semantic information.",
    "The spectral norm of a matrix equals its largest singular value.",
    "Frobenius norm is the square root of the sum of squared matrix entries.",
    "Matrix conditioning determines how numerical errors propagate through computations.",
    "DyT with small alpha behaves approximately linearly near zero.",
    "LayerNorm is invariant to row-wise shifts and scales in the input.",
]


def get_texts(num_texts: int, text_file: Optional[str] = None) -> List[str]:
    if text_file is not None:
        with open(text_file) as fh:
            base = [line.strip() for line in fh if line.strip()]
    else:
        base = _BASE_TEXTS
    if not base:
        raise ValueError("No texts available.")
    return (base * (num_texts // len(base) + 1))[:num_texts]


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available on this machine.")
    if name == "mps" and not _mps_available():
        raise RuntimeError("--device mps requested but MPS is not available on this machine.")
    return torch.device(name)


def core_ln(X: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Rowwise zero-mean unit-variance normalization."""
    mu = X.mean(dim=-1, keepdim=True)
    var = ((X - mu) ** 2).mean(dim=-1, keepdim=True)
    return (X - mu) / torch.sqrt(var + eps)


def core_dyt(X: torch.Tensor, alpha: float) -> torch.Tensor:
    """Elementwise tanh(alpha * X)."""
    return torch.tanh(alpha * X)


def apply_transform(
    X: torch.Tensor,
    transform: str,
    alpha: Optional[float],
    eps: float,
) -> torch.Tensor:
    if transform == "raw":
        return X
    if transform == "ln":
        return core_ln(X, eps=eps)
    if transform == "dyt":
        if alpha is None:
            raise ValueError("alpha required for dyt transform")
        return core_dyt(X, alpha=alpha)
    raise ValueError(f"Unknown transform: {transform!r}")


def transform_label(transform: str, alpha: Optional[float]) -> str:
    return transform if alpha is None else f"{transform}_a{alpha}"


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train(False)
    mtype = getattr(model.config, "model_type", None)
    if mtype != "distilbert":
        raise ValueError(
            f"this script targets DistilBERT (model.transformer.layer[i].attention.q_lin etc.); "
            f"got model_type={mtype!r}. pass --model_name pointing to a distilbert checkpoint."
        )
    return model, tokenizer


def collect_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """returns (hidden_by_state, masks): index 0 is embeddings, l>=1 is layer l output."""
    hidden_by_state: Optional[List[List[torch.Tensor]]] = None
    masks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hs = [h.detach().cpu() for h in out.hidden_states]
            if hidden_by_state is None:
                hidden_by_state = [[] for _ in hs]
            for i, h in enumerate(hs):
                hidden_by_state[i].append(h)
            masks.append(enc["attention_mask"].detach().cpu())

    if hidden_by_state is None:
        raise RuntimeError("no hidden states collected (empty input?)")
    return hidden_by_state, masks


def aggregate_tokens(
    state_batches: List[torch.Tensor],
    masks: List[torch.Tensor],
    max_tokens: int,
    seed: int = 42,
) -> torch.Tensor:
    """flatten non-padding token rows; subsample to max_tokens with fixed seed."""
    parts: List[torch.Tensor] = []
    for h, mask in zip(state_batches, masks):
        parts.append(h[mask.bool()])  # (n_valid, D)
    X = torch.cat(parts, dim=0)
    if X.shape[0] > max_tokens:
        rng = torch.Generator()
        rng.manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=rng)[:max_tokens]
        X = X[idx]
    return X


def svdvals_cpu_f64(M: torch.Tensor) -> np.ndarray:
    return torch.linalg.svdvals(M.detach().cpu().double()).numpy()


def spectrum_metrics(s: np.ndarray, rel_tol: float = 1e-6) -> Dict:
    s = np.asarray(s, dtype=np.float64)
    s = np.sort(s[np.isfinite(s) & (s >= 0)])[::-1]

    if len(s) == 0 or s[0] <= 0:
        raise ValueError("spectrum is empty or has non-positive top singular value")

    thresh = rel_tol * s[0]
    keep = s[s > thresh]
    numerical_rank = int(len(keep))
    kappa_eff = float(keep[0] / keep[-1]) if numerical_rank > 0 else np.nan

    fro_sq = float(np.sum(s ** 2))
    stable_rank = float(fro_sq / s[0] ** 2)
    fro_norm = float(np.sqrt(fro_sq))

    s_sum = float(np.sum(s))
    if s_sum > 0:
        p = s / s_sum
        effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-300))))
    else:
        effective_rank = np.nan

    return dict(
        sigma1=float(s[0]),
        numerical_rank=numerical_rank,
        kappa_eff=kappa_eff,
        stable_rank=stable_rank,
        effective_rank=effective_rank,
        fro_norm=fro_norm,
    )


def compute_svd_metrics(M: torch.Tensor, rel_tol: float) -> Tuple[Dict, np.ndarray]:
    s = svdvals_cpu_f64(M)
    return spectrum_metrics(s, rel_tol=rel_tol), s


def compute_low_rank_errors(s: np.ndarray, ks: List[int]) -> Dict[int, float]:
    """Relative Frobenius best rank-k approximation error."""
    s = np.asarray(s, dtype=np.float64)
    denom = np.sum(s ** 2)
    out: Dict[int, float] = {}
    for k in ks:
        if denom <= 0:
            out[k] = np.nan
        elif k >= len(s):
            out[k] = 0.0
        else:
            out[k] = float(np.sqrt(np.sum(s[k:] ** 2) / denom))
    return out


def compute_gram(X: torch.Tensor) -> torch.Tensor:
    return X @ X.T


def compute_gram_diag_summary(G: torch.Tensor) -> Dict:
    diag = torch.diag(G).detach().cpu().float().numpy()
    return dict(
        diag_min=float(np.min(diag)),
        diag_max=float(np.max(diag)),
        diag_mean=float(np.mean(diag)),
        diag_std=float(np.std(diag)),
        trace=float(np.sum(diag)),
    )


def get_qk_projections(model, layer_idx: int):
    """returns (q_lin, k_lin, n_heads, head_dim) for distilbert layer layer_idx, on cpu."""
    attn = model.transformer.layer[layer_idx].attention
    q_lin = attn.q_lin.cpu()
    k_lin = attn.k_lin.cpu()
    n_heads = int(attn.n_heads)
    head_dim = model.config.dim // n_heads
    q_lin.train(False)
    k_lin.train(False)
    return q_lin, k_lin, n_heads, head_dim


def compute_attention_scores(
    model,
    layer_idx: int,
    state_batches: List[torch.Tensor],
    masks: List[torch.Tensor],
    transform: str,
    alpha: Optional[float],
    eps: float,
    max_sequences: int = 32,
) -> List[Tuple[int, torch.Tensor]]:
    """per-head S_h = Q_h K_h^T / sqrt(head_dim) for up to max_sequences sequences."""
    q_lin, k_lin, n_heads, head_dim = get_qk_projections(model, layer_idx)

    results: List[Tuple[int, torch.Tensor]] = []
    seen = 0

    with torch.no_grad():
        for h_batch, mask_batch in zip(state_batches, masks):
            if seen >= max_sequences:
                break
            for b in range(h_batch.shape[0]):
                if seen >= max_sequences:
                    break
                mask = mask_batch[b].bool()
                X_seq = h_batch[b, mask, :]
                if X_seq.shape[0] < 2:
                    continue
                X_t = apply_transform(X_seq, transform, alpha, eps)
                Q = q_lin(X_t)
                K = k_lin(X_t)
                T = X_t.shape[0]
                Qh = Q.view(T, n_heads, head_dim).transpose(0, 1)  # (H, T, head_dim)
                Kh = K.view(T, n_heads, head_dim).transpose(0, 1)
                for h in range(n_heads):
                    S = (Qh[h] @ Kh[h].T) / math.sqrt(head_dim)
                    results.append((h, S))
                seen += 1

    return results


def compute_attention_perturbation(
    model,
    layer_idx: int,
    state_batches: List[torch.Tensor],
    masks: List[torch.Tensor],
    transform: str,
    alpha: Optional[float],
    eps: float,
    perturb_scales: List[float],
    max_sequences: int = 16,
    seed: int = 42,
) -> List[Dict]:
    """relative Gaussian perturbations of Q and K; reports ||dS||_F/||S||_F and 2-norm version."""
    q_lin, k_lin, n_heads, head_dim = get_qk_projections(model, layer_idx)

    rng = torch.Generator()
    rng.manual_seed(seed)
    rows: List[Dict] = []
    seen = 0

    with torch.no_grad():
        for h_batch, mask_batch in zip(state_batches, masks):
            if seen >= max_sequences:
                break
            for b in range(h_batch.shape[0]):
                if seen >= max_sequences:
                    break
                mask = mask_batch[b].bool()
                X_seq = h_batch[b, mask, :]
                if X_seq.shape[0] < 2:
                    continue
                X_t = apply_transform(X_seq, transform, alpha, eps)
                Q = q_lin(X_t)
                K = k_lin(X_t)
                T = X_t.shape[0]
                Qh = Q.view(T, n_heads, head_dim).transpose(0, 1)
                Kh = K.view(T, n_heads, head_dim).transpose(0, 1)

                for h in range(n_heads):
                    S_ref = (Qh[h] @ Kh[h].T) / math.sqrt(head_dim)
                    s_ref_fro = float(torch.norm(S_ref, "fro"))
                    s_ref_2 = float(torch.linalg.norm(S_ref, ord=2))
                    if s_ref_fro < 1e-12:
                        continue

                    for scale in perturb_scales:
                        dQ = torch.randn(Qh[h].shape, generator=rng)
                        dK = torch.randn(Kh[h].shape, generator=rng)
                        qh_fro = float(torch.norm(Qh[h], "fro"))
                        kh_fro = float(torch.norm(Kh[h], "fro"))
                        if qh_fro > 0:
                            dQ = dQ * (scale * qh_fro / (float(torch.norm(dQ, "fro")) + 1e-30))
                        if kh_fro > 0:
                            dK = dK * (scale * kh_fro / (float(torch.norm(dK, "fro")) + 1e-30))
                        S_pert = ((Qh[h] + dQ) @ (Kh[h] + dK).T) / math.sqrt(head_dim)
                        dS = S_pert - S_ref
                        rows.append(dict(
                            layer=layer_idx,
                            transform=transform,
                            alpha=np.nan if alpha is None else alpha,
                            head=h,
                            perturb_scale=scale,
                            rel_delta_fro=float(torch.norm(dS, "fro")) / (s_ref_fro + 1e-30),
                            rel_delta_2=float(torch.linalg.norm(dS, ord=2)) / (s_ref_2 + 1e-30),
                        ))
                seen += 1

    return rows


def _series_col(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: r["transform"] if pd.isna(r["alpha"]) else f'{r["transform"]}_a{r["alpha"]}',
        axis=1,
    )


def plot_spectrum(
    spectra: Dict[str, np.ndarray],
    title: str,
    path: Path,
    max_i: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, s in spectra.items():
        if len(s) == 0 or s[0] <= 0:
            continue
        y = s[:max_i] / s[0]
        ax.semilogy(np.arange(1, len(y) + 1), y, label=name)
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lines(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    path: Path,
    ylabel: Optional[str] = None,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sub in df.groupby(group):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", markersize=4, label=str(name))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend(fontsize=7)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_low_rank_curve(
    df: pd.DataFrame,
    layer: str,
    matrix: str,
    title: str,
    path: Path,
) -> None:
    sub = df[(df["layer"] == layer) & (df["matrix_type"] == matrix)].copy()
    if sub.empty:
        return
    sub["series"] = _series_col(sub)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in sub.groupby("series"):
        grp = grp.sort_values("k")
        ax.semilogy(grp["k"], grp["rel_fro_error"].clip(lower=1e-16),
                    marker="o", markersize=4, label=name)
    ax.set_xlabel("rank k")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gram_diag_bars(
    gram_diag_df: pd.DataFrame,
    layer: str,
    path: Path,
    d: int,
    transforms: List[Tuple[str, Optional[float]]],
) -> None:
    """bar chart of Gram diagonal mean and std at the given layer, in transforms order."""
    sub = gram_diag_df[gram_diag_df["layer"] == layer].copy()
    if sub.empty:
        return
    def _label(t: str, a: Optional[float]) -> str:
        if t == "raw":
            return "raw"
        if t == "ln":
            return "LN"
        return rf"DyT $\alpha={a}$"
    order: List[Tuple[str, Optional[float], str]] = [
        (t, a, _label(t, a)) for t, a in transforms
    ]
    means, stds, labels = [], [], []
    for t, a, lab in order:
        if a is None:
            row = sub[(sub["transform"] == t) & sub["alpha"].isna()]
        else:
            row = sub[(sub["transform"] == t) & (sub["alpha"] == a)]
        if row.empty:
            continue
        means.append(float(row["diag_mean"].iloc[0]))
        stds.append(float(row["diag_std"].iloc[0]))
        labels.append(lab)
    if not means:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(labels))
    colors = ["#888888", "#1f77b4", "#aec7e8", "#7fbf7f", "#2ca02c"][: len(labels)]
    ax.bar(x, means, yerr=stds, capsize=6, color=colors,
           edgecolor="black", linewidth=0.6)
    ax.axhline(d, color="red", linestyle="--", linewidth=1,
               label=rf"$d={d}$ (LN prediction)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gram diagonal mean (with std as error bars)")
    ax.set_title(f"Row norms at {layer}: $\\mathrm{{diag}}(G)$ across transforms")
    ax.legend(loc="upper right", fontsize=9)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + max(stds) * 0.4, f"{m:.1f}\n$\\pm${s:.2f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_plots(
    metrics_df: pd.DataFrame,
    low_rank_df: pd.DataFrame,
    gram_diag_df: pd.DataFrame,
    spectra_X: Dict[str, np.ndarray],
    spectra_G: Dict[str, np.ndarray],
    final_layer: str,
    out_dir: Path,
    hidden_dim: int,
    transforms: List[Tuple[str, Optional[float]]],
) -> None:
    plot_spectrum(spectra_X, f"Feature singular value decay (layer {final_layer})",
                  out_dir / "fig_spectrum_X_final.png")
    plot_spectrum(spectra_G, f"Gram singular value decay (layer {final_layer})",
                  out_dir / "fig_spectrum_G_final.png")

    for matrix in ["X", "G", "S"]:
        sub = metrics_df[metrics_df["matrix_type"] == matrix].copy()
        if sub.empty:
            continue
        sub["series"] = _series_col(sub)
        plot_lines(sub, x="layer", y="stable_rank", group="series",
                   title=f"Stable rank across layers ({matrix})",
                   path=out_dir / f"fig_stable_rank_{matrix}.png")
        plot_lines(sub, x="layer", y="kappa_eff", group="series",
                   title=f"Effective condition number across layers ({matrix})",
                   path=out_dir / f"fig_kappa_eff_{matrix}.png", logy=True)

    gd = gram_diag_df.copy()
    gd["series"] = _series_col(gd)
    plot_lines(gd, x="layer", y="diag_std", group="series",
               title="Gram diagonal std across layers",
               path=out_dir / "fig_gram_diag_std.png")
    plot_lines(gd, x="layer", y="diag_mean", group="series",
               title="Gram diagonal mean across layers",
               path=out_dir / "fig_gram_diag_mean.png")

    plot_gram_diag_bars(gd, layer=final_layer,
                        path=out_dir / "fig_gram_diag_layer6_bars.png",
                        d=hidden_dim, transforms=transforms)

    for matrix in ["X", "G"]:
        plot_low_rank_curve(
            low_rank_df, layer=final_layer, matrix=matrix,
            title=f"Low-rank approximation error (layer {final_layer}, {matrix})",
            path=out_dir / f"fig_low_rank_{matrix}_final.png",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DistilBERT numerical diagnostics: LN vs DyT")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--num_texts", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=512,
                   help="Max non-padding tokens per layer matrix.")
    p.add_argument("--alphas", type=str, default="0.25,0.5,1.0",
                   help="Comma-separated DyT alpha values.")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--rel_tol", type=float, default=1e-6)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip_attention", action="store_true",
                   help="Skip attention-score and perturbation diagnostics.")
    p.add_argument("--text_file", type=str, default=None,
                   help="Optional path to a text file (one sentence per line).")
    p.add_argument("--perturb_scales", type=str, default="0.0001,0.001,0.01",
                   help="Comma-separated relative perturbation scales.")
    args = p.parse_args()

    for name in ("num_texts", "batch_size", "max_length", "max_tokens"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be >= 1")
    if args.eps <= 0:
        p.error("--eps must be > 0")
    if args.rel_tol <= 0:
        p.error("--rel_tol must be > 0")

    def _parse_float_list(raw: str, name: str) -> List[float]:
        items = [tok.strip() for tok in raw.split(",") if tok.strip()]
        if not items:
            p.error(f"--{name} must contain at least one positive value")
        try:
            vals = [float(t) for t in items]
        except ValueError:
            p.error(f"--{name} must be a comma-separated list of floats")
        if any(v <= 0 for v in vals):
            p.error(f"--{name} entries must all be > 0")
        return vals

    args.alphas = _parse_float_list(args.alphas, "alphas")
    args.perturb_scales = _parse_float_list(args.perturb_scales, "perturb_scales")
    return args


def state_name(idx: int) -> str:
    return "embed" if idx == 0 else f"layer_{idx}"


def run_feature_gram_diagnostics(
    hidden_by_state: List[List[torch.Tensor]],
    masks: List[torch.Tensor],
    transforms: List[Tuple[str, Optional[float]]],
    max_tokens: int,
    eps: float,
    rel_tol: float,
    ks: List[int],
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """spectral diagnostics on X and G for every hidden state and every transform."""
    n_states = len(hidden_by_state)
    state_indices = list(range(n_states))
    final = state_name(state_indices[-1])

    metrics: List[Dict] = []
    low_rank: List[Dict] = []
    gram_diag: List[Dict] = []
    spectra_X: Dict[str, np.ndarray] = {}
    spectra_G: Dict[str, np.ndarray] = {}

    for state_idx in tqdm(state_indices, desc="feature/gram diagnostics"):
        layer = state_name(state_idx)
        X_raw = aggregate_tokens(hidden_by_state[state_idx], masks, max_tokens)

        for transform, alpha in transforms:
            label = transform_label(transform, alpha)
            X = apply_transform(X_raw, transform, alpha, eps)

            met_X, sX = compute_svd_metrics(X, rel_tol)
            metrics.append(dict(
                layer=layer, matrix_type="X",
                transform=transform, alpha=np.nan if alpha is None else alpha,
                n_rows=X.shape[0], n_cols=X.shape[1],
                top_sv=met_X.get("sigma1", np.nan), **met_X,
            ))
            for k, err in compute_low_rank_errors(sX, ks).items():
                low_rank.append(dict(
                    layer=layer, matrix_type="X",
                    transform=transform, alpha=np.nan if alpha is None else alpha,
                    k=k, rel_fro_error=err,
                ))

            G = compute_gram(X)
            gram_diag.append(dict(
                layer=layer, transform=transform,
                alpha=np.nan if alpha is None else alpha,
                **compute_gram_diag_summary(G),
            ))
            met_G, sG = compute_svd_metrics(G, rel_tol)
            metrics.append(dict(
                layer=layer, matrix_type="G",
                transform=transform, alpha=np.nan if alpha is None else alpha,
                n_rows=G.shape[0], n_cols=G.shape[1],
                top_sv=met_G.get("sigma1", np.nan), **met_G,
            ))
            for k, err in compute_low_rank_errors(sG, ks).items():
                low_rank.append(dict(
                    layer=layer, matrix_type="G",
                    transform=transform, alpha=np.nan if alpha is None else alpha,
                    k=k, rel_fro_error=err,
                ))

            if layer == final:
                spectra_X[label] = sX
                spectra_G[label] = sG

    return metrics, low_rank, gram_diag, spectra_X, spectra_G


def run_attention_diagnostics(
    model,
    hidden_by_state: List[List[torch.Tensor]],
    masks: List[torch.Tensor],
    n_transformer_layers: int,
    transforms: List[Tuple[str, Optional[float]]],
    eps: float,
    rel_tol: float,
    perturb_scales: List[float],
) -> Tuple[List[Dict], List[Dict]]:
    """spectral metrics of S and perturbation response for each (layer, transform)."""
    metrics: List[Dict] = []
    perturb: List[Dict] = []

    for t_idx in tqdm(range(n_transformer_layers), desc="attention diagnostics"):
        # attention at transformer layer t_idx consumes hidden_by_state[t_idx] as input.
        layer = state_name(t_idx + 1)
        state_batches = hidden_by_state[t_idx]

        for transform, alpha in transforms:
            head_results = compute_attention_scores(
                model, t_idx, state_batches, masks,
                transform, alpha, eps, max_sequences=32,
            )
            if head_results:
                met_list = [compute_svd_metrics(S, rel_tol)[0] for _, S in head_results]
                avg = {k: float(np.nanmean([r[k] for r in met_list])) for k in met_list[0]}
                metrics.append(dict(
                    layer=layer, matrix_type="S",
                    transform=transform, alpha=np.nan if alpha is None else alpha,
                    n_rows=np.nan, n_cols=np.nan,
                    top_sv=avg.get("sigma1", np.nan), **avg,
                ))

            p_rows = compute_attention_perturbation(
                model, t_idx, state_batches, masks,
                transform, alpha, eps,
                perturb_scales=perturb_scales, max_sequences=16,
            )
            for r in p_rows:
                r["layer"] = layer
            perturb.extend(p_rows)

    return metrics, perturb


_GENERATED_FILES = (
    "metrics_summary.csv",
    "low_rank_errors.csv",
    "gram_diagonal_summary.csv",
    "attention_perturbation.csv",
    "fig_spectrum_X_final.png",
    "fig_spectrum_G_final.png",
    "fig_gram_diag_std.png",
    "fig_gram_diag_mean.png",
    "fig_gram_diag_layer6_bars.png",
    "fig_low_rank_X_final.png",
    "fig_low_rank_G_final.png",
    "fig_stable_rank_X.png",
    "fig_stable_rank_G.png",
    "fig_stable_rank_S.png",
    "fig_kappa_eff_X.png",
    "fig_kappa_eff_G.png",
    "fig_kappa_eff_S.png",
)


def clear_stale_outputs(out_dir: Path) -> None:
    for name in _GENERATED_FILES:
        f = out_dir / name
        if f.exists():
            f.unlink()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(out_dir)

    transforms: List[Tuple[str, Optional[float]]] = (
        [("raw", None), ("ln", None)] + [("dyt", a) for a in args.alphas]
    )
    ks = [1, 2, 4, 8, 16, 32, 64, 128]

    device = choose_device(args.device)
    print(f"device: {device}")
    texts = get_texts(args.num_texts, args.text_file)
    print(f"texts: {len(texts)}")
    model, tokenizer = load_model(args.model_name, device)
    print(f"model: {args.model_name}")

    hidden_by_state, masks = collect_hidden_states(
        model, tokenizer, texts, args.batch_size, args.max_length, device
    )
    n_transformer_layers = model.config.n_layers
    n_states = len(hidden_by_state)
    if n_states != n_transformer_layers + 1:
        raise RuntimeError(
            f"expected {n_transformer_layers + 1} hidden states, got {n_states}"
        )
    final_layer = state_name(n_states - 1)

    metrics_rows, low_rank_rows, gram_diag_rows, spectra_X, spectra_G = (
        run_feature_gram_diagnostics(
            hidden_by_state, masks, transforms,
            args.max_tokens, args.eps, args.rel_tol, ks,
        )
    )

    perturb_rows: List[Dict] = []
    if not args.skip_attention:
        s_metrics, perturb_rows = run_attention_diagnostics(
            model, hidden_by_state, masks, n_transformer_layers,
            transforms, args.eps, args.rel_tol, args.perturb_scales,
        )
        metrics_rows.extend(s_metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    low_rank_df = pd.DataFrame(low_rank_rows)
    gram_diag_df = pd.DataFrame(gram_diag_rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    low_rank_df.to_csv(out_dir / "low_rank_errors.csv", index=False)
    gram_diag_df.to_csv(out_dir / "gram_diagonal_summary.csv", index=False)
    if perturb_rows:
        pd.DataFrame(perturb_rows).to_csv(out_dir / "attention_perturbation.csv", index=False)

    make_plots(metrics_df, low_rank_df, gram_diag_df,
               spectra_X, spectra_G, final_layer, out_dir,
               hidden_dim=int(model.config.dim),
               transforms=transforms)

    print(f"\noutputs saved to: {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
