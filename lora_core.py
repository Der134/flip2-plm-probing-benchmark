# -*- coding: utf-8 -*-
"""Core pure-PyTorch LoRA, batching, FLIP2 split loader, and aggregation helpers for Notebook 04."""
import io
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


def confine_path(base: Path, p: Path) -> Path:
    resolved = p.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes base: {resolved}")
    return resolved


def safe_token(tok: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(tok)):
        raise ValueError(f"invalid token: {tok!r}")
    return str(tok)


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name) or ".." in name:
        raise ValueError(f"invalid filename: {name!r}")
    return name


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around a frozen nn.Linear layer (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, self.r, dtype=dtype, device=device))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out + delta


def inject_esm_lora(
    backbone: nn.Module,
    target_names: Tuple[str, ...] = ("query", "key", "value"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> int:
    for p in backbone.parameters():
        p.requires_grad_(False)
    replaced = 0
    for _, parent in list(backbone.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    if replaced == 0:
        raise RuntimeError(f"No linear modules matched {target_names}")
    return replaced


def reset_esm_lora(backbone: nn.Module) -> int:
    count = 0
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.reset_lora_parameters()
            count += 1
    return count


class EsmLoRARegressor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, special_ids: List[int], dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("special_ids", torch.tensor(special_ids, dtype=torch.long), persistent=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        res_mask = attention_mask.bool() & (~torch.isin(input_ids, self.special_ids))
        mask_f = res_mask.unsqueeze(-1).to(torch.float32)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (hidden.to(torch.float32) * mask_f).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


def load_split_df(raw_dir: Path, ds: str, sp: str):
    fname = safe_name(f"{safe_token(ds)}__{safe_token(sp)}.csv.gz")
    raw_path = confine_path(raw_dir, raw_dir / fname)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw split file: {raw_path}. Run notebook_01 first.")
    df = pd.read_csv(raw_path)
    is_train = df["set"].eq("train")
    fit_mask = (is_train & ~df["validation"]).values
    val_mask = ((is_train & df["validation"]) | df["set"].eq("validation")).values
    test_mask = df["set"].eq("test").values
    y_raw = df["target"].astype(float).values
    scaler = StandardScaler().fit(y_raw[is_train.values].reshape(-1, 1))
    y_std = scaler.transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)
    seqs = df["sequence"].astype(str).tolist()
    return seqs, y_std, fit_mask, val_mask, test_mask


def build_token_batches(
    seq_lens: np.ndarray,
    max_tokens: int,
    max_seqs: int,
    shuffle: bool = False,
    rng=None,
    longest_first: bool = False,
) -> List[List[int]]:
    n = len(seq_lens)
    if longest_first:
        order = np.argsort(-seq_lens, kind="mergesort")
    elif shuffle:
        perm = rng.permutation(n)
        bucket_size = max(max_seqs * 8, 128)
        order_chunks = []
        for s in range(0, n, bucket_size):
            chunk = perm[s : s + bucket_size]
            order_chunks.append(chunk[np.argsort(-seq_lens[chunk], kind="mergesort")])
        order = np.concatenate(order_chunks)
    else:
        order = np.argsort(-seq_lens, kind="mergesort")

    batches, cur, cur_max = [], [], 0
    for idx in order.tolist():
        L = int(seq_lens[idx]) + 2
        if not cur:
            cur, cur_max = [idx], L
            continue
        if len(cur) < max_seqs and (len(cur) + 1) * max(cur_max, L) <= max_tokens:
            cur.append(idx)
            cur_max = max(cur_max, L)
        else:
            batches.append(cur)
            cur, cur_max = [idx], L
    if cur:
        batches.append(cur)
    if shuffle and not longest_first:
        bperm = rng.permutation(len(batches))
        batches = [batches[i] for i in bperm.tolist()]
    return batches


def regression_metrics(y_true, y_pred, const_tol: float = 1e-4) -> Dict[str, object]:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    pred_std = float(np.std(y_pred))
    pred_constant = bool(pred_std < const_tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(y_true, y_pred).correlation) if not pred_constant else float("nan")
    y_pos = y_true - y_true.min()
    return {
        "spearman": rho,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ndcg": float(ndcg_score(y_pos[None, :], y_pred[None, :])),
        "pred_std": pred_std,
        "pred_constant": pred_constant,
    }


def save_json_atomic(path: Path, payload: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def aggregate_lora_runs(run_dir: Path, tab_dir: Path, smoke_test: bool = False):
    rows = []
    for p in sorted(run_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("smoke_test", False) != bool(smoke_test):
            continue
        rows.append({k: v for k, v in d.items() if k != "history"})
    if not rows:
        return None, None

    per_seed = pd.DataFrame(rows).sort_values(["dataset", "split", "seed"]).reset_index(drop=True)
    summary = (
        per_seed.groupby(["dataset", "split", "model"], sort=False)
        .agg(
            n_seeds=("seed", "count"),
            spearman_mean=("test_spearman", "mean"),
            spearman_std=("test_spearman", "std"),
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            r2_mean=("test_r2", "mean"),
            ndcg_mean=("test_ndcg", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            total_seconds=("seconds", "sum"),
        )
        .reset_index()
    )

    ridge_csv = confine_path(tab_dir, tab_dir / "ridge_probing_metrics.csv")
    uq_csv = confine_path(tab_dir, tab_dir / "uq_ensemble_metrics.csv")
    if ridge_csv.exists():
        rd = pd.read_csv(ridge_csv)
        rd3b = rd[rd["model"] == "esm2_3b"][["dataset", "split", "spearman", "mse", "ndcg"]].rename(
            columns={"spearman": "frozen_ridge_rho", "mse": "frozen_ridge_mse", "ndcg": "frozen_ridge_ndcg"}
        )
        summary = summary.merge(rd3b, on=["dataset", "split"], how="left")
    if uq_csv.exists():
        uq = pd.read_csv(uq_csv)
        uq3b = uq[uq["model"] == "esm2_3b"][["dataset", "split", "ensemble_spearman", "ensemble_mse", "ensemble_ndcg"]].rename(
            columns={"ensemble_spearman": "frozen_mlp5_rho", "ensemble_mse": "frozen_mlp5_mse", "ensemble_ndcg": "frozen_mlp5_ndcg"}
        )
        summary = summary.merge(uq3b, on=["dataset", "split"], how="left")

    suffix = "_smoke" if smoke_test else ""
    for out_df, fname in ((per_seed, f"lora_3b_per_seed{suffix}.csv"), (summary, f"lora_3b_summary{suffix}.csv")):
        out_p = confine_path(tab_dir, tab_dir / fname)
        tmp_p = out_p.with_name(out_p.name + ".tmp")
        out_df.to_csv(tmp_p, index=False)
        os.replace(tmp_p, out_p)
    return per_seed, summary
