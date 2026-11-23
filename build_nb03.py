# -*- coding: utf-8 -*-
"""Builder for notebook_03_probing_data_efficiency_uq.ipynb (W1 + W2 unified)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = []
_ID = 0


def _next_id():
    global _ID
    _ID += 1
    return f"cell-{_ID:02d}"


def md(src):
    CELLS.append({"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "id": _next_id(), "execution_count": None, "metadata": {}, "outputs": [], "source": src})


md("""# 03 — FLIP2 Probing, Normalization Ablation, Data Efficiency & UQ (W1 + W2)

**Resource tier: CPU runtime (~0 CU).** Evaluates all cached representations across W1 + W2
(up to 6 datasets, 13 official splits: 12 regression + 1 ordinal 4-bin classification `nucb`).

**Upgrades over W1 pilot:**
1. **3-way feature scaling ablation** (`raw`, `l2_norm`, `zscore_fit`) with `val_mse`-selected
   Ridge $\\alpha \\in [10^{-4}, 10^6]$ (diagnoses variable-length multi-wild-type scaling vs single-WT).
2. **NucB dedicated pipeline** (`accuracy_4bin`, `macro_f1_4bin`, `auroc_bin_ge2`, `auprc_bin_ge2`,
   plus ordinal `spearman` / `ndcg` — safely handles `bin=3` absent from `fit`).
3. **TrpB memory safety**: sparse CSR `one_hot`, sequential scaling with `gc.collect()`,
   mini-batch MLP training for large splits, and automatic fallback to `trpb_sub50k` if full
   3B cache is absent (explicitly tagged in `eval_subset`).
""")

code('''# Cell 1 — Setup, workspace & configuration [CPU, ~0 CU]
import os, sys, json, re, io, gc, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (mean_squared_error, r2_score, ndcg_score,
                             accuracy_score, f1_score, roc_auc_score, average_precision_score)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", message=".*Singular matrix.*")
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    BASE_DIR = Path("/content/drive/MyDrive/flip2_benchmark")
else:
    BASE_DIR = Path("flip2_benchmark").resolve()

PROC_DIR  = (BASE_DIR / "data" / "processed").resolve()
CACHE_DIR = (BASE_DIR / "cache" / "embeddings").resolve()
OUT_DIR   = (BASE_DIR / "outputs").resolve()
FIG_DIR   = (OUT_DIR / "figures").resolve()
TAB_DIR   = (OUT_DIR / "tables").resolve()
for d in (FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _confine(base, p):
    p = p.resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes base: {p}")
    return p

def _safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_.\\-]+", name) or ".." in name:
        raise ValueError(f"invalid filename: {name!r}")
    return name

def safe_proc_path(name):  return _confine(PROC_DIR, PROC_DIR / _safe_name(name))
def safe_cache_path(name): return _confine(CACHE_DIR, CACHE_DIR / _safe_name(name))
def safe_tab_path(name):   return _confine(TAB_DIR, TAB_DIR / _safe_name(name))
def safe_fig_path(name):   return _confine(FIG_DIR, FIG_DIR / _safe_name(name))

# ---- Configuration ----
RUN_SCOPE = "all"   # "w1" | "w2" | "all"
W1_SPLITS = [("amylase", "one_to_many"), ("amylase", "close_to_far"), ("rhomax", "by_wild_type")]
W2_SPLITS = [
    ("ired",  "two_to_many"),
    ("hydro", "three_to_many"), ("hydro", "low_to_high"),
    ("hydro", "to_P06241"), ("hydro", "to_P0A9X9"), ("hydro", "to_P01053"),
    ("nucb",  "two_to_many"),
    ("trpb",  "one_to_many"), ("trpb", "two_to_many"), ("trpb", "by_position"),
]
ACTIVE_SPLITS = W1_SPLITS if RUN_SCOPE == "w1" else (W2_SPLITS if RUN_SCOPE == "w2" else W1_SPLITS + W2_SPLITS)
TASK_TYPES = {"amylase": "regression", "rhomax": "regression", "ired": "regression",
              "hydro": "regression", "nucb": "classification", "trpb": "regression"}

PLM_MODELS = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]
MODELS = ["one_hot"] + PLM_MODELS          # local test: set MODELS = ["one_hot"]
UQ_MODELS = ["esm2_650m", "esm2_3b"]       # local test: set UQ_MODELS = ["one_hot"]
SEEDS = [0, 1, 2]
FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
ALPHAS = np.logspace(-4, 6, 21)
N_ENSEMBLE = 5
CONST_TOL = 1e-4
torch.manual_seed(0)

man_file = "benchmark_manifest.json" if (PROC_DIR / "benchmark_manifest.json").exists() else "pilot_manifest.json"
manifest = json.loads(safe_proc_path(man_file).read_text(encoding="utf-8"))
print(f"[CFG] RUN_SCOPE={RUN_SCOPE} ({len(ACTIVE_SPLITS)} splits) | manifest={man_file}")
print(f"[CFG] MODELS={MODELS} | UQ_MODELS={UQ_MODELS}")''')

code('''# Cell 2 — Helpers: sparse one-hot, alignment, scaling, Ridge & metrics [CPU, ~0 CU]
AAINDEX_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AAINDEX_ALPHABET)}

def encode_onehot_csr(sequences, max_len: int) -> sparse.csr_matrix:
    n = len(sequences)
    lens = np.fromiter((min(len(s), max_len) for s in sequences), dtype=np.int64, count=n)
    rows = np.repeat(np.arange(n, dtype=np.int64), lens)
    cols = np.empty(len(rows), dtype=np.int64)
    pos = 0
    for s in sequences:
        L = min(len(s), max_len)
        for j in range(L):
            cols[pos + j] = j * 20 + AA_TO_IDX[s[j]]
        pos += L
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(n, max_len * 20), dtype=np.float32)

def make_masks(df):
    is_train = df["set"].eq("train")
    fit = (is_train & ~df["validation"]).values
    val = ((is_train & df["validation"]) | df["set"].eq("validation")).values
    test = df["set"].eq("test").values
    return fit, val, test

def scale_targets(df):
    y = df["target"].astype(float).values
    train_all = df["set"].eq("train").values
    scaler = StandardScaler().fit(y[train_all].reshape(-1, 1))
    return scaler.transform(y.reshape(-1, 1)).ravel(), scaler

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    pred_std = float(np.std(y_pred))
    pred_constant = bool(pred_std < CONST_TOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(y_true, y_pred).correlation) if not pred_constant else float("nan")
    y_pos = y_true - y_true.min()
    return {"spearman": rho, "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "ndcg": float(ndcg_score(y_pos[None, :], y_pred[None, :])),
            "pred_std": pred_std, "pred_constant": pred_constant}

def apply_scaling(X, ref_idx_or_mask, scaling: str):
    if scaling == "raw":
        return X
    if scaling == "l2_norm":
        return normalize(X, norm="l2", axis=1, copy=True)
    if scaling == "zscore_fit":
        if sparse.issparse(X):
            raise ValueError("zscore_fit disallowed on sparse one_hot (use raw or l2_norm)")
        mu = X[ref_idx_or_mask].mean(0)
        sd = X[ref_idx_or_mask].std(0); sd[sd < 1e-8] = 1.0
        return ((X - mu) / sd).astype(np.float32)
    raise ValueError(f"unknown scaling: {scaling}")

def primary_scaling(ds, model_key):
    if model_key == "one_hot":
        return "raw"
    # Same-WT fixed-length deep-mutational scans benefit from zscore_fit;
    # multi-wild-type variable-length homologs (rhomax, hydro) preserve signal under l2_norm/raw.
    return "l2_norm" if ds in ("rhomax", "hydro") else "zscore_fit"

def load_split(ds, sp, model_key):
    """Load aligned split; falls back to trpb_sub50k cache if full trpb cache is absent."""
    df = pd.read_csv(safe_proc_path(f"{ds}__{sp}__cleaned.csv"))
    eval_subset = "full"
    if model_key == "one_hot":
        X = encode_onehot_csr(df["sequence"].tolist(), int(df["seq_len"].max()))
    else:
        uniq_sha = manifest["datasets"][ds]["unique_sha256"]
        cf = safe_cache_path(f"{model_key}__{ds}__{uniq_sha[:12]}.npz")
        if not cf.exists() and ds == "trpb" and "trpb_sub50k" in manifest["datasets"]:
            sub_sha = manifest["datasets"]["trpb_sub50k"]["unique_sha256"]
            cf_sub = safe_cache_path(f"{model_key}__trpb_sub50k__{sub_sha[:12]}.npz")
            if cf_sub.exists():
                cf = cf_sub
                eval_subset = "sub50k"
        if not cf.exists():
            return None
        d = np.load(cf)
        h2i = {h: i for i, h in enumerate(d["seq_hashes"].astype(str))}
        if eval_subset == "sub50k":
            keep = df["seq_hash"].isin(h2i).values
            df = df[keep].reset_index(drop=True)
        X = d["embeddings"][[h2i[h] for h in df["seq_hash"]]].astype(np.float32)
    fit, val, test = make_masks(df)
    if TASK_TYPES[ds] == "regression":
        ys, _ = scale_targets(df)
    else:
        ys = df["target"].astype(int).values
    return {"X": X, "ys": ys, "fit": fit, "val": val, "test": test,
            "df": df, "eval_subset": eval_subset}

def _fit_ridge(X_tr, y_tr, alpha: float):
    # Dense exact solver up to ~150M float32 entries (~600 MB); sparse lsqr for TrpB one-hot.
    if sparse.issparse(X_tr) and (X_tr.shape[0] * X_tr.shape[1] <= 150_000_000):
        X_tr = X_tr.toarray()
    solver = "lsqr" if sparse.issparse(X_tr) else "auto"
    tol = 1e-6 if solver == "lsqr" else 1e-5
    return Ridge(alpha=float(alpha), solver=solver, tol=tol, max_iter=10000).fit(X_tr, y_tr)

def _predict_ridge(clf, X_sub):
    if sparse.issparse(X_sub) and (X_sub.shape[0] * X_sub.shape[1] <= 150_000_000) and not clf.solver == "lsqr":
        X_sub = X_sub.toarray()
    return clf.predict(X_sub)

def fit_select_ridge(X, ys, fit, val):
    """Exact primal/dual eigh grid search for dense matrices; sparse lsqr for sparse CSR."""
    X_fit, y_fit = X[fit], ys[fit]
    X_val, y_val = X[val], ys[val]
    if sparse.issparse(X_fit):
        if X_fit.shape[0] * X_fit.shape[1] <= 150_000_000 and X_val.shape[0] * X_val.shape[1] <= 150_000_000 and X_fit.shape[1] <= 6000:
            X_fit = X_fit.toarray(); X_val = X_val.toarray()
        else:
            best_a, best_mse, best_clf = 10.0, np.inf, None
            for a in ALPHAS:
                clf = Ridge(alpha=float(a), solver="lsqr", tol=1e-6, max_iter=10000).fit(X_fit, y_fit)
                mse = float(mean_squared_error(y_val, clf.predict(X_val)))
                if mse < best_mse:
                    best_mse, best_a, best_clf = mse, float(a), clf
            return best_a, best_clf

    Xf = np.asarray(X_fit, dtype=np.float64); yf = np.asarray(y_fit, dtype=np.float64)
    mu_x = Xf.mean(0); mu_y = float(yf.mean())
    Xc = Xf - mu_x; yc = yf - mu_y
    Xv_c = np.asarray(X_val, dtype=np.float64) - mu_x
    n, d = Xc.shape
    if n >= d:
        eigvals, V = np.linalg.eigh(Xc.T @ Xc)
        eigvals = np.maximum(eigvals, 0.0)
        v_xty = V.T @ (Xc.T @ yc)
        Xv_V = Xv_c @ V
        W_V = v_xty[None, :] / (eigvals[None, :] + ALPHAS[:, None])
        vp_all = mu_y + W_V @ Xv_V.T
        idx = int(np.argmin(((vp_all - y_val[None, :]) ** 2).mean(1)))
        w = V @ W_V[idx]
    else:
        eigvals, U = np.linalg.eigh(Xc @ Xc.T)
        eigvals = np.maximum(eigvals, 0.0)
        ut_y = U.T @ yc
        Xv_K = Xv_c @ Xc.T
        D_U = ut_y[None, :] / (eigvals[None, :] + ALPHAS[:, None])
        vp_all = mu_y + (D_U @ U.T) @ Xv_K.T
        idx = int(np.argmin(((vp_all - y_val[None, :]) ** 2).mean(1)))
        w = Xc.T @ (U @ D_U[idx])
    b = float(mu_y - mu_x @ w)

    class _LinearModel:
        solver = "eigh"
        def __init__(self, coef, intercept):
            self.coef_ = coef; self.intercept_ = intercept
        def predict(self, X_in):
            # Chunked prediction avoids allocating a multi-GB float64 copy on TrpB test sets.
            if sparse.issparse(X_in):
                return np.asarray(X_in @ self.coef_, dtype=np.float64).ravel() + self.intercept_
            n_rows = X_in.shape[0]
            out = np.empty(n_rows, dtype=np.float64)
            for i in range(0, n_rows, 32768):
                out[i:i+32768] = np.asarray(X_in[i:i+32768], dtype=np.float64) @ self.coef_ + self.intercept_
            return out

    return float(ALPHAS[idx]), _LinearModel(w, b)

print("[OK] helpers ready.")''')

code('''# Cell 3a — Regression Ridge probing + scaling ablation (raw / l2_norm / zscore_fit) [CPU, ~0 CU]
reg_splits = [(ds, sp) for ds, sp in ACTIVE_SPLITS if TASK_TYPES[ds] == "regression"]
ablation_rows, probe_rows, alpha_map = [], [], {}

for ds, sp in reg_splits:
    for m in MODELS:
        D = load_split(ds, sp, m)
        if D is None:
            print(f"[SKIP] {ds}/{sp}/{m}: no cache"); continue
        X_raw, ys, fit, val, test = D["X"], D["ys"], D["fit"], D["val"], D["test"]
        eval_sub = D["eval_subset"]; del D
        pref_scale = primary_scaling(ds, m)
        scalings = ["raw", "l2_norm"] if m == "one_hot" else ["raw", "l2_norm", "zscore_fit"]

        for scaling in scalings:
            X_use = apply_scaling(X_raw, fit, scaling)
            a, clf = fit_select_ridge(X_use, ys, fit, val)
            met = regression_metrics(ys[test], clf.predict(X_use[test]))
            row = {"dataset": ds, "split": sp, "model": m, "eval_subset": eval_sub,
                   "feature_scaling": scaling, "val_criterion": "val_mse",
                   "alpha": a, "n_fit": int(fit.sum()), "n_test": int(test.sum()), **met}
            ablation_rows.append(row)
            if scaling == pref_scale:
                alpha_map[(ds, sp, m)] = a
                probe_rows.append(row)
                print(f"[REG {ds:7s}/{sp:13s}] {m:9s} ({scaling:10s}) "
                      f"a={a:7.2g} rho={met['spearman']:.4f} ndcg={met['ndcg']:.4f} "
                      f"mse={met['mse']:.4f}{' (const)' if met['pred_constant'] else ''}")
            if scaling != "raw":
                del X_use
        del X_raw; gc.collect()

ablation_df = pd.DataFrame(ablation_rows)
probe_df = pd.DataFrame(probe_rows)
ablation_df.to_csv(safe_tab_path("ridge_probing_ablation.csv"), index=False)
probe_df.to_csv(safe_tab_path("ridge_probing_metrics.csv"), index=False)
print(f"\\n[SAVED] ridge_probing_metrics.csv ({len(probe_df)}) & ridge_probing_ablation.csv ({len(ablation_df)})")
probe_df''')

code('''# Cell 3b — NucB 4-bin ordinal classification & high-activity detection [CPU, ~0 CU]
nucb_splits = [(ds, sp) for ds, sp in ACTIVE_SPLITS if TASK_TYPES[ds] == "classification"]
cls_rows = []
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

for ds, sp in nucb_splits:
    for m in MODELS:
        D = load_split(ds, sp, m)
        if D is None:
            print(f"[SKIP CLS] {ds}/{sp}/{m}: no cache"); continue
        X_raw, y_int, fit, val, test = D["X"], D["ys"], D["fit"], D["val"], D["test"]
        scalings = ["raw", "l2_norm"] if m == "one_hot" else ["raw", "l2_norm", "zscore_fit"]
        for scaling in scalings:
            X_use = apply_scaling(X_raw, fit, scaling)
            # 1) Ordinal Ridge -> Spearman & NDCG
            a, rclf = fit_select_ridge(X_use, y_int.astype(float), fit, val)
            m_ord = regression_metrics(y_int[test].astype(float), rclf.predict(X_use[test]))
            # 2) Multi-class LogisticRegression (tune C on val ordinal/macro-F1; bin=3 unseen in fit)
            best_c, best_f1, best_lclf = 1.0, -1.0, None
            for C in C_GRID:
                lc = LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(X_use[fit], y_int[fit])
                f1 = f1_score(y_int[val], lc.predict(X_use[val]), labels=[0,1,2,3], average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_c, best_lclf = f1, C, lc
            pred_cls = best_lclf.predict(X_use[test])
            # 3) Binary high-activity detector (bin >= 2), tuned on val AUROC
            y_hi_fit = (y_int[fit] >= 2).astype(int)
            y_hi_val = (y_int[val] >= 2).astype(int)
            y_hi_test = (y_int[test] >= 2).astype(int)
            best_bc, best_auc, best_bclf = 1.0, -1.0, None
            for C in C_GRID:
                bc = LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(X_use[fit], y_hi_fit)
                auc = roc_auc_score(y_hi_val, bc.predict_proba(X_use[val])[:, 1])
                if auc > best_auc:
                    best_auc, best_bc, best_bclf = auc, C, bc
            prob_hi = best_bclf.predict_proba(X_use[test])[:, 1]
            row = {
                "dataset": ds, "split": sp, "model": m, "feature_scaling": scaling,
                "n_fit": int(fit.sum()), "n_test": int(test.sum()),
                "accuracy_4bin": float(accuracy_score(y_int[test], pred_cls)),
                "macro_f1_4bin": float(f1_score(y_int[test], pred_cls, labels=[0,1,2,3], average="macro", zero_division=0)),
                "auroc_bin_ge2": float(roc_auc_score(y_hi_test, prob_hi)),
                "auprc_bin_ge2": float(average_precision_score(y_hi_test, prob_hi)),
                "ordinal_spearman": m_ord["spearman"], "ordinal_ndcg": m_ord["ndcg"],
                "ridge_alpha": a, "C_multi": best_c, "C_bin2": best_bc,
            }
            cls_rows.append(row)
            if scaling == primary_scaling(ds, m):
                print(f"[CLS {ds}/{sp}] {m:9s} ({scaling:10s}) acc={row['accuracy_4bin']:.4f} "
                      f"F1={row['macro_f1_4bin']:.4f} auroc(>=2)={row['auroc_bin_ge2']:.4f} "
                      f"auprc={row['auprc_bin_ge2']:.4f} ord_rho={row['ordinal_spearman']:.4f}")
            if scaling != "raw":
                del X_use
        del D, X_raw; gc.collect()

cls_df = pd.DataFrame(cls_rows)
if not cls_df.empty:
    cls_df.to_csv(safe_tab_path("nucb_classification_metrics.csv"), index=False)
    print(f"\\n[SAVED] nucb_classification_metrics.csv ({len(cls_df)} rows)")
cls_df''')

code('''# Cell 4 — Data efficiency across regression splits (3 seeds x 6 fractions, no leakage) [CPU, ~0 CU]
eff_rows = []
for ds, sp in reg_splits:
    for m in MODELS:
        D = load_split(ds, sp, m)
        if D is None:
            continue
        X_raw, ys, fit, test = D["X"], D["ys"], D["fit"], D["test"]
        eval_sub = D["eval_subset"]; del D
        scaling = primary_scaling(ds, m)
        a = alpha_map.get((ds, sp, m), 10.0)
        fit_idx = np.where(fit)[0]; test_idx = np.where(test)[0]
        if scaling == "l2_norm":
            X_pre = apply_scaling(X_raw, fit, "l2_norm")
            del X_raw; gc.collect()
        else:
            X_pre = X_raw
        for frac in FRACTIONS:
            n = min(max(5, int(round(frac * len(fit_idx)))), len(fit_idx))
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                sub = rng.choice(fit_idx, size=n, replace=False)
                if scaling == "zscore_fit":
                    X_sub = X_pre[sub]
                    mu = X_sub.mean(0); sd = X_sub.std(0); sd[sd < 1e-8] = 1.0
                    X_tr = ((X_sub - mu) / sd).astype(np.float32)
                    clf = _fit_ridge(X_tr, ys[sub], a)
                    # Fold z-score into weights so X_pre never needs to be duplicated!
                    w_raw = np.asarray(clf.coef_, dtype=np.float64) / sd
                    b_raw = float(clf.intercept_) - float(np.sum((mu / sd) * clf.coef_))
                    pred = np.empty(len(test_idx), dtype=np.float64)
                    for i in range(0, len(test_idx), 32768):
                        blk = test_idx[i:i+32768]
                        pred[i:i+32768] = np.asarray(X_pre[blk], dtype=np.float64) @ w_raw + b_raw
                else:
                    clf = _fit_ridge(X_pre[sub], ys[sub], a)
                    pred = clf.predict(X_pre[test_idx])
                met = regression_metrics(ys[test_idx], pred)
                eff_rows.append({"dataset": ds, "split": sp, "model": m,
                                 "eval_subset": eval_sub, "feature_scaling": scaling,
                                 "fraction": frac, "n_train": int(n), "seed": seed, "alpha": a, **met})
        del X_pre; gc.collect()

eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv(safe_tab_path("data_efficiency_per_seed.csv"), index=False)
agg = (eff_df.groupby(["dataset", "split", "model", "eval_subset", "fraction", "n_train"])
       .agg(spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
            ndcg_mean=("ndcg", "mean"), ndcg_std=("ndcg", "std"),
            mse_mean=("mse", "mean"), n_seeds=("seed", "nunique")).reset_index())
agg.to_csv(safe_tab_path("data_efficiency_summary.csv"), index=False)
print(f"[SAVED] data_efficiency_per_seed.csv ({len(eff_df)}) & summary ({len(agg)})")
agg.head(12)''')

code('''# Cell 5 — 5-member deep ensemble UQ (mini-batch safe for large W2 splits) [CPU, ~0 CU]
class MLPProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_member(Xtr, ytr, Xval, yval, in_dim, seed, max_epochs=250, patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPProbe(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xval, dtype=torch.float32); yv = torch.tensor(yval, dtype=torch.float32)
    use_minibatch = len(Xt) > 2048
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True) if use_minibatch else None
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        if dl is None:
            opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward(); opt.step()
        else:
            for bx, by in dl:
                opt.zero_grad(); loss = lossf(model(bx), by); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xv), yv).item()
        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model

def predict_batched(model, X_arr, batch_size=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X_arr), batch_size):
            out.append(model(torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32)).numpy())
    return np.concatenate(out)

uq_rows, rej_rows = [], []
for ds, sp in reg_splits:
    for m in UQ_MODELS:
        D = load_split(ds, sp, m)
        if D is None:
            print(f"[SKIP UQ] {ds}/{sp}/{m}: no cache"); continue
        X_raw, ys, fit, val, test = D["X"], D["ys"], D["fit"], D["val"], D["test"]
        eval_sub = D["eval_subset"]; del D
        if sparse.issparse(X_raw):
            if X_raw.shape[0] * X_raw.shape[1] > 150_000_000:
                print(f"[SKIP UQ] {ds}/{sp}/{m}: sparse matrix too large to densify safely")
                del X_raw; gc.collect()
                continue
            X_raw = X_raw.toarray()
        Xs = apply_scaling(X_raw, fit, "zscore_fit")
        del X_raw; gc.collect()
        yte = ys[test]
        preds = []
        for seed in range(N_ENSEMBLE):
            model = train_member(Xs[fit], ys[fit], Xs[val], ys[val], Xs.shape[1], seed)
            preds.append(predict_batched(model, Xs[test]))
        P = np.array(preds)
        mean_pred = P.mean(0); epi_std = P.std(0)
        met = regression_metrics(yte, mean_pred)
        err = np.abs(yte - mean_pred)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            std_err_rho = float(spearmanr(epi_std, err).correlation) if epi_std.std() >= CONST_TOL else float("nan")
            std_err_r = float(pearsonr(epi_std, err).statistic) if epi_std.std() >= CONST_TOL else float("nan")
            member_rhos = [spearmanr(yte, p).correlation for p in P]
        k10 = max(1, int(0.10 * len(yte)))
        ndcg10 = float(ndcg_score((yte - yte.min())[None, :], mean_pred[None, :], k=k10))
        uq_rows.append({"dataset": ds, "split": sp, "model": m, "eval_subset": eval_sub,
                        "n_ensemble": N_ENSEMBLE,
                        "ensemble_spearman": met["spearman"], "ensemble_ndcg": met["ndcg"],
                        "ensemble_mse": met["mse"], "ndcg_at_10pct": ndcg10,
                        "std_abserr_spearman": std_err_rho, "std_abserr_pearson": std_err_r,
                        "mean_member_spearman": float(np.nanmean(member_rhos)),
                        "pred_constant": met["pred_constant"]})
        order = np.argsort(epi_std)
        for keep in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            nk = max(1, int(len(order) * keep)); kept = order[:nk]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rk = spearmanr(yte[kept], mean_pred[kept]).correlation
            rej_rows.append({"dataset": ds, "split": sp, "model": m, "keep_fraction": keep,
                             "n_keep": int(nk),
                             "rmse": float(np.sqrt(mean_squared_error(yte[kept], mean_pred[kept]))),
                             "spearman": float(rk)})
        print(f"[UQ {ds:7s}/{sp:13s}/{m:9s}] ens rho={met['spearman']:.4f} "
              f"ndcg@10%={ndcg10:.4f} std~|err| rho={std_err_rho:.4f}")
        del Xs; gc.collect()

uq_df = pd.DataFrame(uq_rows); rej_df = pd.DataFrame(rej_rows)
uq_df.to_csv(safe_tab_path("uq_ensemble_metrics.csv"), index=False)
rej_df.to_csv(safe_tab_path("uq_error_coverage_curve.csv"), index=False)
print(f"\\n[SAVED] uq_ensemble_metrics.csv ({len(uq_df)}) & uq_error_coverage_curve.csv ({len(rej_df)})")
uq_df''')

code('''# Cell 6 — Multi-panel publication figures (auto-grid for W1+W2) [CPU, ~0 CU]
import math

def save_fig(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    safe_fig_path(name).write_bytes(buf.getvalue())
    plt.close(fig)
    print(f"[FIG] {name}")

def make_grid(n_panels, max_cols=4, panel_w=4.4, panel_h=3.6):
    ncols = min(max_cols, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]

# Fig 1 — Regression probing across model scales
if not probe_df.empty:
    fig, axes = make_grid(len(reg_splits), max_cols=4)
    for ax, (ds, sp) in zip(axes, reg_splits):
        sub = probe_df[(probe_df.dataset == ds) & (probe_df.split == sp)]
        if sub.empty:
            ax.set_visible(False); continue
        order = [m for m in MODELS if m in set(sub.model)]
        vals = [sub[sub.model == m].spearman.iloc[0] for m in order]
        ax.bar(range(len(order)), [0 if np.isnan(v) else v for v in vals], color="steelblue")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{ds} / {sp}", fontsize=9.5); ax.set_ylabel("Spearman ρ")
        ax.axhline(0, color="gray", lw=0.6)
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.text(i, 0.02, "NaN\\n(const)", ha="center", fontsize=7, color="crimson")
    fig.suptitle("FLIP2 Frozen-Embedding Ridge Probing (val-MSE α)", fontsize=12)
    fig.tight_layout(); save_fig(fig, "fig1_model_scale_probing.png")

# Fig 2 — Data efficiency curves
if not eff_df.empty:
    fig, axes = make_grid(len(reg_splits), max_cols=4)
    for ax, (ds, sp) in zip(axes, reg_splits):
        sub = agg[(agg.dataset == ds) & (agg.split == sp)]
        if sub.empty:
            ax.set_visible(False); continue
        plotted_any = False
        for m in [x for x in MODELS if x in set(sub.model)]:
            ms = sub[sub.model == m].sort_values("fraction")
            y = ms.spearman_mean.values
            if np.isfinite(y).any():
                ax.plot(ms.fraction, y, marker="o", ms=4, label=m)
                ax.fill_between(ms.fraction, y - ms.spearman_std.values,
                                y + ms.spearman_std.values, alpha=0.14)
                plotted_any = True
            else:
                ax.plot([], [], marker="o", label=f"{m} (ρ=NaN)")
        if not plotted_any:
            ax.text(0.5, 0.5, "constant predictions\\nSpearman = NaN",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="crimson")
            ax.set_ylim(-0.05, 1.0)
        ax.set_title(f"{ds} / {sp}", fontsize=9.5)
        ax.set_xlabel("training fraction"); ax.set_ylabel("Spearman ρ (mean±std)")
        ax.legend(fontsize=6.5)
    fig.suptitle("Data efficiency on official FLIP2 distribution-shift splits", fontsize=12)
    fig.tight_layout(); save_fig(fig, "fig2_data_efficiency.png")

# Fig 3 — UQ error-coverage
if not rej_df.empty:
    combos = rej_df[["dataset", "split", "model"]].drop_duplicates().values.tolist()
    fig, axes = make_grid(len(combos), max_cols=4, panel_w=4.0, panel_h=3.4)
    for ax, (ds, sp, m) in zip(axes, combos):
        sub = rej_df[(rej_df.dataset == ds) & (rej_df.split == sp) & (rej_df.model == m)].sort_values("keep_fraction")
        ax.plot(sub.keep_fraction, sub.rmse, marker="s", color="purple")
        ax.set_title(f"{ds}/{sp}\\n{m}", fontsize=9)
        ax.set_xlabel("retained fraction"); ax.set_ylabel("RMSE")
        ax.invert_xaxis()
    fig.suptitle("UQ selective prediction (5x deep ensemble)", fontsize=12)
    fig.tight_layout(); save_fig(fig, "fig3_uq_error_coverage.png")

# Fig 4 — NucB classification summary
if not cls_df.empty:
    sub = cls_df[cls_df.apply(lambda r: r.feature_scaling == primary_scaling(r.dataset, r.model), axis=1)]
    order = [m for m in MODELS if m in set(sub.model)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(axes,
                              ["auroc_bin_ge2", "macro_f1_4bin", "ordinal_spearman"],
                              ["High-activity AUROC (bin>=2)", "4-bin Macro-F1", "Ordinal Spearman ρ"]):
        vals = [sub[sub.model == m][col].iloc[0] for m in order]
        ax.bar(range(len(order)), vals, color="teal")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("NucB / two_to_many (4 ordinal activity bins; bin 3 unseen in fit)", fontsize=11.5)
    fig.tight_layout(); save_fig(fig, "fig4_nucb_classification.png")

print("[DONE] figures written to", FIG_DIR)''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TEST = "--test" in sys.argv
if TEST:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["source"] = (c["source"]
                           .replace('MODELS = ["one_hot"] + PLM_MODELS', 'MODELS = ["one_hot"]')
                           .replace('UQ_MODELS = ["esm2_650m", "esm2_3b"]', 'UQ_MODELS = ["one_hot"]')
                           .replace('RUN_SCOPE = "all"', 'RUN_SCOPE = "w2"')
                           .replace('("trpb",  "one_to_many"), ("trpb", "two_to_many"), ("trpb", "by_position"),',
                                    '("trpb",  "one_to_many"),')
                           .replace('max_epochs=250, patience=12', 'max_epochs=15, patience=5'))
    out = ROOT / "nb03_test.ipynb"
else:
    out = ROOT / "notebook_03_probing_data_efficiency_uq.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells){' [TEST CONFIG]' if TEST else ''}")