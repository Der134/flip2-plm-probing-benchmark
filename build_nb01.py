# -*- coding: utf-8 -*-
"""Builder for notebook_01_data_prep_one_hot_baseline.ipynb (W1 + W2 unified)."""
import json
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


md("""# 01 — FLIP2 Data Preparation, Official Splits & One-Hot Baselines (W1 + W2)

**Resource tier: CPU runtime (~0 CU).** Supports both W1 pilot (`amylase`, `rhomax`) and
W2 expansion (`ired`, `hydro`, `nucb`, `trpb` — 6 datasets, 13 official splits total).

**Protocol (matches FLIP2 official `baselines/linear_models.py`):**
- `fit`  = `set == "train"` AND `validation == False`
- `val`  = `set == "train"` AND `validation == True`  (hyperparameter selection only)
- `test` = `set == "test"`  (touch once, final config)
- Regression targets: `StandardScaler` fit on **all** `set == "train"` targets; report
  Spearman ρ, MSE, R², and **NDCG = `ndcg_score(y_test - y_test.min(), pred)`**.
- `nucb` (4 ordinal activity bins `{0,1,2,3}`; note `fit` has `{0,1,2}` only, `3` emerges in
  `val`/`test`!): report both **ordinal ranking** (`spearman`, `ndcg`) and **classification**
  (`accuracy`, `macro_f1`, `auroc_bin2`, `auprc_bin2` for high-activity `bin >= 2`).
- One-hot uses **sparse CSR matrices** (`scipy.sparse.csr_matrix`) so even `trpb`
  (228,298 × 7,780) fits in **< 1 GB RAM** instead of 7.1 GB dense.

**No synthetic fallback.** All downloads are restricted to `flip.protein.properties`.
**Citations required:** Didi et al. (ICML 2026) *and* Dallago et al. (NeurIPS 2021).
""")

code('''# Cell 1 — Setup, workspace verification & split registry [CPU, ~0 CU]
import os, sys, json, hashlib, re, warnings
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Singular matrix.*")
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    BASE_DIR = Path("/content/drive/MyDrive/flip2_benchmark")
else:
    BASE_DIR = Path("flip2_benchmark").resolve()

SENTINEL_FILE = BASE_DIR / ".workspace_id"
EXPECTED_ID = "flip2-shared-prod-2026"
if IN_COLAB:
    if not SENTINEL_FILE.exists():
        raise RuntimeError(f"Workspace sentinel missing: {SENTINEL_FILE}")
    actual = SENTINEL_FILE.read_text(encoding="utf-8").strip()
    assert actual == EXPECTED_ID, f"Workspace ID mismatch: {actual!r} != {EXPECTED_ID!r}"
    print("[OK] Shared workspace verified.")
    print(f"  logical : {BASE_DIR}")
    print(f"  physical: {os.path.realpath(BASE_DIR)}")

RAW_DIR  = (BASE_DIR / "data" / "raw").resolve()
PROC_DIR = (BASE_DIR / "data" / "processed").resolve()
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---- Scope configuration: "w1" | "w2" | "all" ----
RUN_SCOPE = "all"

W1_SPLITS = [
    ("amylase", "one_to_many"),
    ("amylase", "close_to_far"),
    ("rhomax",  "by_wild_type"),
]
W2_SPLITS = [
    ("ired",  "two_to_many"),
    ("hydro", "three_to_many"),
    ("hydro", "low_to_high"),
    ("hydro", "to_P06241"),
    ("hydro", "to_P0A9X9"),
    ("hydro", "to_P01053"),
    ("nucb",  "two_to_many"),
    ("trpb",  "one_to_many"),
    ("trpb",  "two_to_many"),
    ("trpb",  "by_position"),
]
TASK_TYPES = {
    "amylase": "regression",
    "rhomax":  "regression",
    "ired":    "regression",
    "hydro":   "regression",
    "nucb":    "classification",
    "trpb":    "regression",
}

if RUN_SCOPE == "w1":
    ACTIVE_SPLITS = W1_SPLITS
elif RUN_SCOPE == "w2":
    ACTIVE_SPLITS = W2_SPLITS
elif RUN_SCOPE == "all":
    ACTIVE_SPLITS = W1_SPLITS + W2_SPLITS
else:
    raise ValueError(f"Unknown RUN_SCOPE: {RUN_SCOPE}")

ALLOWED_HOST = "flip.protein.properties"
ALLOWED_SCHEMES = ("https", "http")
URL_PATH_TEMPLATE = "/assets/splits/{dataset}/{split}.csv.gz"
SEED = 42
np.random.seed(SEED)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def validate_token(tok: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", tok):
        raise ValueError(f"invalid dataset/split token: {tok!r}")
    return tok

def _confine(base: Path, p: Path) -> Path:
    p = p.resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes base dir: {p}")
    return p

def safe_raw_path(dataset: str, split: str) -> Path:
    validate_token(dataset); validate_token(split)
    return _confine(RAW_DIR, RAW_DIR / f"{dataset}__{split}.csv.gz")

def safe_proc_path(filename: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.\\-]+", filename) or ".." in filename:
        raise ValueError(f"invalid processed filename: {filename!r}")
    return _confine(PROC_DIR, PROC_DIR / filename)

print(f"[CFG] RUN_SCOPE={RUN_SCOPE} | {len(ACTIVE_SPLITS)} splits")''')

code('''# Cell 2 — Download real official split files (fixed host, https -> http fallback) [CPU, ~0 CU]
import urllib.request

class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urlparse(newurl).hostname
        if host != ALLOWED_HOST:
            raise RuntimeError(f"blocked cross-host redirect to {host!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

_OPENER = urllib.request.build_opener(_SameHostRedirect)

def build_url(scheme: str, dataset: str, split: str) -> str:
    validate_token(dataset); validate_token(split)
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"scheme not allowed: {scheme!r}")
    url = f"{scheme}://{ALLOWED_HOST}{URL_PATH_TEMPLATE.format(dataset=dataset, split=split)}"
    parsed = urlparse(url)
    if parsed.hostname != ALLOWED_HOST or parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL outside allowlist: {url}")
    return url

def download_split(dataset: str, split: str) -> dict:
    out = safe_raw_path(dataset, split)
    canonical_url = build_url("https", dataset, split)
    retrieved = datetime.now(timezone.utc).isoformat()
    if out.exists() and out.stat().st_size > 0:
        print(f"[SKIP] exists: {out.name}")
        return {"raw_file": out.name, "raw_sha256": sha256_file(out),
                "url": canonical_url, "retrieved_utc": retrieved, "source": "cached"}
    last_err = None
    for scheme in ALLOWED_SCHEMES:
        url = build_url(scheme, dataset, split)
        try:
            print(f"[GET ] {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "flip2-benchmark/1.0"})
            with _OPENER.open(req, timeout=90) as r:
                data = r.read()
            if len(data) == 0:
                raise IOError("empty response")
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, out)
            print(f"[OK  ] {out.name} ({len(data)} bytes, {scheme})")
            return {"raw_file": out.name, "raw_sha256": sha256_file(out),
                    "url": url, "retrieved_utc": retrieved, "source": scheme}
        except Exception as e:
            last_err = e
            print(f"[WARN] {scheme} failed: {e}")
    raise RuntimeError(f"Could not download {dataset}/{split}: {last_err}")

DOWNLOAD_META = {}
for ds, sp in ACTIVE_SPLITS:
    DOWNLOAD_META[(ds, sp)] = download_split(ds, sp)
print(f"\\n[DONE] {len(DOWNLOAD_META)} split files ready in {RAW_DIR}")''')

code('''# Cell 3 — Validate, clean & export per-split CSVs (column-order independent) [CPU, ~0 CU]
REQUIRED_COLS = ["sequence", "target", "set", "validation"]

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

SPLIT_META = {}
CLEANED = {}
for ds, sp in ACTIVE_SPLITS:
    meta = DOWNLOAD_META[(ds, sp)]
    cleaned_path = safe_proc_path(f"{ds}__{sp}__cleaned.csv")
    df = pd.read_csv(safe_raw_path(ds, sp))
    # Column order varies across official files (e.g., ired puts target last).
    assert set(REQUIRED_COLS).issubset(df.columns), f"{ds}/{sp}: missing cols in {list(df.columns)}"
    df = df[REQUIRED_COLS].copy()
    assert not df.isnull().any().any(), f"{ds}/{sp}: null values present"
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    bad = ~df["sequence"].str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
    assert not bad.any(), f"{ds}/{sp}: {int(bad.sum())} non-canonical sequences"
    df["validation"] = as_bool(df["validation"])
    assert set(df["set"].unique()).issubset({"train", "test", "validation"}), \\
        f"{ds}/{sp}: unexpected set labels {df['set'].unique()}"

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    df["dataset"] = ds
    df["split"] = sp
    df["seq_len"] = df["sequence"].str.len()
    df["seq_hash"] = df["sequence"].map(sha256_text)

    is_train = df["set"].eq("train")
    fit_mask = is_train & ~df["validation"]
    val_mask = (is_train & df["validation"]) | df["set"].eq("validation")
    test_mask = df["set"].eq("test")
    assert (fit_mask.sum() + val_mask.sum() + test_mask.sum()) == len(df), "partition mismatch"

    cols = ["row_id", "dataset", "split", "sequence", "target", "set",
            "validation", "seq_len", "seq_hash"]
    df[cols].to_csv(cleaned_path, index=False)
    CLEANED[(ds, sp)] = df

    tgt = df["target"].astype(float)
    SPLIT_META[(ds, sp)] = {
        **meta,
        "task_type": TASK_TYPES[ds],
        "n_rows": int(len(df)),
        "n_fit": int(fit_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_unique_sequences": int(df["sequence"].nunique()),
        "seq_len_min": int(df["seq_len"].min()),
        "seq_len_max": int(df["seq_len"].max()),
        "target_mean": float(tgt.mean()),
        "target_std": float(tgt.std()),
        "target_min": float(tgt.min()),
        "target_max": float(tgt.max()),
    }
    m = SPLIT_META[(ds, sp)]
    print(f"[{ds:7s}/{sp:13s}] ({m['task_type'][:3]}) rows={m['n_rows']:6d} "
          f"fit={m['n_fit']:5d} val={m['n_val']:5d} test={m['n_test']:6d} "
          f"uniq={m['n_unique_sequences']:6d} len=[{m['seq_len_min']},{m['seq_len_max']}]")''')

code('''# Cell 4 — Per-dataset unique registries, TrpB sub50k registry & manifests [CPU, ~0 CU]
DATASETS = sorted({ds for ds, _ in ACTIVE_SPLITS}, key=lambda d: [x for x, _ in ACTIVE_SPLITS].index(d))
dataset_meta = {}
for ds in DATASETS:
    ds_splits = [(d, s) for (d, s) in ACTIVE_SPLITS if d == ds]
    concat = pd.concat([CLEANED[k] for k in ds_splits], ignore_index=True)
    uniq = (concat.drop_duplicates(subset=["seq_hash"])
            [["dataset", "sequence", "seq_len", "seq_hash"]]
            .sort_values("seq_hash").reset_index(drop=True))
    uniq_sha = sha256_text("".join(uniq["seq_hash"].tolist()))
    uniq.to_csv(safe_proc_path(f"unique_sequences__{ds}.csv"), index=False)
    dataset_meta[ds] = {
        "task_type": TASK_TYPES[ds],
        "n_unique": int(len(uniq)),
        "unique_sha256": uniq_sha,
        "splits": {sp: SPLIT_META[(d, sp)] for (d, sp) in ds_splits},
    }
    print(f"[{ds:7s}] unique={len(uniq):6d} registry_sha={uniq_sha[:12]} -> unique_sequences__{ds}.csv")

# Optional stratified 50k subset for TrpB 3B (keeps 100% of one_to_many train + decile-stratified sample)
if "trpb" in dataset_meta:
    trpb_ref = CLEANED[("trpb", "one_to_many")][["sequence", "seq_len", "seq_hash", "target", "set"]].copy()
    o2m_train_hashes = set(trpb_ref.loc[trpb_ref["set"] == "train", "seq_hash"])
    must_keep = trpb_ref[trpb_ref["seq_hash"].isin(o2m_train_hashes)].copy()
    rest = trpb_ref[~trpb_ref["seq_hash"].isin(o2m_train_hashes)].copy()
    n_need = 50000 - len(must_keep)
    rest["decile"] = pd.qcut(rest["target"], q=10, labels=False, duplicates="drop")
    rng = np.random.RandomState(SEED)
    sampled_idx = []
    for dec, grp in rest.groupby("decile"):
        k = int(round(n_need * len(grp) / len(rest)))
        sampled_idx.extend(rng.choice(grp.index.values, size=min(k, len(grp)), replace=False).tolist())
    # Top up or trim to exact 50,000
    if len(sampled_idx) < n_need:
        avail = np.setdiff1d(rest.index.values, np.array(sampled_idx))
        sampled_idx.extend(rng.choice(avail, size=n_need - len(sampled_idx), replace=False).tolist())
    elif len(sampled_idx) > n_need:
        sampled_idx = rng.choice(np.array(sampled_idx), size=n_need, replace=False).tolist()
    sub50k = pd.concat([must_keep, rest.loc[sampled_idx]], ignore_index=True)
    sub50k = (sub50k.assign(dataset="trpb_sub50k")[["dataset", "sequence", "seq_len", "seq_hash"]]
              .sort_values("seq_hash").reset_index(drop=True))
    sub_sha = sha256_text("".join(sub50k["seq_hash"].tolist()))
    sub50k.to_csv(safe_proc_path("unique_sequences__trpb_sub50k.csv"), index=False)
    dataset_meta["trpb_sub50k"] = {
        "task_type": "regression",
        "parent_dataset": "trpb",
        "n_unique": int(len(sub50k)),
        "unique_sha256": sub_sha,
        "sampling": "all one_to_many train (380) + target-decile stratified sample to 50,000",
    }
    print(f"[trpb_sub50k] unique={len(sub50k):6d} registry_sha={sub_sha[:12]} -> unique_sequences__trpb_sub50k.csv")

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "run_scope": RUN_SCOPE,
    "url_template": f"https://{ALLOWED_HOST}{URL_PATH_TEMPLATE}",
    "protocol": {
        "fit": "set=='train' & ~validation",
        "val": "set=='train' & validation (hyperparameter selection only)",
        "test": "set=='test' (final evaluation only)",
        "target_scaling": "StandardScaler fit on all set=='train' targets for regression; raw integer bins for nucb",
        "ndcg": "ndcg_score(y_test - y_test.min(), pred) (official)",
    },
    "canonical_aas": "ACDEFGHIKLMNPQRSTVWY",
    "datasets": dataset_meta,
}
safe_proc_path("benchmark_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# Keep W1 pilot_manifest.json & unique_sequences.csv backward-compatible if W1 datasets are present
if "amylase" in dataset_meta and "rhomax" in dataset_meta:
    w1_uniq = pd.concat([
        pd.read_csv(safe_proc_path("unique_sequences__amylase.csv")),
        pd.read_csv(safe_proc_path("unique_sequences__rhomax.csv")),
    ], ignore_index=True)
    w1_uniq.to_csv(safe_proc_path("unique_sequences.csv"), index=False)
    w1_man = {**manifest, "datasets": {k: dataset_meta[k] for k in ("amylase", "rhomax")},
              "total_unique_sequences": int(len(w1_uniq))}
    safe_proc_path("pilot_manifest.json").write_text(
        json.dumps(w1_man, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\\n[SAVED] benchmark_manifest.json ({len(dataset_meta)} dataset entries)")''')

code('''# Cell 5 — Sparse CSR one-hot encoding & evaluation utilities [CPU, ~0 CU]
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import (mean_squared_error, r2_score, ndcg_score,
                             accuracy_score, f1_score, roc_auc_score, average_precision_score)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

AAINDEX_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AAINDEX_ALPHABET)}
CONST_TOL = 1e-4

def fit_ridge_auto(X_tr, y_tr, alpha: float):
    # Use exact dense solver when <= 150M float32 entries (~600 MB); sparse lsqr for TrpB.
    if X_tr.shape[0] * X_tr.shape[1] <= 150_000_000:
        Xd = X_tr.toarray() if sparse.issparse(X_tr) else X_tr
        return Ridge(alpha=alpha, solver="auto", tol=1e-5, max_iter=1000000).fit(Xd, y_tr)
    return Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=10000).fit(X_tr, y_tr)

def encode_onehot_csr(sequences, max_len: int) -> sparse.csr_matrix:
    """Memory-efficient sparse one-hot (trpb 228k x 7780 is ~710 MB vs 7.1 GB dense)."""
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
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, max_len * 20), dtype=np.float32)

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

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
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

print("[OK] sparse CSR one-hot & metrics ready.")''')

code('''# Cell 6 — Official one-hot baselines across all active splits (regression + NucB classification) [CPU, ~0 CU]
import gc
reg_rows, cls_rows = [], []

for ds, sp in ACTIVE_SPLITS:
    df = CLEANED[(ds, sp)]
    fit, val, test = make_masks(df)
    X = encode_onehot_csr(df["sequence"].tolist(), int(df["seq_len"].max()))

    if TASK_TYPES[ds] == "regression":
        ys, _ = scale_targets(df)
        clf = fit_ridge_auto(X[fit], ys[fit], alpha=10.0)
        X_te = X[test].toarray() if not sparse.issparse(clf.coef_) and X.shape[0]*X.shape[1] <= 150_000_000 else X[test]
        m = regression_metrics(ys[test], clf.predict(X_te))
        reg_rows.append({"dataset": ds, "split": sp, "model": "one_hot_ridge_official",
                         "alpha": 10.0, "n_fit": int(fit.sum()), "n_test": int(test.sum()), **m})
        print(f"[REG {ds:7s}/{sp:13s}] one_hot(a=10) rho={m['spearman']:.4f} "
              f"ndcg={m['ndcg']:.4f} mse={m['mse']:.4f}{' (const)' if m['pred_constant'] else ''}")
    else:
        # NucB: 4 ordinal bins {0,1,2,3}; fit contains {0,1,2} only, bin 3 appears in val/test.
        y_int = df["target"].astype(int).values
        # 1) Ordinal Ridge(alpha=10) -> Spearman & NDCG
        rclf = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=5000).fit(X[fit], y_int[fit].astype(float))
        pred_cont = rclf.predict(X[test])
        m_ord = regression_metrics(y_int[test].astype(float), pred_cont)
        # 2) Multi-class LogisticRegression on observed fit classes + binary active (bin>=2) AUROC/AUPRC
        lclf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X[fit], y_int[fit])
        pred_cls = lclf.predict(X[test])
        bclf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X[fit], (y_int[fit] >= 2).astype(int))
        prob_hi = bclf.predict_proba(X[test])[:, 1]
        y_hi_test = (y_int[test] >= 2).astype(int)
        cls_row = {
            "dataset": ds, "split": sp, "model": "one_hot_linear",
            "n_fit": int(fit.sum()), "n_test": int(test.sum()),
            "accuracy_4bin": float(accuracy_score(y_int[test], pred_cls)),
            "macro_f1_4bin": float(f1_score(y_int[test], pred_cls, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            "auroc_bin_ge2": float(roc_auc_score(y_hi_test, prob_hi)),
            "auprc_bin_ge2": float(average_precision_score(y_hi_test, prob_hi)),
            "ordinal_spearman": m_ord["spearman"],
            "ordinal_ndcg": m_ord["ndcg"],
        }
        cls_rows.append(cls_row)
        print(f"[CLS {ds:7s}/{sp:13s}] one_hot acc={cls_row['accuracy_4bin']:.4f} "
              f"macroF1={cls_row['macro_f1_4bin']:.4f} auroc(>=2)={cls_row['auroc_bin_ge2']:.4f} "
              f"ord_rho={cls_row['ordinal_spearman']:.4f} ndcg={cls_row['ordinal_ndcg']:.4f}")
    del X; gc.collect()

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(safe_proc_path("onehot_baseline_official_all.csv"), index=False)
if cls_rows:
    pd.DataFrame(cls_rows).to_csv(safe_proc_path("onehot_baseline_nucb_cls.csv"), index=False)
print("\\n[SAVED] onehot_baseline_official_all.csv" + (" & onehot_baseline_nucb_cls.csv" if cls_rows else ""))
reg_df''')

md("""## Next steps

1. Run `notebook_02` on **L4** for `ired`, `hydro`, `nucb` (~1.2 CU total) and on **G4** for `3b` (~1.2 CU),
   then `trpb` (L4 ≤650M ~12 CU; G4 `3b` full ~7.7 CU or `sub50k` ~1.7 CU).
2. Run `notebook_03` on **CPU** (0 CU) for probing + L2/z-score ablations + data efficiency + UQ.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebook_01_data_prep_one_hot_baseline.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")