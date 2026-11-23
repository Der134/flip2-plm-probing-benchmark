# -*- coding: utf-8 -*-
"""Builder for notebook_04_esm2_3b_lora_comparison.ipynb (W2 3B LoRA comparison arm)."""
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


md("""# 04 — ESM-2 3B LoRA Fine-Tuning Comparison Arm (FLIP2)

**Resource tier: G4 (96 GB, ~10.5 CU/h) or A100 (40/80 GB).** Compares LoRA fine-tuned `esm2_3b`
against frozen `esm2_3b` probing (`ridge` & `5xMLP` from Notebook 03) under the exact same FLIP2
official protocol (`fit = train & ~validation`, `val = train & validation`, `test` evaluated once,
`StandardScaler` fit on all `set == "train"` targets).

**Dual-account parallel execution (zero file conflicts):**
- Set `TASK_PRESET = "acct1_amy"` on Account 1 (`amylase`: `one_to_many`, `close_to_far`).
- Set `TASK_PRESET = "acct2_hydro"` on Account 2 (`hydro`: `three_to_many`, `to_P0A9X9`).
- Each `(dataset, split, seed)` writes atomically to `outputs/lora/runs/{ds}__{sp}__seed{seed}.json`
  and `outputs/lora/preds/{ds}__{sp}__seed{seed}.npz`. Completed runs are skipped on resume.
- Cell 7 merges all completed runs from Drive into `outputs/tables/lora_3b_per_seed.csv` and
  `outputs/tables/lora_3b_summary.csv` (paired with frozen `esm2_3b` baselines if present).
""")

code('''# Cell 1 — Workspace verification & GPU telemetry [GPU]
import os, sys, gc, io, json, time, re, math, warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/content/hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        raise RuntimeError(f"Sentinel check failed: {SENTINEL_FILE} not found.")
    actual = SENTINEL_FILE.read_text(encoding="utf-8").strip()
    assert actual == EXPECTED_ID, f"ID mismatch: {actual!r} != {EXPECTED_ID!r}"
    print("[OK] Shared workspace verified.")

RAW_DIR       = (BASE_DIR / "data" / "raw").resolve()
PROC_DIR      = (BASE_DIR / "data" / "processed").resolve()
OUT_DIR       = (BASE_DIR / "outputs").resolve()
TAB_DIR       = (OUT_DIR / "tables").resolve()
LORA_RUN_DIR  = (OUT_DIR / "lora" / "runs").resolve()
LORA_PRED_DIR = (OUT_DIR / "lora" / "preds").resolve()
for d in (TAB_DIR, LORA_RUN_DIR, LORA_PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    BF16_SUPPORTED = torch.cuda.is_bf16_supported()
else:
    GPU_NAME, GPU_VRAM_GB, BF16_SUPPORTED = "CPU", 0.0, False

if GPU_VRAM_GB >= 70:
    DETECTED_TIER = "G4_OR_A100_80G"
elif GPU_VRAM_GB >= 35:
    DETECTED_TIER = "A100_40G"
elif HAS_CUDA:
    DETECTED_TIER = "SMALL_GPU"
else:
    DETECTED_TIER = "CPU"

print(f"[HW] {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) | tier={DETECTED_TIER} | bf16={BF16_SUPPORTED}")''')

code('''# Cell 2 — Dependencies [GPU]
try:
    import transformers
    from tqdm.auto import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46.0", "tqdm"])
    import transformers
    from tqdm.auto import tqdm
print(f"transformers {transformers.__version__} | torch {torch.__version__}")''')

code('''# Cell 3 — Configuration & dual-account parallel task selector [GPU]
SMOKE_TEST = False              # True: tiny esm2_8m 2-epoch test on 64 train/32 val/32 test rows
ALLOW_SMALL_GPU = False         # Must remain False for 3B LoRA (needs >=35 GB VRAM: G4 or A100)
OVERWRITE_RUNS = False          # False: skip already-saved (dataset, split, seed) runs

# Parallel split presets:
#   "all"         : run all 4 comparison tasks sequentially
#   "acct1_amy"   : Account 1 -> amylase (one_to_many, close_to_far)
#   "acct2_hydro" : Account 2 -> hydro (three_to_many, to_P0A9X9)
#   "custom"      : use TASKS_OVERRIDE
TASK_PRESET = "all"
TASKS_OVERRIDE = []

SEEDS = [42, 43, 44]

# LoRA & optimization hyperparameters (fixed a priori, selected by val_mse)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("query", "key", "value")
LR_LORA = 2e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
CONST_TOL = 1e-4

PRESET_MAP = {
    "all": [("amylase", "one_to_many"), ("amylase", "close_to_far"),
            ("hydro", "three_to_many"), ("hydro", "to_P0A9X9")],
    "acct1_amy":   [("amylase", "one_to_many"), ("amylase", "close_to_far")],
    "acct2_hydro": [("hydro", "three_to_many"), ("hydro", "to_P0A9X9")],
}
TASKS = list(TASKS_OVERRIDE) if TASK_PRESET == "custom" else PRESET_MAP[TASK_PRESET]
assert TASKS, "TASKS cannot be empty"

MODEL_KEY = "esm2_8m" if SMOKE_TEST else "esm2_3b"
HF_ID = "facebook/esm2_t6_8M_UR50D" if SMOKE_TEST else "facebook/esm2_t36_3B_UR50D"

if not SMOKE_TEST:
    if not HAS_CUDA:
        raise RuntimeError("CUDA GPU required for ESM-2 3B LoRA.")
    if DETECTED_TIER == "SMALL_GPU" and not ALLOW_SMALL_GPU:
        raise RuntimeError(f"[Guard] {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) is <35 GB; use G4 (96 GB) or A100 (40/80 GB).")

# Hardware-tuned batching & checkpointing
USE_GRAD_CKPT = (not SMOKE_TEST) and (DETECTED_TIER == "A100_40G")
TRAIN_MAX_TOKENS = 2048 if SMOKE_TEST else (4096 if DETECTED_TIER == "A100_40G" else 8192)
TRAIN_MAX_SEQS   = 32
EVAL_MAX_TOKENS  = 8192 if SMOKE_TEST else (16384 if DETECTED_TIER == "A100_40G" else 32000)
EVAL_MAX_SEQS    = 256

print(f"[CFG] preset={TASK_PRESET} | tasks={TASKS} | seeds={SEEDS} | model={MODEL_KEY} ({HF_ID})")
print(f"[CFG] train_max_tokens={TRAIN_MAX_TOKENS} max_seqs={TRAIN_MAX_SEQS} grad_ckpt={USE_GRAD_CKPT}")''')

# Embed lora_core.py verbatim after its import header so notebook_04 is 100% self-contained
core_lines = (ROOT / "lora_core.py").read_text(encoding="utf-8").splitlines()
start_idx = next(i for i, ln in enumerate(core_lines) if ln.startswith("def confine_path"))
core_body = "\n".join(core_lines[start_idx:])
code(f"# Cell 4 — Native PyTorch LoRA injector, pooling head & FLIP2 utilities [GPU]\n{core_body}\n\nfor _ds, _sp in TASKS:\n    safe_token(_ds); safe_token(_sp)\nprint('[OK] LoRA modules and FLIP2 utilities ready.')")

code('''# Cell 5 — Load base ESM-2 once onto GPU & inject LoRA [GPU]
from transformers import AutoTokenizer, EsmModel

device = torch.device("cuda" if HAS_CUDA else "cpu")
dtype = torch.bfloat16 if (HAS_CUDA and BF16_SUPPORTED) else (torch.float16 if HAS_CUDA else torch.float32)

t_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(HF_ID)
try:
    backbone = EsmModel.from_pretrained(HF_ID, torch_dtype=dtype, add_pooling_layer=False,
                                        attn_implementation="sdpa").to(device)
except Exception:
    backbone = EsmModel.from_pretrained(HF_ID, torch_dtype=dtype, add_pooling_layer=False).to(device)

if USE_GRAD_CKPT:
    backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    else:
        backbone.get_input_embeddings().register_forward_hook(lambda _m, _in, out: out.requires_grad_(True))

n_lora = inject_esm_lora(backbone, LORA_TARGET_MODULES, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
hidden_dim = int(backbone.config.hidden_size)
special_ids = list(tokenizer.all_special_ids)
n_total = sum(p.numel() for p in backbone.parameters())
n_train_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"[MODEL] {HF_ID} | dtype={dtype} | replaced={n_lora} layers | backbone_trainable={n_train_lora:,}/{n_total:,} ({100*n_train_lora/n_total:.3f}%) | loaded in {time.time()-t_load:.1f}s")''')

code('''# Cell 6 — LoRA training & evaluation loop (val_mse selection, atomic per-run cache) [GPU]
def get_trainable_state(model: EsmLoRARegressor) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable_state(model: EsmLoRARegressor, state: Dict[str, torch.Tensor]):
    with torch.no_grad():
        params = dict(model.named_parameters())
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, dtype=params[k].dtype))


@torch.inference_mode()
def predict_split(model: EsmLoRARegressor, seqs: List[str], max_tokens: int, max_seqs: int) -> np.ndarray:
    model.eval()
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    batches = build_token_batches(lens, max_tokens=max_tokens, max_seqs=max_seqs, longest_first=True)
    preds = np.empty(len(seqs), dtype=np.float32)
    for b_idx in batches:
        b_seqs = [seqs[i] for i in b_idx]
        enc = tokenizer(b_seqs, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
        out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu().numpy()
        for local_i, orig_i in enumerate(b_idx):
            preds[orig_i] = out[local_i]
    return preds


def run_one(ds: str, sp: str, seed: int):
    run_tag = f"{safe_token(ds)}__{safe_token(sp)}__seed{int(seed)}"
    run_json = confine_path(LORA_RUN_DIR, LORA_RUN_DIR / safe_name(f"{run_tag}.json"))
    pred_npz = confine_path(LORA_PRED_DIR, LORA_PRED_DIR / safe_name(f"{run_tag}.npz"))
    if run_json.exists() and not OVERWRITE_RUNS:
        print(f"[SKIP] {run_tag} already finished.")
        return

    seqs_all, y_std, fit_mask, val_mask, test_mask = load_split_df(RAW_DIR, ds, sp)
    fit_seqs = [s for s, m in zip(seqs_all, fit_mask) if m]
    val_seqs = [s for s, m in zip(seqs_all, val_mask) if m]
    test_seqs = [s for s, m in zip(seqs_all, test_mask) if m]
    y_fit = y_std[fit_mask]
    y_val = y_std[val_mask]
    y_test = y_std[test_mask]

    if SMOKE_TEST:
        fit_seqs, y_fit = fit_seqs[:64], y_fit[:64]
        val_seqs, y_val = val_seqs[:32], y_val[:32]
        test_seqs, y_test = test_seqs[:32], y_test[:32]

    torch.manual_seed(seed)
    if HAS_CUDA:
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    reset_esm_lora(backbone)
    model = EsmLoRARegressor(backbone, hidden_dim, special_ids, dropout=0.1).to(device)
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    max_epochs = 2 if SMOKE_TEST else (10 if len(fit_seqs) >= 8000 else 15)
    patience = 2 if SMOKE_TEST else (3 if len(fit_seqs) >= 8000 else 4)
    fit_lens = np.array([len(s) for s in fit_seqs], dtype=np.int64)

    n_batches_est = len(build_token_batches(fit_lens, TRAIN_MAX_TOKENS, TRAIN_MAX_SEQS, shuffle=True, rng=rng))
    total_steps = max(1, n_batches_est * max_epochs)
    warmup_steps = max(1, int(0.05 * total_steps))

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    best_val_mse = float("inf")
    best_val_rho = float("nan")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        batches = build_token_batches(fit_lens, TRAIN_MAX_TOKENS, TRAIN_MAX_SEQS,
                                      shuffle=True, rng=rng, longest_first=(epoch == 1))
        ep_loss, n_seen = 0.0, 0
        for b_idx in batches:
            b_seqs = [fit_seqs[i] for i in b_idx]
            b_y = torch.from_numpy(y_fit[b_idx]).to(device=device, dtype=torch.float32)
            enc = tokenizer(b_seqs, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.detach().cpu()) * len(b_idx)
            n_seen += len(b_idx)

        val_preds = predict_split(model, val_seqs, EVAL_MAX_TOKENS, EVAL_MAX_SEQS)
        vm = regression_metrics(y_val, val_preds, const_tol=CONST_TOL)
        history.append({"epoch": epoch, "train_mse": ep_loss / max(1, n_seen),
                        "val_mse": vm["mse"], "val_spearman": vm["spearman"]})
        print(f"  [{run_tag}] ep {epoch:02d}/{max_epochs} | train_mse={ep_loss/n_seen:.4f} | val_mse={vm['mse']:.4f} | val_rho={vm['spearman']:.4f}")

        if vm["mse"] < best_val_mse:
            best_val_mse = vm["mse"]
            best_val_rho = vm["spearman"]
            best_epoch = epoch
            best_state = get_trainable_state(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 3:
                print(f"  [{run_tag}] early stop at epoch {epoch} (best={best_epoch})")
                break

    load_trainable_state(model, best_state)
    val_best_preds = predict_split(model, val_seqs, EVAL_MAX_TOKENS, EVAL_MAX_SEQS)
    test_preds = predict_split(model, test_seqs, EVAL_MAX_TOKENS, EVAL_MAX_SEQS)
    tm = regression_metrics(y_test, test_preds, const_tol=CONST_TOL)
    elapsed = time.time() - t0

    save_npz_atomic(pred_npz, val_preds=val_best_preds, val_targets=y_val,
                    test_preds=test_preds, test_targets=y_test)
    payload = {
        "dataset": ds, "split": sp, "model": f"{MODEL_KEY}_lora", "seed": int(seed),
        "smoke_test": bool(SMOKE_TEST), "tier": DETECTED_TIER,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr_lora": LR_LORA, "lr_head": LR_HEAD,
        "n_fit": len(fit_seqs), "n_val": len(val_seqs), "n_test": len(test_seqs),
        "best_epoch": int(best_epoch), "best_val_mse": float(best_val_mse), "best_val_spearman": float(best_val_rho),
        "test_spearman": tm["spearman"], "test_mse": tm["mse"], "test_r2": tm["r2"], "test_ndcg": tm["ndcg"],
        "pred_std": tm["pred_std"], "pred_constant": tm["pred_constant"],
        "seconds": round(elapsed, 1), "history": history,
    }
    save_json_atomic(run_json, payload)
    print(f"[SAVED] {run_tag} | best_ep={best_epoch} | test_rho={tm['spearman']:.4f} | test_mse={tm['mse']:.4f} | test_ndcg={tm['ndcg']:.4f} | {elapsed:.1f}s")
    del model, optimizer, scheduler, best_state
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()


for ds, sp in TASKS:
    for seed in SEEDS:
        print(f"\\n=== Running {ds}/{sp} (seed={seed}) ===")
        run_one(ds, sp, seed)

print("\\n[DONE] Scheduled LoRA runs completed.")''')

code('''# Cell 7 — Aggregate all LoRA runs across accounts & compare with frozen 3B [GPU or CPU]
per_seed, summary = aggregate_lora_runs(LORA_RUN_DIR, TAB_DIR, smoke_test=SMOKE_TEST)
if summary is not None:
    suffix = "_smoke" if SMOKE_TEST else ""
    print(f"[SAVED] {len(per_seed)} runs -> lora_3b_summary{suffix}.csv")
    print(summary.to_string(index=False))
else:
    print("[WARN] No matching LoRA run JSON files found yet.")''')

code('''# Cell 8 — Flush Drive & unmount [GPU]
if IN_COLAB:
    print("[SYNC] Flushing Google Drive buffers...")
    drive.flush_and_unmount()
    print("[DONE] Drive unmounted safely. Disconnect the runtime now to stop CU billing.")
else:
    print("[DONE] Local run completed.")''')

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebook_04_esm2_3b_lora_comparison.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
