# -*- coding: utf-8 -*-
"""Builder for notebook_02_frozen_plm_embedding_extraction.ipynb (W1 + W2)."""
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


md("""# 02 — Frozen PLM Embedding Extraction (GPU, W1 + W2)

**Recommended W2 execution order (minimizes CU & session-timeout risk):**
1. **`DATASET_GROUP = "w2_medium"`** (`ired`, `hydro`, `nucb` — ~89k seqs, short/medium):
   - **L4** (`RUN_PROFILE = "auto"` → 8M/35M/150M/650M): ~12–15 min (**~1.1 CU**).
   - **G4** (`RUN_PROFILE = "auto"` → 3B): ~5–6 min (**~1.0 CU**).
2. **`DATASET_GROUP = "w2_trpb"`** (`trpb` — 228,298 seqs, 389 aa):
   - **L4** (`RUN_PROFILE = "auto"` → 8M..650M, ~2.5 h / **~12 CU**; or set `MODELS_OVERRIDE = ["esm2_150m", "esm2_650m"]` for ~1.8 h / **~8.5 CU**).
   - **G4** (`RUN_PROFILE = "auto"` → 3B full 228k, ~44 min / **~7.7 CU**).
   - *Budget alternative*: set `DATASET_GROUP = "w2_trpb_sub50k"` on G4 for 3B stratified 50k (~10 min / **~1.7 CU**).

**Cost safety rules (enforced in code):**
- ≤650M on G4 is blocked unless `ALLOW_SMALL_ON_G4=True`.
- Longest-first dynamic token-budget batching (OOM fails on batch 0, never mid-run).
- Existing `(model, dataset, unique_sha[:12]).npz` caches are skipped (`OVERWRITE_CACHE=False`).
- Atomic `.npz` serialization + `drive.flush_and_unmount()` in Cell 7.
""")

code('''# Cell 1 — Workspace verification & GPU telemetry [GPU]
import os, sys, gc, io, json, time, re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch

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
    print(f"  logical : {BASE_DIR}")
    print(f"  physical: {os.path.realpath(BASE_DIR)}")

PROC_DIR  = (BASE_DIR / "data" / "processed").resolve()
CACHE_DIR = (BASE_DIR / "cache" / "embeddings").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this notebook. Use an L4 or G4 runtime.")

GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
BF16_SUPPORTED = torch.cuda.is_bf16_supported()
if GPU_VRAM_GB >= 70:
    DETECTED_TIER, EST_CU_PER_HOUR = "G4", 10.5
elif GPU_VRAM_GB >= 20:
    DETECTED_TIER, EST_CU_PER_HOUR = "L4", 4.7
else:
    DETECTED_TIER, EST_CU_PER_HOUR = "T4", 1.8
print(f"[GPU] {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) | tier={DETECTED_TIER} (~{EST_CU_PER_HOUR} CU/h) | bf16={BF16_SUPPORTED}")''')

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

code('''# Cell 3 — Configuration & model registry [GPU]
SMOKE_TEST = False
RUN_PROFILE = "auto"            # "auto" | "l4_small" | "g4_large"
DATASET_GROUP = "w2_medium"     # "w1" | "w2_medium" | "w2_trpb" | "w2_trpb_sub50k" | "custom"
TARGET_DATASETS = []            # used only when DATASET_GROUP == "custom"
MODELS_OVERRIDE = None          # e.g. ["esm2_150m", "esm2_650m"], or None for profile default
ALLOW_SMALL_ON_G4 = False
OVERWRITE_CACHE = False

DATASET_GROUPS = {
    "w1":             ["amylase", "rhomax"],
    "w2_medium":      ["ired", "hydro", "nucb"],
    "w2_trpb":        ["trpb"],
    "w2_trpb_sub50k": ["trpb_sub50k"],
}
if DATASET_GROUP != "custom":
    TARGET_DATASETS = DATASET_GROUPS[DATASET_GROUP]
assert TARGET_DATASETS, "TARGET_DATASETS cannot be empty"

MODEL_REGISTRY = {
    "esm2_8m":   {"hf_id": "facebook/esm2_t6_8M_UR50D",   "hidden_dim": 320,  "tier": "small", "max_tokens_l4": 48000, "max_tokens_g4": 96000},
    "esm2_35m":  {"hf_id": "facebook/esm2_t12_35M_UR50D", "hidden_dim": 480,  "tier": "small", "max_tokens_l4": 36000, "max_tokens_g4": 80000},
    "esm2_150m": {"hf_id": "facebook/esm2_t30_150M_UR50D","hidden_dim": 640,  "tier": "small", "max_tokens_l4": 24000, "max_tokens_g4": 64000},
    "esm2_650m": {"hf_id": "facebook/esm2_t33_650M_UR50D","hidden_dim": 1280, "tier": "small", "max_tokens_l4": 16000, "max_tokens_g4": 48000},
    "esm2_3b":   {"hf_id": "facebook/esm2_t36_3B_UR50D",  "hidden_dim": 2560, "tier": "large", "max_tokens_l4": 0,     "max_tokens_g4": 32000},
}

def resolve_models() -> List[str]:
    if SMOKE_TEST:
        return ["esm2_8m"]
    if MODELS_OVERRIDE is not None:
        return list(MODELS_OVERRIDE)
    if RUN_PROFILE == "auto":
        return ["esm2_3b"] if DETECTED_TIER == "G4" else ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]
    if RUN_PROFILE == "l4_small":
        return ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]
    if RUN_PROFILE == "g4_large":
        return ["esm2_3b"]
    raise ValueError(f"Invalid RUN_PROFILE: {RUN_PROFILE}")

MODELS_TO_RUN = resolve_models()
for m in MODELS_TO_RUN:
    spec = MODEL_REGISTRY[m]
    if DETECTED_TIER == "G4" and spec["tier"] == "small" and not ALLOW_SMALL_ON_G4:
        raise RuntimeError(f"[Anti-Waste Guard] Refusing small model {m} on G4 without ALLOW_SMALL_ON_G4=True.")
    if DETECTED_TIER != "G4" and spec["tier"] == "large":
        raise RuntimeError(f"[Tier Guard] {m} requires G4 (96 GB). Current tier: {DETECTED_TIER}.")

ACTIVE_CACHE_DIR = (CACHE_DIR / "_smoke_test") if SMOKE_TEST else CACHE_DIR
ACTIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"[CFG] tier={DETECTED_TIER} group={DATASET_GROUP} targets={TARGET_DATASETS} smoke={SMOKE_TEST}")
print(f"[CFG] scheduled models: {MODELS_TO_RUN}")''')

code('''# Cell 4 — Load per-dataset deduplicated sequences & manifest [GPU]
man_path = PROC_DIR / "benchmark_manifest.json"
if not man_path.exists():
    man_path = PROC_DIR / "pilot_manifest.json"
manifest = json.loads(man_path.read_text(encoding="utf-8"))

dataset_tables = {}
for ds in TARGET_DATASETS:
    if not re.fullmatch(r"[A-Za-z0-9_]+", ds):
        raise ValueError(f"invalid dataset token: {ds!r}")
    per_ds_csv = (PROC_DIR / f"unique_sequences__{ds}.csv").resolve()
    assert per_ds_csv.is_relative_to(PROC_DIR.resolve())
    if per_ds_csv.exists():
        sub = pd.read_csv(per_ds_csv)
    else:
        sub = pd.read_csv((PROC_DIR / "unique_sequences.csv").resolve())
        sub = sub[sub["dataset"] == ds].reset_index(drop=True)
    uniq_sha = manifest["datasets"][ds]["unique_sha256"]
    assert len(sub) == manifest["datasets"][ds]["n_unique"], f"{ds}: registry/manifest count mismatch"
    if SMOKE_TEST:
        sub = sub.head(10).reset_index(drop=True)
        uniq_sha = f"smoke10_{uniq_sha[:8]}"
    dataset_tables[ds] = (sub, uniq_sha)
    print(f"[LOAD] {ds:12s}: {len(sub):6d} unique sequences | registry_sha={uniq_sha[:12]}")''')

code('''# Cell 5 — Token-budget batching, pooling & atomic disk serialization [GPU]
from transformers import AutoTokenizer, EsmModel

def build_token_batches(seq_lens: np.ndarray, max_tokens: int) -> List[List[int]]:
    order = np.argsort(-seq_lens, kind="mergesort")  # longest-first: OOM fails on batch 0
    batches, cur, cur_max = [], [], 0
    for idx in order.tolist():
        L = int(seq_lens[idx]) + 2
        if not cur:
            cur, cur_max = [idx], L
            continue
        if (len(cur) + 1) * max(cur_max, L) <= max_tokens:
            cur.append(idx); cur_max = max(cur_max, L)
        else:
            batches.append(cur); cur, cur_max = [idx], L
    if cur:
        batches.append(cur)
    return batches

@torch.inference_mode()
def extract_esm2_embeddings(seqs, seq_lens, tokenizer, model, hidden_dim, max_tokens, device):
    out = np.empty((len(seqs), hidden_dim), dtype=np.float32)
    batches = build_token_batches(np.asarray(seq_lens), max_tokens)
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)
    for batch_indices in tqdm(batches, desc="extract"):
        batch_seqs = [seqs[i] for i in batch_indices]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                        truncation=False, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        hidden = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
        res_mask = mask.bool() & (~torch.isin(input_ids, special_ids))
        mask_f = res_mask.unsqueeze(-1).to(torch.float32)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (hidden.to(torch.float32) * mask_f).sum(dim=1) / denom
        pooled_np = pooled.cpu().numpy()
        for local_i, orig_i in enumerate(batch_indices):
            out[orig_i] = pooled_np[local_i]
    return out

def save_npz_atomic(out_path: Path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, out_path)

print("[OK] extraction utilities ready.")''')

code('''# Cell 6 — Extraction loop (cache-aware, atomic, memory-safe) [GPU]
device = torch.device("cuda")
dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float16
run_log = []

for model_key in MODELS_TO_RUN:
    spec = MODEL_REGISTRY[model_key]
    print(f"\\n=== {model_key} ({spec['hf_id']}) ===")
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    try:
        model = EsmModel.from_pretrained(spec["hf_id"], torch_dtype=dtype,
                                         add_pooling_layer=False, attn_implementation="sdpa").to(device)
    except Exception:
        model = EsmModel.from_pretrained(spec["hf_id"], torch_dtype=dtype,
                                         add_pooling_layer=False).to(device)
    model.eval()
    max_tok = spec["max_tokens_g4"] if DETECTED_TIER == "G4" else spec["max_tokens_l4"]

    for ds in TARGET_DATASETS:
        sub_df, uniq_sha = dataset_tables[ds]
        fname = f"{model_key}__{ds}__{uniq_sha[:12]}.npz"
        out_path = (ACTIVE_CACHE_DIR / fname).resolve()
        if not out_path.is_relative_to(ACTIVE_CACHE_DIR.resolve()):
            raise ValueError(f"cache path escapes dir: {out_path}")
        if out_path.exists() and not OVERWRITE_CACHE:
            print(f"[SKIP] cache exists: {fname}")
            continue
        t0 = time.time()
        embs = extract_esm2_embeddings(sub_df["sequence"].tolist(), sub_df["seq_len"].values,
                                       tokenizer, model, spec["hidden_dim"], max_tok, device)
        elapsed = time.time() - t0
        save_npz_atomic(out_path, embeddings=embs,
                        seq_hashes=np.asarray(sub_df["seq_hash"].tolist(), dtype=np.str_),
                        seq_lens=sub_df["seq_len"].values)
        run_log.append({"model": model_key, "dataset": ds, "file": fname,
                        "n": int(len(embs)), "dim": int(embs.shape[1]),
                        "seconds": round(elapsed, 1), "tier": DETECTED_TIER})
        print(f"[SAVED] {fname} | shape={embs.shape} | {elapsed:.1f}s")
        del embs; gc.collect()

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

pd.DataFrame(run_log).to_csv(ACTIVE_CACHE_DIR / "extraction_log.csv", index=False)
print(f"\\n[DONE] {len(run_log)} new cache files. Tier={DETECTED_TIER} (~{EST_CU_PER_HOUR} CU/h).")''')

code('''# Cell 7 — Flush Drive & unmount [GPU]
if IN_COLAB:
    print("[SYNC] Flushing Google Drive buffers...")
    drive.flush_and_unmount()
    print("[DONE] Drive unmounted safely. Disconnect the runtime now to stop CU billing.")
else:
    print("[DONE] Local run; caches written to", ACTIVE_CACHE_DIR)''')

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
out = ROOT / "notebook_02_frozen_plm_embedding_extraction.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")