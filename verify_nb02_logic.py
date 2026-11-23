# -*- coding: utf-8 -*-
"""CPU verification of notebook_02/03 PLM-cache logic (no GPU needed).

Validates: (1) longest-first token-budget batching, (2) special-token-excluded mean
pooling, (3) hash alignment of a deduplicated embedding cache back to labelled split
rows including Rhomax duplicate measurements. Mock embeddings are written ONLY to
_local_test/mock_cache, never to the real cache.
"""
import io, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "flip2_benchmark"
PROC = BASE / "data" / "processed"
MOCK = ROOT / "_local_test" / "mock_cache"
MOCK.mkdir(parents=True, exist_ok=True)

manifest = json.loads((PROC / "pilot_manifest.json").read_text(encoding="utf-8"))
uniq = pd.read_csv(PROC / "unique_sequences.csv")


def build_token_batches(seq_lens, max_tokens):
    order = np.argsort(-seq_lens, kind="mergesort")
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


# ---- (1) batching ----
lens = np.array([425] * 3706)
batches = build_token_batches(lens, 16000)
flat = [i for b in batches for i in b]
assert sorted(flat) == list(range(3706)), "batching must cover every index exactly once"
for b in batches:
    assert (len(b)) * (425 + 2) <= 16000, "token cap violated"
mixed = np.array([100, 400, 50, 300, 200])
mb = build_token_batches(mixed, 600)
first_len = mixed[mb[0][0]]
assert first_len == mixed.max(), "longest-first ordering violated"
print(f"[OK] batching: {len(batches)} batches for 3706x425 (cap 16000); longest-first verified")

# ---- (2) special-token-excluded mean pooling ----
import torch
# fake batch: 2 sequences, 5 positions; special ids = {0, 2}
input_ids = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 9, 2, 2]])
mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
special = torch.tensor([0, 2])
hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
res_mask = mask.bool() & (~torch.isin(input_ids, special))
mf = res_mask.unsqueeze(-1).float()
pooled = (hidden * mf).sum(1) / mf.sum(1).clamp(min=1.0)
# row0 residues at pos1,2,3 -> hidden values; row1 residues at pos1,2 only
exp0 = hidden[0, 1:4].mean(0)
exp1 = hidden[1, 1:3].mean(0)
assert torch.allclose(pooled[0], exp0) and torch.allclose(pooled[1], exp1), "pooling mask wrong"
print("[OK] mean pooling excludes BOS/EOS/PAD correctly")

# ---- (3) hash alignment with mock deduplicated cache (Rhomax duplicates) ----
ds = "rhomax"
sub = uniq[uniq.dataset == ds].reset_index(drop=True)
sha = manifest["datasets"][ds]["unique_sha256"]
dim = 320
rng = np.random.RandomState(0)
mock_embs = rng.randn(len(sub), dim).astype(np.float32)
fname = f"esm2_8m__{ds}__{sha[:12]}.npz"
buf = io.BytesIO()
np.savez(buf, embeddings=mock_embs, seq_hashes=np.asarray(sub["seq_hash"].tolist(), dtype=np.str_),
         seq_lens=sub["seq_len"].values)
(MOCK / fname).write_bytes(buf.getvalue())

cleaned = pd.read_csv(PROC / f"{ds}__by_wild_type__cleaned.csv")
d = np.load(MOCK / fname)
h2i = {h: i for i, h in enumerate(d["seq_hashes"].astype(str))}
assert set(cleaned.seq_hash).issubset(set(h2i)), "cache missing some split hashes"
X = d["embeddings"][[h2i[h] for h in cleaned.seq_hash]]
assert X.shape == (len(cleaned), dim), f"alignment shape {X.shape}"
# duplicate measurements must map to identical embedding rows
dup_hashes = cleaned.seq_hash[cleaned.seq_hash.duplicated(keep=False)].unique()
n_dup_checked = 0
for h in dup_hashes:
    rows = np.where(cleaned.seq_hash.values == h)[0]
    assert np.allclose(X[rows[0]], X[rows[1]]), "duplicate seq got different embeddings"
    n_dup_checked += 1
print(f"[OK] alignment: {len(cleaned)} labelled rows <- {len(sub)} unique embeddings; "
      f"{n_dup_checked} duplicate sequences map identically (884 rows / 878 unique)")

print("\n[ALL LOGIC CHECKS PASSED]")