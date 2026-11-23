# -*- coding: utf-8 -*-
"""CPU verification tests for notebook_04_esm2_3b_lora_comparison.ipynb and lora_core.py."""
import ast
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import lora_core

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "notebook_04_esm2_3b_lora_comparison.ipynb"
RAW_DIR = (ROOT / "flip2_benchmark" / "data" / "raw").resolve()


class DummySelfAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.out_proj(self.query(x) + self.key(x) + self.value(x))


class DummyOut:
    def __init__(self, t):
        self.last_hidden_state = t


class DummyEsmBackbone(nn.Module):
    def __init__(self, vocab=32, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.attn = DummySelfAttn(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        h = h + self.attn(h)
        return DummyOut(self.norm(h))


def test_lora_injection_and_determinism():
    torch.manual_seed(123)
    backbone = DummyEsmBackbone(vocab=32, d=64)
    x_ids = torch.randint(4, 24, (4, 12))
    mask = torch.ones(4, 12, dtype=torch.long)
    mask[0, 9:] = 0

    with torch.no_grad():
        base_out = backbone(x_ids, mask).last_hidden_state.clone()
        orig_q_weight = backbone.attn.query.weight.clone()
        orig_emb_weight = backbone.emb.weight.clone()

    replaced = lora_core.inject_esm_lora(backbone, target_names=("query", "key", "value"), r=4, alpha=8, dropout=0.0)
    assert replaced == 3, f"expected 3 replaced layers, got {replaced}"

    backbone.eval()
    with torch.no_grad():
        lora_init_out = backbone(x_ids, mask).last_hidden_state
    assert torch.allclose(base_out, lora_init_out, atol=1e-6), "Initial LoRA output must match base model"

    reg = lora_core.EsmLoRARegressor(backbone, hidden_dim=64, special_ids=[0, 1, 2], dropout=0.0)
    trainable = [p for p in reg.parameters() if p.requires_grad]
    frozen_names = [n for n, p in reg.named_parameters() if not p.requires_grad]
    assert "backbone.emb.weight" in frozen_names
    assert "backbone.attn.query.base.weight" in frozen_names
    assert len(trainable) == 3 * 2 + 4

    opt = torch.optim.AdamW(trainable, lr=1e-2)
    y = torch.randn(4)
    for _ in range(3):
        opt.zero_grad()
        loss = nn.MSELoss()(reg(x_ids, mask), y)
        loss.backward()
        opt.step()

    assert torch.equal(orig_q_weight, backbone.attn.query.base.weight), "Frozen base linear changed!"
    assert torch.equal(orig_emb_weight, backbone.emb.weight), "Frozen embedding changed!"
    assert backbone.attn.query.lora_B.abs().sum().item() > 0, "LoRA B did not update!"

    torch.manual_seed(42)
    lora_core.reset_esm_lora(backbone)
    a1 = backbone.attn.query.lora_A.clone()
    assert backbone.attn.query.lora_B.abs().sum().item() == 0.0

    torch.manual_seed(42)
    lora_core.reset_esm_lora(backbone)
    a2 = backbone.attn.query.lora_A.clone()
    assert torch.equal(a1, a2), "reset_esm_lora must be deterministic under seed"
    print("[PASS] test_lora_injection_and_determinism")


def test_special_token_pooling_mask():
    torch.manual_seed(0)
    backbone = DummyEsmBackbone(vocab=32, d=16)
    reg = lora_core.EsmLoRARegressor(backbone, hidden_dim=16, special_ids=[0, 1, 2], dropout=0.0)
    ids1 = torch.tensor([[0, 5, 6, 7, 2]])
    msk1 = torch.tensor([[1, 1, 1, 1, 1]])
    ids2 = torch.tensor([[0, 5, 6, 7, 2, 1, 1]])
    msk2 = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    res_mask1 = msk1.bool() & (~torch.isin(ids1, reg.special_ids))
    res_mask2 = msk2.bool() & (~torch.isin(ids2, reg.special_ids))
    assert res_mask1.tolist() == [[False, True, True, True, False]]
    assert res_mask2.tolist() == [[False, True, True, True, False, False, False]]
    print("[PASS] test_special_token_pooling_mask")


def test_batching_and_real_data_alignment():
    rng = np.random.default_rng(42)
    lens = rng.integers(50, 425, size=200)
    batches = lora_core.build_token_batches(lens, max_tokens=4096, max_seqs=32, shuffle=True, rng=rng)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(200)), "Batching dropped or duplicated samples"
    assert all(len(b) <= 32 for b in batches), "Exceeded max_seqs"
    assert all(len(b) * (max(lens[b]) + 2) <= 4096 for b in batches), "Exceeded max_tokens"

    for ds, sp, exp_fit, exp_val, exp_test in [
        ("amylase", "one_to_many", 2574, 644, 488),
        ("amylase", "close_to_far", 1426, 356, 1924),
        ("hydro", "three_to_many", 1160, 290, 23485),
        ("hydro", "to_P0A9X9", 13148, 3287, 8500),
    ]:
        seqs, y_std, fit_m, val_m, test_m = lora_core.load_split_df(RAW_DIR, ds, sp)
        assert (fit_m.sum(), val_m.sum(), test_m.sum()) == (exp_fit, exp_val, exp_test)
        assert not (fit_m & val_m).any() and not (fit_m & test_m).any() and not (val_m & test_m).any()
        train_all = fit_m | val_m
        assert abs(float(y_std[train_all].mean())) < 1e-5
        assert abs(float(y_std[train_all].std()) - 1.0) < 1e-4
    print("[PASS] test_batching_and_real_data_alignment")


def test_notebook_syntax_and_aggregation():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse(c["source"], filename=f"nb04_cell_{i}")

    # Verify gradient checkpointing hook preserves LoRA gradients when embeddings are frozen
    torch.manual_seed(7)
    ckpt_backbone = DummyEsmBackbone(vocab=32, d=32)
    lora_core.inject_esm_lora(ckpt_backbone, target_names=("query", "key", "value"), r=4, alpha=8, dropout=0.0)
    ckpt_backbone.emb.register_forward_hook(lambda _m, _in, out: out.requires_grad_(True))
    x_ids = torch.randint(4, 24, (2, 8))
    ckpt_opt = torch.optim.SGD([p for p in ckpt_backbone.parameters() if p.requires_grad], lr=1e-2)
    # Step 1: lora_B is zero-init so lora_B gets gradient first; Step 2: lora_A also gets non-zero gradient
    for step in range(2):
        ckpt_opt.zero_grad()
        h = ckpt_backbone.emb(x_ids)
        out = torch.utils.checkpoint.checkpoint(ckpt_backbone.attn, h, use_reentrant=True)
        out.sum().backward()
        if step == 0:
            assert ckpt_backbone.attn.query.lora_B.grad is not None and ckpt_backbone.attn.query.lora_B.grad.abs().sum() > 0
        else:
            assert ckpt_backbone.attn.query.lora_A.grad is not None and ckpt_backbone.attn.query.lora_A.grad.abs().sum() > 0
        ckpt_opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        run_dir = tmp_p / "runs"
        tab_dir = tmp_p / "tables"
        run_dir.mkdir()
        tab_dir.mkdir()
        for fn in ("ridge_probing_metrics.csv", "uq_ensemble_metrics.csv"):
            src = ROOT / "_colab_outputs_w2" / "outputs" / "tables" / fn
            (tab_dir / fn).write_bytes(src.read_bytes())
        for seed, rho in [(42, 0.55), (43, 0.57), (44, 0.56)]:
            (run_dir / f"amylase__close_to_far__seed{seed}.json").write_text(
                json.dumps({
                    "dataset": "amylase", "split": "close_to_far", "model": "esm2_3b_lora",
                    "seed": seed, "smoke_test": False, "best_epoch": 6,
                    "test_spearman": rho, "test_mse": 0.85, "test_r2": 0.25, "test_ndcg": 0.94,
                    "seconds": 320.0, "history": [],
                }),
                encoding="utf-8",
            )
        per_seed, summary = lora_core.aggregate_lora_runs(run_dir, tab_dir, smoke_test=False)
        assert len(per_seed) == 3
        assert len(summary) == 1
        assert abs(summary.loc[0, "spearman_mean"] - 0.56) < 1e-6
        assert abs(summary.loc[0, "frozen_ridge_rho"] - 0.537705) < 1e-4
        assert abs(summary.loc[0, "frozen_mlp5_rho"] - 0.588837) < 1e-4
    print("[PASS] test_notebook_syntax_and_aggregation")


if __name__ == "__main__":
    test_lora_injection_and_determinism()
    test_special_token_pooling_mask()
    test_batching_and_real_data_alignment()
    test_notebook_syntax_and_aggregation()
    print("\nALL NB04 CHECKS PASSED.")
