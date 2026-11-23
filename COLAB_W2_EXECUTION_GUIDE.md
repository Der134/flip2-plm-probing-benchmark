# FLIP2 W2 扩展数据集 — Colab 执行手册

> 三个 notebook 已全部升级为 **W1 + W2 统一版**，**W1 已缓存的 `amylase`（`ce78b13d3cbe`）与 `rhomax`（`c378d59a8726`）SHA-256 完全不变**，可直接复用。

---

## 一、W2 数据集与本次代码核查到的关键特征

| 数据集 | 切分数 | 唯一序列 | 序列长度 | 任务类型与关键协议细节 |
|---|---|---:|---:|---|
| **IRED** | 1（`two_to_many`） | 8,586 | 290 aa | 回归（注：官方 CSV 列序不同，nb01 已按列名对齐） |
| **Hydro** | 5（`three_to_many`, `low_to_high`, `to_P06241`, `to_P0A9X9`, `to_P01053`） | 24,935 | 57–65 aa | 回归（变长同源小蛋白，nb03 主表自动采用 `l2_norm`） |
| **NucB** | 1（`two_to_many`） | 55,759 | 142 aa | **4 活性 bin 序数分类**：`fit` 只有 `{0,1,2}`、`bin=3` 仅在 `val/test` 出现；nb03 报 4-bin Acc/Macro-F1 + 高活性 `bin>=2` AUROC/AUPRC + 序数 Spearman/NDCG |
| **TrpB** | 3（`one_to_many`, `two_to_many`, `by_position`） | 228,298 | 389 aa | 回归（`one_hot` 已改用稀疏 CSR 矩阵，内存从 7.1 GB 降至 710 MB） |

---

## 二、Colab 执行顺序与实测推算 CU 预算

先将根目录更新后的三个 `.ipynb` 上传覆盖 Drive / Colab 中的旧版：
- `notebook_01_data_prep_one_hot_baseline.ipynb`
- `notebook_02_frozen_plm_embedding_extraction.ipynb`
- `notebook_03_probing_data_efficiency_uq.ipynb`

### Step 1：数据准备与基线（CPU，~0 CU，~1–2 min）
- 运行 `notebook_01`（默认 `RUN_SCOPE = "all"`，已存在的 W1 文件会自动 `[SKIP]`）。
- 产出 `benchmark_manifest.json`、各数据集独立的 `unique_sequences__{ds}.csv`、`trpb_sub50k` 注册表，以及全部 13 个切分的官方 `one_hot` 基线表。

### Step 2：中等规模数据集 `w2_medium`（`ired` + `hydro` + `nucb`，~89k 序列）
1. **L4 runtime**（`notebook_02` Cell 3 保持默认 `DATASET_GROUP = "w2_medium"`, `RUN_PROFILE = "auto"`）：
   - 提取 `esm2_8m / 35m / 150m / 650m` × 3 数据集（12 个缓存文件）。
   - 预估耗时 **~12–15 min**，消耗 **~1.1 CU**。跑完 Cell 7 后断开。
2. **G4 runtime**（`notebook_02` Cell 3 保持 `DATASET_GROUP = "w2_medium"`, `RUN_PROFILE = "auto"`）：
   - 自动识别 G4 只提 `esm2_3b` × 3 数据集（3 个缓存文件）。
   - 预估耗时 **~5–6 min**，消耗 **~1.0 CU**。跑完 Cell 7 **立刻断开**。

### Step 3：超大规模数据集 `trpb`（228,298 序列 × 389 aa）
- **L4 runtime**：Cell 3 改 `DATASET_GROUP = "w2_trpb"`：
  - 方案 A（全梯度 `8M..650M`）：预估 **~2.5 h / ~12 CU**（每个模型写盘一次，断连也不丢已完模型；也可两账号分工，账号 1 设 `MODELS_OVERRIDE=["esm2_8m","esm2_35m","esm2_150m"]`、账号 2 设 `MODELS_OVERRIDE=["esm2_650m"]` 并发）。
  - 方案 B（研究方案原定省算力档）：设 `MODELS_OVERRIDE = ["esm2_150m", "esm2_650m"]`，预估 **~1.8 h / ~8.5 CU**。
- **G4 runtime**：二选一：
  - **推荐全量**（按实测 G4 吞吐推算很便宜）：Cell 3 设 `DATASET_GROUP = "w2_trpb"`，提取完整 228k `esm2_3b`，预估 **~44 min / ~7.7 CU**（无任何子采样偏差，直接与其他尺度 1:1 对比）。
  - **省 CU 备选**：Cell 3 设 `DATASET_GROUP = "w2_trpb_sub50k"`，提取分层 50k `esm2_3b`，预估 **~10 min / ~1.7 CU**（nb03 会自动识别并在表里标 `eval_subset=sub50k`）。

### Step 4：全量分析与出图（CPU，~0 CU，~15–25 min）
- 可在 Step 2 完成后就先跑一次 `notebook_03`（缺失的 `trpb` 会自动 `[SKIP]`，先看 IRED/Hydro/NucB 结果），等 Step 3 提完 TrpB 后再重跑一遍 `notebook_03` 出全量 6 数据集 · 13 切分总表与 4 张图件。

---

## 三、推荐分步节奏（最稳不超时）

1. 先跑 **Step 1（CPU）+ Step 2（L4 ~1.1 CU + G4 ~1.0 CU）+ Step 4（CPU）**：只花 **~2 CU** 就先把 **IRED + Hydro + NucB** 全出表！
2. 看完结果后，再开 **Step 3（TrpB，L4 + G4）** 并跑最终 `notebook_03`。
