# FLIP2 Pilot — Colab 执行手册（方向 A · W1）

> 配套三个 notebook：`notebook_01`（CPU 数据准备）→ `notebook_02`（GPU 嵌入提取）→ `notebook_03`（CPU 探针/数据效率/UQ）。
> 本手册给出**开机顺序、每单元 CU 标注、双账号共享纪律、砍单优先级**。所有 GPU 代码已在本地用真实 FLIP2 数据把 CPU 链路调通；GPU 前向逻辑（batching/特殊 token 池化/哈希对齐）已用 CPU 单测验证。

---

## 0. 一次性工作区准备（CPU，≈0 CU）

两个账号共享同一个 Google Drive 目录，嵌入缓存互通，**绝不重复提取**。

1. 账号 1（Owner）在 Google Drive 创建目录 `MyDrive/flip2_benchmark/`，子目录：
   ```
   flip2_benchmark/
   ├── data/raw/          # 官方 .csv.gz（notebook_01 自动下载）
   ├── data/processed/    # 清洗切分 + manifest + 唯一序列注册表
   ├── cache/embeddings/  # 原子 .npz 嵌入缓存（资产）
   └── outputs/{figures,tables}/
   ```
2. 在 `flip2_benchmark/` 根创建 sentinel 文件 `.workspace_id`，内容**恰好一行**：
   ```
   flip2-shared-prod-2026
   ```
   （notebook 在 Colab 下会硬校验此文件；缺失或内容不符立即终止，防止静默落到本地隔离目录。）
3. 把 `flip2_benchmark/` 共享给账号 2（编辑权限）。账号 2 通过“共享给我”挂载后，其逻辑路径仍是
   `/content/drive/MyDrive/flip2_benchmark`（若以快捷方式挂载，notebook 会打印 `os.path.realpath` 解析到的
   `.shortcut-targets-by-id/...` 真实路径——两账号必须指向同一底层存储）。
4. 把三个 `.ipynb` 上传到该目录（或从 GitHub 仓库直接在 Colab 打开）。

**验证共享是否生效**：两账号各开一个 CPU runtime 跑 notebook_01 Cell 1，对比打印的 `physical` 真实路径一致即可。

---

## 1. 执行顺序与 CU 预算

| 步骤 | Notebook / 单元 | Runtime | 预估墙钟 | 预估 CU | 说明 |
|---|---|---|---|---|---|
| 1 | nb01 全部（Cell 1–7） | **CPU** | ~3–5 min | **≈0** | 下载真实切分、清洗、官方切分协议、去重、manifest、one-hot ridge 基线 |
| 2 | nb02 Cell 1–5 + `SMOKE_TEST=True` | **L4** | ~3 min | **~0.3** | 8M 模型 × 10 序列冒烟，验证 GPU 前向/池化/原子写 |
| 3 | nb02 `SMOKE_TEST=False, RUN_PROFILE="l4_small"` | **L4** | ~30–45 min | **~3.5** | ESM-2 8M/35M/150M/650M × (amylase+rhomax) 唯一序列 |
| 4 | nb02 `RUN_PROFILE="g4_large"` | **G4** | ~15–20 min | **~3.5** | ESM-2 3B（仅 G4；≤650M 上 G4 会被防浪费守卫拦截） |
| 5 | nb03 全部（Cell 1–6） | **CPU** | ~10–20 min | **≈0** | ridge 探针 + 3 种子数据效率 + 5×MLP UQ + 图表 |
| | **Pilot 合计** | | ~1–1.5 h | **~7–8 CU** | 与 `flip2_experimental_plan.md` 的 ~7 CU 一致 |

> 研究方案给 pilot 留了 10–15 CU 上限（含调试缓冲）；按本手册纪律实际约 7–8 CU。**先只花这 ~8 CU 出第一张表，看到结果再决定是否投满 W2 的 ~135 CU。**

### 每单元 CU 标注约定
- nb01 / nb03 每个 code 单元标题已标 `[CPU, ~0 CU]`。
- nb02 每个单元标 `[GPU]`；Cell 1 打印探测到的卡级与费率（G4≈10.5、L4≈4.7、T4≈1.8 CU/h），Cell 6 结束打印本次新增缓存数。
- **G4 空转 10 分钟 ≈ 1.75 CU**：数据/缓存/脚本就位后再开 G4，跑完立即 Cell 7 flush+unmount 并断开 runtime。

---

## 2. GPU 会话纪律（nb02）

1. **先冒烟**：`SMOKE_TEST=True` 只用 8M + 10 条序列，写入独立 `cache/embeddings/_smoke_test/`，不污染正式缓存。冒烟通过再改 `False`。
2. **卡级匹配**：
   - L4 → `RUN_PROFILE="l4_small"`（或 `auto`）。
   - G4 → `RUN_PROFILE="g4_large"`（或 `auto`）。`auto` 按显存自动选：≥70GB→3B，20–70GB→8M–650M。
   - 防浪费守卫：G4 上跑 ≤650M 会 `raise`，除非显式 `ALLOW_SMALL_ON_G4=True`。
3. **缓存即资产**：缓存名 `{model}__{dataset}__{unique_sha[:12]}.npz`，已存在则 `[SKIP]`。两账号共享 Drive，账号 1 提过的嵌入账号 2 直接复用。`OVERWRITE_CACHE=False` 默认不重提。
4. **最长序列优先组 batch**：任何 OOM 在第 0 个 batch 立即触发，不会跑到一半才炸。token 预算见 `MODEL_REGISTRY`（含 padding）。
5. **bf16 + 原生 SDPA**：Blackwell（sm_120）安全；`attn_implementation="sdpa"` 失败时自动回退默认注意力。不装 flash-attn 等第三方算子。
6. **原子写**：嵌入先写 `.tmp` 再 `os.replace`，断连/超时不会留下半截缓存。
7. **收尾**：务必跑 Cell 7 `drive.flush_and_unmount()` 再断开，确保缓存落盘到 Drive。

---

## 3. 协议与指标（与官方可比，已在代码中锁定）

精读官方 `J-SNACKKB/FLIP` 的 `baselines/linear_models.py` 后确认：
- **切分**：`fit = set=='train' & ~validation`；`val = set=='train' & validation`（仅选超参）；`test = set=='test'`（最终配置只跑一次）。
- **目标标准化**：`StandardScaler` 在**全部 train**（含 validation 行）上 fit —— 官方约定，代码已复现。
- **one-hot 基线**：官方固定 `alpha=10, tol=1e-5, max_iter=1e6`，one-hot 字母表 `ARNDCQEGHILKMFPSTWYV`，padding 到 train+test 全局最大长度。nb01 Cell 6 逐字复现作 sanity check。
- **NDCG**：`ndcg_score((y_test - y_test.min())[None,:], pred[None,:])`（标准化尺度、min-shift、全量）—— 与官方一致。另报 NDCG@10% 用于 UQ。
- **新颖性缺口确认**：官方 repo 无冻结 ESM-2 多尺度探针、无官方切分上的数据效率曲线（其 `linear_models.py` 的数据效率是**随机切分**且仅 one-hot/似然）、无 UQ/选择性预测、无 3B LoRA。本 pilot 增量落在这些空白。

### 一个必须如实报告的基线失效模式
`amylase/close_to_far` 上 one-hot ridge **输出常数预测**（Spearman=NaN，`pred_constant=True`）。
原因：one-hot 只在训练集中**发生过变异的位点**学到权重；close→far 的远端测试突变落在训练未见变异的位点，权重为 0 → 预测坍缩为常数。
这不是 bug，而是**冻结 PLM 嵌入应当修复的失效模式**（表征可跨位点泛化）。报告保留该切分并如实标注 NaN，不删除。

---

## 4. 余额吃紧时的砍单优先级（研究方案 §3）

按序降级，pilot 核心表不受影响：
1. 砍 **3B LoRA 臂**（W2，~42 CU）。
2. 砍 **TrpB 3B 分层子采样**（~16 CU）。
3. 砍 **650M 以上数据效率曲线**。
4. pilot 的 one-hot + 8M–650M + 3B 冻结探针首表**保留**。

---

## 5. 本地复现（无需 GPU，≈0 CU）

仓库根目录已提供构建器与验证脚本（Windows / Git Bash 示例）：
```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install scipy scikit-learn matplotlib seaborn torch --index-url https://download.pytorch.org/whl/cpu
# 生成三个 notebook（构建器内容已过安全扫描）
python build_nb01.py && python build_nb02.py && python build_nb03.py
# 本地 CPU 端到端验证（one-hot 作为真实表征，无需 GPU 缓存）
python build_nb03.py --test          # 生成根目录 nb03_test.ipynb（one-hot 配置）
./.venv/Scripts/python.exe -m nbconvert --to notebook --execute notebook_01_data_prep_one_hot_baseline.ipynb --output-dir=_local_test --output=nb01_executed.ipynb
./.venv/Scripts/python.exe -m nbconvert --to notebook --execute nb03_test.ipynb --output-dir=_local_test --output=nb03_executed.ipynb
./.venv/Scripts/python.exe verify_nb02_logic.py   # batching/池化/哈希对齐 CPU 单测
```
- 本地 `BASE_DIR=./flip2_benchmark`；Colab 下为 Drive 绝对路径。notebook 自动判别 `IN_COLAB`。
- 交付版 nb03 默认 `MODELS=["one_hot"]+PLM_MODELS`、`UQ_MODELS=["esm2_650m","esm2_3b"]`；缺失缓存的模型自动 `[SKIP]`，因此**先跑 nb02 再跑 nb03** 才能得到 PLM 行。

---

## 6. 故障排查

| 现象 | 处理 |
|---|---|
| `https` 下载被重置 | nb01 自动回退 `http`（固定主机 `flip.protein.properties`，已加同主机重定向限制） |
| sentinel 校验失败 | 确认 `.workspace_id` 内容为 `flip2-shared-prod-2026`，且两账号指向同一 Drive 底层目录 |
| G4 上跑小模型被拦 | 换 L4，或确需时设 `ALLOW_SMALL_ON_G4=True` |
| `np.load` 报 object arrays / allow_pickle | 已修复：缓存 `seq_hashes` 以 `np.str_` unicode 数组保存（兼容 pandas 2/3） |
| Spearman=NaN | 检查 `pred_constant` 列；close_to_far 的 one-hot 常数预测属预期（见 §3） |
| Blackwell 算子报错 | 已用原生 SDPA；如仍异常，回退默认注意力（代码已含 try/except） |
| Ridge `Singular matrix`/`ill-conditioned` 警告 | 高维 one-hot 共线性的良性警告，已在 nb01/nb03 中按消息过滤 |

---

## 7. 引用合规

使用 FLIP2 须同时引用 **Didi et al. (ICML 2026)** 与 **Dallago et al. (NeurIPS 2021 Datasets & Benchmarks)**，
并遵守各数据集 README 的独立署名（CC-BY 4.0 / MIT）。manifest 已记录每个切分的下载 URL、检索时间、原始 SHA-256，供复现与署名。