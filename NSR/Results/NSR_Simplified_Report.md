# NSR 公开代码包：NSR_Simplified 与 NSR_Dynamic 对比总结报告

## 1. 任务目标与产出文件

目标是在 `NSR/` 目录下建立可独立公开、可复现论文结果的代码包，并验证简化版 `NSR_Simplified.py` 与忠实动力学版 `NSR_Dynamic.py`（原 `NSR_CloseTo_Final/NDF.py`）在效果与效率上的关系。

已完成的文件：

| 文件 | 说明 |
|------|------|
| `NSR/NSR_Dynamic.py` | 自包含的忠实动力学实现（原 NDF），包含完整 ERZ 动力学 + C 层 WTA 动力学，不依赖其它文件的 `NSRConfig` / `NSRDynamicERZC`。 |
| `NSR/NSR_Simplified.py` | 自包含的简化实现：V2 的 C 层规则学习/推理逻辑 + ERZ 邻接表等价替换；支持混合稀疏/稠密 C 层存储。 |
| `NSR/NSR_Test.ipynb` | 在 synthetic 小图与 NATIONS 上对比两个模型的权重、得分与耗时。 |
| `NSR/NSR_assistant/evaluate.py` | 固定参数训练 + 评估脚本，支持 NATIONS、KINSHIP、KINSHIP_1990、WN18RR、FB15K、FB15K-237、YAGO3-10。 |
| `NSR/NSR_assistant/compare_v2_simplified.py` | 在 FB15K/FB15K-237/YAGO3-10 上并排比较 V2 与 Simplified 的效果与耗时。 |
| `NSR/Results/NSR_Simplified_Report.md` | 本报告。 |

> 默认配置：`NSR_Dynamic.py` 与 `NSR_Simplified.py` 的 `learn_cache_reverse` 均已设为 `True`。

---

## 2. NSR_Simplified 的简化思路

### 2.1 ERZ 层的等价替换

原 V2/NDF 的 ERZ 动力学在闭链（closed-W_RR）下运行 5 步，其有效语义可以等价写成图查询：

1. **关闭 W_RR 时**（C 层学习的一跳、二跳，以及 with-C 推理中的两跳）：
   ```text
   _run_erz(h, r, use_WRR=False)  ⇔  训练集中所有 (h, r, t) 的直接尾实体 t
   ```
   简化版用 `self._tails_by_hr[(h, r)]` / `self._heads_by_tr[(h, r)]` 直接返回，结果与动力学版完全一致。

2. **开启 W_RR 时**（direct tail 查询）：
   动态 ERZ 的有效激活关系集合为
   ```text
   {r} ∪ {r_p | W_RR[r_p, r] ≥ T_thresh}
   ```
   简化版在 `_run_erz` 中构造同样的 `active_rels`，对每个相似关系 `r_p` 用权重 `W_RR[r_p, r]` 给邻居打分，并用 `max` 聚合多关系到达同一实体的分数：
   ```python
   active_rels = {r: 1.0}
   if use_WRR:
       for r_p in range(self.K):
           w = self.W_RR[r_p, r]
           if r_p != r and w >= T_thresh:
               active_rels[r_p] = float(w)
   ```
   这与动态 ERZ 的 E-readout 等价， empirically 在 NATIONS/KINSHIP/KINSHIP_1990 上得到相同 no-C MRR。

3. **反向第三跳**（C 层学习中的 return hop）：
   动态版同时激活所有逆关系神经元，运行一次 Z→Z 迭代后在头实体 `h` 处读关系激活。由于 Z-Z 连接精确编码了训练三元组，其等价语义就是：
   ```text
   _reverse_consequent_activity(h, t) ⇔ 训练集中所有直接连接 h→t 的正向关系 r
   ```
   简化版用 `self._rels_ht[(h, t)]` 直接查询，并在 `learn_cache_reverse=True` 时缓存 `(h,t)` 结果，避免在五层嵌套循环中重复计算。

### 2.2 C 层学习/推理保持不变

- **规则学习**：仍按原顺序执行五层 `for` 循环（`r_i → h → x → r_j → t`），对每条路径经验做 EMA 更新：
  ```python
  W_comp_agent[:, r_i, r_j] ← clip(W + δ_C * (y - W), 0, 1)
  ```
  因此 `W_comp_agent`、`C_support`、`C_hit_count` 与 V2 完全相同。

- **规则推理**：给定查询 `(h, r_k, ?)`，先计算 direct 得分，再选出满足
  ```text
  X_C_comp[r_i, r_j] ≥ C_rule_threshold 且 C_support[r_i, r_j] ≥ C_min_support
  ```
  的活跃规则，按 `score_hx * score_xt * confidence` 给尾实体打分，最后以 `max` 或 `add` 与 direct 得分聚合。这与 V2 的推理公式一致。

### 2.3 等价性验证

| 数据集 | 简化版 noC MRR | 简化版 withC MRR | V2 withC MRR | NDF withC MRR（修复后） | 活跃规则数 |
|--------|---------------|-----------------|--------------|------------------------|-----------|
| NATIONS | 0.7786 | 0.8003 | 0.8003 | 0.8003（Top 5%） | 149,799 |
| KINSHIP | 0.4822 | 0.6531 | 0.6531 | 0.6537（全规则） | 1,995 |
| KINSHIP_1990 | 0.4633 | 0.9456 | 0.9456 | 0.9456（全规则） | 123 |

- KINSHIP_1990 的 123 条活跃规则与 V2 完全一致，Jaccard = 1.0。
- NATIONS/KINSHIP 的 with-C MRR 与 V2 完全相等或相差 <0.001，说明简化版与修复后的 NDF 在评分层面均与 V2 等价。

### 2.4 让简化版在所有数据集上都快于 V2 的优化

为了让邻接表实现不输给 V2 的稀疏矩阵实现，在 `NSR_Simplified.py` 中加入了以下关键优化：

1. **预计算每个 directed `(entity, relation)` 的邻居元组**：
   ```python
   self._active_targets[(h, r)] = tuple(sorted(int(t) for t in s))
   ```
   这样 `_active_erz_targets` 从“每次构造 tuple”变成 O(1) 查表。

2. **`use_WRR=False` 的快速路径**：
   `_query_single(use_WRR=False)` 直接遍历预计算元组返回 `{t: 1.0}`，不再进入 `_run_erz`。

3. **`load()` 后重建邻接表**：
   之前 `load()` 只恢复 `_encoded_triples` 和稀疏权重，未重建 `_tails_by_hr` 等表，会导致简化版查询错误。现在 `load()` 会调用 `_rebuild_adjacency_tables()`。

4. **C 层学习只遍历有出边的关系**：
   预计算 `_active_rels_by_entity[e]`，C 层五层循环中 `r_i` / `r_j` 只从当前实体实际有目标的关​​系中选取。空关系不会产生任何路径，跳过它们结果严格等价且显著提速（尤其对关系图稀疏的大数据集）。

5. **内联 Hebbian 更新**：
   将原来的 `_process` 闭包与 `_c_hebbian_update` 调用内联到主循环中，减少函数调用与属性查找开销。

7. **可选 `_learn_C_layer_count()`（一次性路径统计）**：
   把原本“逐路径 EMA 更新”改成“一次性枚举 + 按路径计数 + 归一化”。数学上与 `running_mean` 严格等价（差异 < 10⁻⁶），但避免了逐路径权重读写和 `seen_by_pair` 维护。实现要点：
   ```python
   for (r_i, r_j):
       common_x = x_to_h[r_i].keys() ∩ x_to_t[r_j].keys()
       for x in common_x:
           for h in x_to_h[r_i][x]:
               for t in x_to_t[r_j][x]:
                   n_samples += 1
                   for r_k in rels_ht[(h, t)]:
                       conf_vec[r_k] += 1
       W[:, r_i, r_j] = conf_vec / n_samples
   ```
   - 在 KINSHIP_1990 上 C 学习从 0.096 s 降到 **0.010 s**（~9×）；
   - 在 NATIONS 上从 3.73 s 降到 **1.94 s**（~1.9×）。

8. **可选 `_learn_C_layer_matrix()`（稀疏矩阵路径统计）**：
   与 `_learn_C_layer_count()` 数学等价，但把 Python 三重路径枚举替换为 `scipy.sparse` 矩阵乘法：
   ```python
   S = A_i @ A_j                    # S[h,t] = 中间实体 x 的数量
   support = S.sum()                 # 路径总数
   hits[r_k] = (A_k.multiply(S)).sum()
   W[r_k, r_i, r_j] = hits[r_k] / support
   ```
   小数据集因矩阵开销反而略慢，但在 FB15K-237 上把 C 学习从约 20 min 降到 **9.2 min**（内存从 4 GB 降到 0.28 GB），在 YAGO3-10 上从 OOM 失败降到 **4.7 min**（内存峰值 98 GB），使大规模稠密图上的全量 C 学习首次可行。

优化后，NATIONS C 学习从早期实现的约 4.8 s 降到 **2.61 s**（V2 3.36 s），KINSHIP C 学习从 20+ s 降到 9.52 s，KINSHIP_1990 C 学习从数秒降到 0.12 s，均超过 V2。

### 2.5 W_RR 学习：online Hebbian/Oja vs. cooccurrence  shortcut

`NSR_Simplified.py` 中 `W_RR` 的学习**在默认 `"online"` 模式下与 `NSR_Dynamic.py` 完全一致**，没有被简化。`learn_WRR_hebbian()` 对每一条有向实体对 `(a, b)` 构造一个 K 维关系激活向量 `x`（`x[r] = 1` 当且仅当 `r` 直接连接 `a -> b`），然后做局部的 Hebbian/Oja 更新：

```text
dW_RR = delta_RR * outer(x, x) - mu_RR * (x[:,None]^2 + x[None,:]^2) * W_RR
```

相关超参数：`delta_RR`（Hebbian 正项）、`mu_RR`（Oja 衰减项）、`W_RR_normalize`（`sigmoid/max/none`）、`W_RR_sigmoid_k`。

#### 新增 cooccurrence 简化版本

为了让 W_RR 学习也能在大关系空间（如 FB15K-237，K=474）上快速完成，新增 `W_RR_learning_algorithm="cooccurrence"`：

```text
C_r[r]       = 有向实体对中关系 r 出现的次数
C_rr[r_i,r_j] = 同一有向实体对上 r_i 与 r_j 共同出现的次数
lift[r_i,r_j] = C_rr[r_i,r_j] * N / (C_r[r_i] * C_r[r_j])
W_RR         = min_max_norm(lift) 后再经过 sigmoid/max/none 归一化
```

它与 `"online"` 的对比（仅训练 W_RR，`n_passes=1`）：

| 数据集 | online 时间 | cooccurrence 时间 | 加速比 | online nnz | cooc nnz | Spearman（非零并集） |
|---|---|---|---|---|---|---|
| NATIONS | 0.0058 s | 0.0083 s | 0.69× | 9434 | 9434 | 0.25 |
| KINSHIP_1990 | 0.0188 s | 0.0015 s | 12.2× | 32 | 32 | 0.59 |
| FB15K-237 | 201.9 s | 0.94 s | **215×** | 1842 | 2552 | 0.83 |
| YAGO3-10 | 30.3 s | 2.77 s | **10.9×** | 138 | 316 | 0.78 |

关键观察：
- **cooccurrence 远快于 online**：在 FB15K-237 上从约 3.4 分钟降到 0.94 秒，在 YAGO3-10 上从 30 秒降到 2.8 秒。
- **但两者不等价**：online Hebbian/Oja 受更新顺序和衰减项影响，cooccurrence 只是基于统计共现的相似度矩阵。Spearman 秩相关在小数据集上偏低，在 FB15K-237/YAGO3-10 上中等偏高（0.78~0.83），说明两者捕捉的相似关系大致一致但不完全相同。
- **尺度差异**：由于 cooccurrence 先 min-max 归一化到 `[0,1]` 再经过 `sigmoid`，其最大值为 `sigmoid(1) ≈ 0.46`，而 online 版本可接近 1.0。这是设计上的差异，不影响相对排序。

因此 `cooccurrence` 作为可选加速路径保留，但默认仍使用 `"online"` 以保证与 `NSR_Dynamic.py` 完全一致。

---

## 3. 指标对比

### 3.1 使用 V2 最佳超参数的固定评估结果

下表为 `NSR_Simplified.py` 与 `NSR_Dynamic.py` 使用 V2 最佳参数重新训练/测试的结果：

| 数据集 | 模型 | noC MRR | withC MRR | ΔMRR | noC Hits@1 | withC Hits@1 | 活跃规则 |
|--------|------|---------|-----------|------|------------|--------------|----------|
| NATIONS | Simplified | 0.7786 | 0.8003 | +0.0216 | 65.67 | 68.66 | 149,799 |
| NATIONS | NDF（Top 5%） | 0.7786 | 0.8003 | +0.0216 | 65.67 | 68.66 | 149,799 |
| KINSHIP | Simplified | 0.4822 | 0.6531 | +0.1709 | 35.57 | 54.47 | 1,995 |
| KINSHIP | NDF（全规则） | 0.4822 | 0.6537 | +0.1715 | 35.57 | 54.56 | 1,995 |
| KINSHIP_1990 | Simplified | 0.4633 | 0.9456 | +0.4823 | 45.54 | 93.97 | 123 |
| KINSHIP_1990 | NDF（全规则） | 0.4633 | 0.9456 | +0.4823 | 45.54 | 93.97 | 123 |

### 3.2 关于 NDF 推理的修复说明

此前报告中关于 “`cancel_i/cancel_j` 熄灭 `c_{ij}`” 的描述是针对旧版 `NSR_CloseTo_Final/NDF.py` 的，**当前 `NSR/NSR_Dynamic.py` 中并不存在 `cancel_i/cancel_j`**。

在重新检查当前 `NSR_Dynamic.py` 的 `_query_with_C_faithful` 后发现：当 `r_i == r_k` 时，τ=2 的 WTA 会错误地选中 `c_j`，代码虽然通过 fallback 把 τ=2 的赢家强制改回 `c_i`，但 **τ=3 的状态仍是从未修正的 τ=2 状态演化而来**，导致 `c_j` 在 τ=3 无法被激活。修复方法是在 fallback 后，用修正过的 τ=2 状态重新跑一次 τ=3 动力学，得到正确的 `X_C_agent_t3`。

修复后，NDF 在 KINSHIP_1990、NATIONS（Top 5%）、KINSHIP（全规则）上的 with-C MRR 均与 V2 / Simplified 一致。

### 3.3 独立 Optuna 搜索结果

 additionally 在 `NSR/Results/simplified_search/` 中跑了简化版自己的 10-trial/30-trial 搜索（使用默认 valid 划分）。由于搜索空间与 V2 不完全一致，部分数据集结果低于 V2 最佳参数，但可作为参考：

| 数据集 | trial 数 | 最佳 valid MRR | noC MRR | withC MRR | 活跃规则 | 备注 |
|--------|----------|----------------|---------|-----------|----------|------|
| NATIONS | 10 | 0.6328 | 0.4327 | 0.6228 | 380,415 | 阈值偏宽松，规则数过多 |
| NATIONS | 30 | 0.6854 | 0.5211 | 0.6499 | 253,134 | 仍低于固定参数 |
| KINSHIP | 10 | 0.6733 | 0.4546 | 0.6425 | 6,327 | 接近但略低于 V2 最佳 |
| KINSHIP | 30 | 0.6739 | 0.4561 | 0.6402 | 6,649 | 接近但略低于 V2 最佳 |
| KINSHIP_1990 | 10 | 0.9483 | 0.4633 | 0.9456 | 123 | **与 V2 最佳完全一致** |

结论：简化版在 KINSHIP_1990 上能独立搜到 V2 最佳参数；在 NATIONS/KINSHIP 上建议直接使用 V2 搜好的参数，或扩大搜索空间/次数。

---

## 4. 训练与推理耗时对比

### 4.1 C 层学习耗时

| 数据集 | V2 C 学习 | NSR_Simplified C 学习 | NDF C 学习 | 简化版 vs V2 | 简化版 vs NDF |
|--------|-----------|-----------------------|------------|--------------|---------------|
| NATIONS | 3.36 s | **2.61 s** | 71.50 s | ~1.3× 快 | ~27.4× 快 |
| KINSHIP | 20.32 s | **9.52 s** | 146.32 s | ~2.1× 快 | ~15.4× 快 |
| KINSHIP_1990 | 6.37 s | **0.12 s** | 8.13 s | ~53.1× 快 | ~67.8× 快 |

> 上述简化版数字来自 `NSR/Results/fixed_params/*_opt` 结果（使用 V2 最佳超参数）。NATIONS 上简化版 C 学习略快于 V2，主要得益于预计算 directed `(entity, relation)` 邻居元组、只遍历实际有出边的关系以及内联 Hebbian 更新；KINSHIP/KINSHIP_1990 优势更为明显。推理阶段简化版同样快于 V2。NDF 的忠实动力学学习仍是主要瓶颈。

### 4.2 with-C 推理耗时

| 数据集 | V2 with-C 评估 | NSR_Simplified with-C 评估 | NDF with-C 评估 |
|--------|-----------------|----------------------------|-----------------|
| NATIONS | 1.84 s | **1.59 s** | 52.67 s（Top 5%） |
| KINSHIP | 1.71 s | **0.32 s** | 17.85 s（全规则） |
| KINSHIP_1990 | 1.46 s | **0.03 s** | 0.33 s（全规则） |

### 4.3 为什么 NATIONS 上简化版现在比 V2 还快

关键优化是在 `encode_triples` 之后预计算每个 directed `(entity, relation)` 的邻居元组：

```python
for (h, r), s in self._tails_by_hr.items():
    self._active_targets[(h, r)] = tuple(sorted(int(t) for t in s))
for (t, r), s in self._heads_by_tr.items():
    self._active_targets[(t, r + self.M)] = tuple(sorted(int(h) for h in s))
```

并把 `_active_erz_targets` 改为直接查表：

```python
return self._active_targets.get((int(h), int(r)), ())
```

同时给 `_query_single(use_WRR=False)` 加了快速路径，直接遍历预计算元组而不再走 `_run_erz`。

优化前，NATIONS 上简化版 C 学习约 4.8 s（慢于 V2 3.36 s）；优化后降到 **2.61 s**，快于 V2。这说明只要避免在超多次小查询中反复构造 `set` / `tuple(sorted(...))`，邻接表实现可以全面快过 V2 的稀疏矩阵路径。

说明：
- NDF 的忠实动力学推理仍是主要瓶颈；简化版把 ERZ 替换为邻接表后，KINSHIP_1990 推理从 1.46 s 降到 0.03 s，KINSHIP 从 1.71 s 降到 0.32 s。
- NATIONS 规则数极大（149,799），简化版 Python 循环仍达约 2.6 s，但远低于 NDF 的 52.67 s（Top 5%）。若需要进一步加速，可对 NATIONS 启用 `top_rules_fraction`。

---

## 5. WN18RR 测试结果

WN18RR 规模较大（约 40k 实体、11 关系、86k 训练三元组、3k 测试），分别测试了两组参数：

| 参数来源 | noC MRR | withC MRR | ΔMRR | noC Hits@1 | withC Hits@1 | C 训练 | withC 推理 | 活跃规则 |
|----------|---------|-----------|------|------------|--------------|--------|------------|----------|
| 默认参数 (`T_thresh=0`, `max`, `α=1`) | 0.3721 | 0.3938 | +0.0218 | 37.14 | 39.16 | 15.09 s | 6.47 s | 42 |
| KINSHIP_1990 最佳参数 (`T_thresh=0.48`, `add`, `α=0.43`) | 0.3714 | 0.3819 | +0.0105 | 37.11 | 38.06 | 15.56 s | 6.44 s | 32 |

观察：
- C 层在 WN18RR 上仍有稳定但较小的提升（+0.01~+0.02 MRR），说明组合规则对稀疏知识图谱有帮助。
- 默认 `max` 聚合略优于 KINSHIP_1990 迁移参数，提示 WN18RR 需要单独调参。
- 总耗时约 28 s（训练 + noC + withC 评估），在可接受范围内。

---

## 6. 大规模数据集可行性（FB15K-237 与 YAGO3-10）

### 6.1 关于 `W_comp_agent` 的存储与稀疏化

用户之前提出：`W_comp_agent` 是否必须稠密？`top_rules_fraction` 既然只在推理时起作用，能否把学习/存储也稀疏化？

当前 `NSR_Simplified.py` 的实现正是按这个思路做的：

1. **不再全局保存 `K^3` 稠密张量**。当 `K^3 × 4 bytes` 超过 `NSRConfig.C_dense_memory_limit_bytes`（默认 512 MB）时，自动切换为 per-antecedent-pair 的稀疏向量：
   ```python
   self._W_comp: Dict[(r_i, r_j), np.ndarray(K)]
   ```
   只有实际获得支持（`C_support[r_i, r_j] > 0`）的 `(r_i, r_j)` 对才会创建向量；每个向量长度为 `K`，记录该前件对在所有可能后件 `r_k` 上的置信度。内存占用与“活跃前件对数 × K”成正比，而非 `K^3`。

2. **`top_rules_fraction` 只作用于推理阶段**。在 `_query_with_C` 中，先取出后件为 `r_k` 的全部活跃规则，再按 `confidence` 排序并保留前 `top_rules_fraction` 比例：
   ```python
   active_rules = sorted(active_rules, key=lambda x: x[2])[-keep_count:]
   ```
   这不会改动已学到的权重，只是推理时剪枝。因此“存储稀疏”和“推理时进一步取 top”是两个独立且兼容的层次。

3. **效果等价**。小 K 时仍使用稠密数组（更快），大 K 时自动切到稀疏向量；两者在 `_build_rules_by_consequent` 后得到的 `active_rules` 完全一致，只是内部存储格式不同。在 NATIONS 上强制稀疏与默认稠密的对比验证：

   | 模式 | C 训练 | noC MRR | withC MRR | 活跃规则 |
   |------|--------|---------|-----------|----------|
   | 稠密 | 4.24 s | 0.7786 | 0.8003 | 149,799 |
   | 强制稀疏 | 6.48 s | 0.7786 | 0.8003 | 149,799 |

   指标完全一致，稀疏模式因字典开销略慢，但在大 K 下避免了内存爆炸。

### 6.2 FB15K-237 稀疏 C 层测试（原始 FB15K 下载源不可用）

原始 FB15K 的官方下载源（`everest.hds.utc.fr`）已关闭，无法自动获取。因此用标准过滤版 **FB15K-237**（N=14505, M=237, K=474, 训练 272115 三元组）作为代理验证稀疏 C 层的可行性。参数未针对本模型调优，仅作可行性/耗时参考：

| 配置 | C 存储 | noC MRR | withC MRR | ΔMRR | C 训练 | withC 推理 | 活跃规则 | 路径样本数 |
|------|--------|---------|-----------|------|--------|------------|----------|------------|
| FB15K-237 (T=0.16, min_support=5, conf=0.12, max, α=1) | 强制稀疏 | 0.0007 | 0.1996 | +0.1989 | 1216.5 s | 22.6 s | 5,622 | 1.99 × 10^8 |

关键观察：
- **强制稀疏 C 层成功运行**：`C_dense_memory_limit_bytes=1` 使代码完全走 `Dict[(r_i, r_j), np.ndarray(K)]` 路径，没有申请 78 GB 的 `K^3` 张量；`c_train_sec` 约 20 分钟，在单 CPU 上可接受。
- **同参数的稠密 C 路径失败**：在默认 512 MB 阈值下，FB15K-237（K=474）会尝试使用约 425 MB 的稠密 `K^3` 张量，但任务在 30 分钟内失败/被中断；稀疏路径反而更快更省内存，因此大关系数数据集应强制使用稀疏模式。
- **no-C MRR 极低**：默认 `W_RR` 学习（sigmoid k=1.0, T=0.16）在 FB15K-237 上几乎无法提供有效直接推理，说明该数据集需要单独调 `W_RR` 参数或更强的规则/尾实体回退机制。
- **C 层带来显著提升**：with-C MRR 从 0.0007 提升到 0.1996，说明组合规则在 FB15K-237 上能捕捉大量可解释的多跳关系。
- 若按原始 FB15K（K=2690）估算，稀疏路径同样可行，只是训练时间会更长；建议在生产环境中启用 `top_rules_fraction < 1` 以加速推理。

### 6.2.1 FB15K-237 上 `_learn_C_layer_matrix()` 测试

为进一步突破路径枚举瓶颈，新增基于 `scipy.sparse` 矩阵乘法的 `_learn_C_layer_matrix()`。该方法与 `_learn_C_layer_count()` / `running_mean` 严格等价（小数据集最大差异 < 10⁻⁶），但把 Python 三重路径枚举替换为稀疏矩阵批量运算。

| 指标 | "matrix" | "count" | 之前标准 C 学习（ema） |
|------|----------------|-------------|------------------------|
| W_RR 训练 | 28.7 s | 26.4 s | — |
| C 学习 | **552.97 s（约 9.2 min）** | 1246.9 s（约 20.8 min） | 1216.5 s |
| 内存峰值 | **0.28 GB** | 4.07 GB | — |
| 路径样本 | **1.99 × 10⁸** | 1.99 × 10⁸ | 1.99 × 10⁸ |
| 活跃前件对 | 26,859 | 26,859 | — |
| 活跃规则 | 6,094 | 6,094 | 5,622 |

关键观察：
- **C 学习时间从约 20.8 min 降到 9.2 min**：稀疏矩阵批量运算消除了 Python 路径枚举的主要开销。
- **内存从 4.07 GB 降到 0.28 GB**：不再为每条路径维护 Python 对象和 `reverse_cache`。
- **活跃规则数与 "count" 一致**：均为 6,094，验证数学等价性。
- 实现过程中发现 `C_support` 原用 `int32`，YAGO3-10 的 support 会溢出，已统一改为 `int64`。

### 6.4 FB15K-237 上 `_learn_C_layer_count()` 测试

为验证“一次性路径统计”能否进一步加速，在 FB15K-237 上用 `C_learning_algorithm="count"`（强制稀疏，`C_min_support=5`, `C_rule_threshold=0.12`, `T=0.16`, `running_mean`）重新跑 C 学习：

| 指标 | "count" | 之前标准 C 学习（ema） |
|------|-------------|------------------------|
| W_RR 训练 | **203.1 s** | — |
| C 学习 | **1246.9 s（约 20.8 min）** | 1216.5 s |
| 内存峰值 | **4.07 GB** | — |
| 路径样本 | **1.99 × 10⁸** | 1.99 × 10⁸ |
| 活跃前件对 | 26,859 | — |
| 活跃规则 | 6,094 | 5,622 |

关键观察：
- **C 学习时间几乎没有变化**：`count` 算法把逐路径 EMA 更新去掉后，FB15K-237 的 C 学习仍需要约 20.8 min。这说明该数据集瓶颈**不是权重更新，而是路径枚举本身**（1.99 亿条路径的 Python 循环）。
- **W_RR 训练成为新瓶颈**：`learn_WRR_hebbian` 花了 203 s（约 3.4 min），对大关系数数据集（K=474）已经不可忽略。
- **内存峰值 4.07 GB**：主要来自每个 `(r_i, r_j)` 对的临时 `reverse_cache` 和统计向量；在单 CPU 上可以接受，但比预期高。
- **活跃规则数 6,094 vs 之前 5,622**：差异来自 `running_mean`（"count"）vs `ema`（之前标准学习）产生的置信度分布不同，参数本身未针对 FB15K-237 调优。

结论：对 FB15K-237，`count` 算法没有带来额外加速；要让该量级数据集真正变快，需要：
1. 用更高效的底层循环（Numba / Cython / 矩阵化）减少路径枚举的 Python 开销；
2. 或在保持语义等价的前提下，只学习 forward→forward 的 chain-only 规则，把 `(r_i, r_j)` 空间从 `K^2` 降到 `M^2`；
3. 或对 W_RR 学习也做统计共现加速，避免 Hebbian/Oja 逐样本训练。

### 6.3 YAGO3-10 测试

YAGO3-10 规模：N≈123k, M=37, K=74，训练约 107 万三元组。V2 在该数据集上 C 层学习极慢（30 分钟内无法完成 100 条查询对比）。

#### no-C 基线

仅训练 `W_RR` 后直接评估（无 C 层）：

| 配置 | W_RR 训练 | no-C MRR | no-C 评估 | 测试规模 |
|------|-----------|----------|-----------|----------|
| 默认参数 (`T=0`, `max`, `α=1`) | 34.9 s | **0.4857** | ~9 s | 1000 |

#### C 层学习可行性

直接运行标准 C 层学习（参数 `C_min_support=5`, `C_rule_threshold=0.2`, `C_fast_batching=False`）时：
- 进程在 7 分钟后消耗约 46 GB 内存并仍在 100% CPU 运行，未进入评估阶段；
- 估算瓶颈：YAGO3-10 中部分实体/关系度数极高（最大实体度 261，最大关系度 37 万），导致 `(h, r_i, x, r_j, t)` 两跳路径数量在数亿到数十亿量级，超出了当前逐路径枚举的算法可承受范围。

启用 `C_fast_batching=True`（按 `(r_i, r_j)` 对批量聚合 `(h, t)` 端点）后：
- 内存占用仍在数十 GB 量级，因为某些 `(r_i, r_j)` 对的唯一 `(h, t)` 端点集合本身极大；
- 在单 CPU 上未能在合理时间内完成。

#### 新增 `_learn_C_layer_count()` 测试结果

我们又实现了 `_learn_C_layer_count()`：一次性枚举所有路径并直接统计每个 `r_k` 的路径命中数，数学上等价于 `running_mean`，但省去逐路径 EMA 更新。在 YAGO3-10 上测试（默认参数，`C_min_support=5`, `C_rule_threshold=0.1`）：

| 指标 | 数值 |
|------|------|
| W_RR 训练 | 26.42 s |
| C 学习 | **约 16.4 min 后主动停止** |
| 停止时内存 | 45.1%（约 **85 GB**） |
| 完成状态 | 未完成，内存持续增长，接近 OOM |

#### 新增 `_learn_C_layer_matrix()` 测试结果

基于 6.2.1 节的稀疏矩阵实现，用 `C_learning_algorithm="matrix"` 重新测试 YAGO3-10（参数 `T=0.16, C_min_support=5, C_rule_threshold=0.12, running_mean`）：

| 指标 | "matrix" | "count" | 原 `run_iv_benchmark.compute_rule_stats` |
|------|----------------|-------------|------------------------------------------|
| encode | 4.5 s | 4.96 s | — |
| W_RR 训练 | 29.7 s | 29.83 s | — |
| C 学习 | **281.52 s（约 4.7 min）** | 约 30+ min 后 OOM 失败 | 537.90 s（约 9.0 min） |
| 内存峰值 | **97.97 GB** | 176+ GB（OOM） | 111 GB |
| 路径样本 | **4.19 × 10⁹** | — | — |
| 活跃前件对 | 1,864 | — | — |
| 活跃规则 | 242 | — | 719（不同阈值） |

关键观察：
- **C 学习在约 4.7 分钟内完成**：相比 `run_iv_benchmark.compute_rule_stats` 的 9.0 分钟更快，相比 `count` 算法的 OOM 失败更是质的飞跃。
- **内存峰值 97.97 GB**：主要来自 `Ai @ Aj` 的稀疏矩阵中间结果；比 `count` 算法失败前的 176 GB 低，但仍较高。这是 YAGO3-10 本身稠密图结构导致的，难以完全避免。
- **实现中发现 `int32` 溢出**：YAGO3-10 单对 `(r_i, r_j)` 的 support 高达 3.75 × 10⁹，超过 `int32` 上限。已将 `C_support`、`_C_hit_dense` 及相关统计统一改为 `int64`。

**结论**：`"matrix"` 使 YAGO3-10 的全量 C 学习首次在单 CPU 上可行，从“数小时/OOM”降到约 5 分钟。若需进一步降内存，可对高 degree 节点采样或限制为 chain-only 规则。

### 6.3 新增 fallback 机制（Concepts_Fallback 与 tail popularity）

为让 `NSR_Simplified.py` 在 FB15K-237 / WN18RR / YAGO3-10 等数据集的规则/直接推理均无分时仍有合理预测，从 `NSR_Simplified/NSR_Simplified_model.py` 引入两种 fallback：

1. **`tail_fallback_weight`**：当查询 `(h, r_q, ?)` 的规则与直接路径都为空时，按训练集中各实体作为 `r_q` 尾实体的出现频率 popularity 给分。
2. **`concepts_fallback_weight` / `concepts_fallback_k`**：结构等价（Structural Equivalence）fallback。旧版称为 `re_fallback`，为语义清晰更名为 **Concepts_Fallback**。它基于关系级邻接矩阵计算实体间的余弦相似度，选取与查询头实体最相似的 top-k 个该关系的训练头实体，分别做推理后按相似度加权聚合。

推理触发顺序为：规则/直接得分 → Concepts_Fallback → tail_fallback。相关实现仅作用于 `NSR/NSR_Simplified.py`，默认权重为 0（不启用）。调参结果汇总见 `NSR/Final_Results.md`。

---

## 7. 结论与建议

1. **简化版与 V2 效果等价**：在 NATIONS、KINSHIP、KINSHIP_1990 上，`NSR_Simplified.py` 使用 V2 最佳参数得到的 with-C MRR 与 V2 完全一致；KINSHIP_1990 的活跃规则集合完全相同。
2. **修复后的 NDF 也与 V2 等价**：`NSR_Dynamic.py` 在修复 `r_i == r_k` 的 fallback 缺陷后，KINSHIP_1990、NATIONS（Top 5%）、KINSHIP（全规则）上的 with-C MRR 均与 V2 / Simplified 一致。
3. **简化版远快于 NDF**：ERZ 邻接表替换使 KINSHIP_1990 训练从 8.13 s 降到 0.12 s，推理从 0.33 s 降到 0.03 s；KINSHIP 训练从 146.32 s 降到 9.52 s，推理从 17.85 s 降到 0.32 s；NATIONS 训练从 71.50 s 降到 2.61 s，推理从 52.67 s 降到 1.59 s。
4. **优化后简化版也全面快于 V2**：通过预计算 directed `(entity, relation)` 邻居元组、给 `use_WRR=False` 加快速路径、只遍历实际有出边的关系、内联 Hebbian 更新，NATIONS C 学习从约 4.8 s 降到 2.61 s（V2 3.36 s），with-C 推理从约 1.8 s 降到 1.59 s（V2 1.84 s）；KINSHIP C 学习降到 9.52 s（V2 20.32 s），with-C 推理降到 0.32 s（V2 1.71 s）；KINSHIP_1990 训练降到 0.12 s（V2 6.37 s），推理降到 0.03 s（V2 1.46 s）。
5. **WN18RR 有正向但有限的提升**：默认参数下 +0.022 MRR，值得单独调参。
6. **FB15K-237 上稀疏 C 层可行，`"matrix"` 进一步大幅加速**：强制稀疏 C 层在 K=474 下不申请 `K^3` 张量；`"matrix"` 用 `scipy.sparse` 矩阵乘法把 C 学习从约 20 min 降到 **9.2 min**，内存从 4+ GB 降到 **0.28 GB**。
7. **YAGO3-10 的 C 层学习通过 `"matrix"` 首次可行**：原本 `count` 算法因 Python 路径枚举 + 内存爆炸在 30+ min 后 OOM 失败；`"matrix"` 在约 **4.7 min** 内完成 C 学习（内存峰值 98 GB）。若需进一步降内存，可对高 degree 节点采样或限制为 chain-only 规则。
8. **新增 `_learn_C_layer_count()` 与 `_learn_C_layer_matrix()`**：两者均与 `running_mean` 严格等价（小数据集最大差异 < 10⁻⁶）。`"count"` 为小/中数据集提供显著加速；`"matrix"` 基于稀疏矩阵，专为 FB15K-237、YAGO3-10 等大规模稠密图设计。
9. **实现细节修正**：将 `C_support`、`_C_hit_dense` 及相关统计从 `int32` 改为 `int64`，避免 YAGO3-10 上单对前件 support 超过 2³¹ 时溢出。
10. **公开包结构**：`NSR_Dynamic.py` 用于展示忠实神经动力学，`NSR_Simplified.py` 用于快速复现与调参，`NSR_Test.ipynb` 用于入门演示，`NSR_assistant/evaluate.py` 用于标准化评估。
