# NSR 公开代码包说明

本目录（`NSR/`）存放论文《Neural Structural Reason》计划公开的精简代码包。所有核心实现均自包含，不依赖 `NSR_CloseTo_Final/` 或 `NSR_Simplified/` 中的旧文件。

> **维护规则**：每次新增、删除或修改本目录下的文件，都必须同步更新本 `Introduction.md` 中对应条目，避免后续混淆。

---

## 1. 核心模型文件

### `NSR_Dynamic.py`
- **作用**：忠实的神经动力学实现（对应原 `NSR_CloseTo_Final/NDF.py`）。
- **内容**：包含完整的 ERZ 五层动力学、C Agent / C Composition WTA 动力学、Hebbian/Oja 学习、C 层规则学习，以及查询接口。
- **特点**：
  - 完全自包含，`NSRConfig` 与 `NSRDynamicERZC` 均定义在本文件中。
  - 动力学细节与论文公式一致，适合用于展示模型的神经可解释性。
  - 推理速度较慢，尤其在大关系数数据集上。
- **使用**：
  ```python
  from NSR_Dynamic import NSRConfig, NSRDynamicERZC
  cfg = NSRConfig(T_thresh=0.0, W_RR_sigmoid_k=1.0, ...)
  model = NSRDynamicERZC(N, M, cfg)
  model.encode_triples(train_triples)
  model.learn_WRR_hebbian()
  model.learn_C_layer()
  scores = model.query(h, r, use_C_layer=True, agg_mode="add")
  ```

### `NSR_Simplified.py`
- **作用**：效果与 `NSR_Fake_C_Dynamic_V2.py` 等价的快速实现。
- **内容**：保留 V2 的 C 层规则学习/推理逻辑，但把 ERZ 动力学替换为等价的邻接表查询。
- **特点**：
  - 完全自包含。
  - 与 V2 学到的 `W_comp_agent`、活跃规则集合、查询得分一致（已在 KINSHIP_1990 等数据集验证）。
  - **混合稀疏/稠密 C 层存储**：当 `K^3` 稠密张量超过 `NSRConfig.C_dense_memory_limit_bytes`（默认 512 MB）时，自动切换为 per-antecedent-pair 的稀疏向量存储，使 FB15K-237 等关系数极大的数据集也能运行 C 层；小数据集仍使用完整稠密张量以保证速度。
  - **C 层学习算法统一入口**：`NSRConfig.C_learning_algorithm`（ `"online" | "count" | "matrix"` ）作为顶层互斥开关；`learn_C_layer()` 根据该字段分发到三个内部方法：
    - `"online"`：标准逐路径 Hebbian/running_mean/EMA 更新；
    - `"count"`：一次性路径计数（Python 显式枚举），与 `running_mean` 严格等价；
    - `"matrix"`：一次性路径计数（`scipy.sparse` 矩阵乘法），与 `"count"` 严格等价，但更适合 FB15K-237 / YAGO3-10 等大规模稠密图。
  - **次级开关仅在 `"online"` 下生效**：`C_fast_batching`、`C_hebbian_mode`、`delta_C`、`C_n_passes`、`learn_cache_reverse`。
  - **横切开关**：`C_dense_memory_limit_bytes` 控制内部用稠密 `K³` 张量还是 `Dict[(r_i,r_j), vec]` 稀疏存储，与学习算法独立；`C_min_support`、`C_rule_threshold` 在所有算法中用于规则抽取。
  - **W_RR 学习算法**：`W_RR_learning_algorithm`（`"online" | "cooccurrence"`）作为顶层开关。`"online"` 是标准逐样本 Hebbian/Oja 更新（超参数 `delta_RR`、`mu_RR`、`W_RR_normalize`、`W_RR_sigmoid_k`）；`"cooccurrence"` 从关系共现矩阵直接构造 `W_RR`，速度远快于 `"online"`（FB15K-237 上约 215×），但与 `"online"` 不等价。保留向后兼容别名 `W_RR_fast=True -> "cooccurrence"`。
  - **Fallback 机制**：整合自 `NSR_Simplified/NSR_Simplified_model.py`，用于在规则/直接推理无分时补充候选：
    - `tail_fallback_weight`：按关系 `r_q` 的尾实体 popularity 给分；
    - `concepts_fallback_weight` / `concepts_fallback_k`：结构等价（Structural Equivalence）fallback，原名 `re_fallback`，为与旧版区分更名为 Concepts_Fallback；取与查询头实体最相似的 top-k 个训练头实体做推理并加权聚合。
  - 废弃别名 `C_fast_approx` / `C_fast_approx_v2` 仍保留向后兼容，分别等价于 `"count"` / `"matrix"`。
  - 训练/推理速度显著快于 `NSR_Dynamic.py`。
- **使用**：同 `NSR_Dynamic.py`，类名与接口相同。

---

## 2. 测试与演示

### `NSR_Test.ipynb`
- **作用**：Jupyter Notebook 入门演示。
- **内容**：
  - 构造 synthetic 小图，验证 `NSR_Dynamic.py` 与 `NSR_Simplified.py` 学到的权重一致。
  - 在 NATIONS 上对比两个模型的 direct / with-C 得分与耗时。
  - 展示 W_RR 与 C 层学习算法的配置层级。
- **使用**：在 `NSR/` 目录下启动 Jupyter，依次执行 cell。

### `Test_On_KGs.ipynb`
- **作用**：多数据集 Optuna 超参数搜索示例（参考 `NSR_Simplified/Test_On_KGs.ipynb`）。
- **内容**：
  - 在 WN18RR、FB15K-237、YAGO3-10、KINSHIP、KINSHIP_1990_EXTENDED、NATIONS 上运行 `NSR_Simplified.py`。
  - 缓存 C 层学习结果，只调 W_RR + fallback 超参数。
  - 汇总 MRR / Hits@1/3/10。
- **使用**：在 `NSR/` 目录下启动 Jupyter，依次执行数据集 cell。

---

## 3. 辅助评估脚本

### `NSR_assistant/evaluate.py`
- **作用**：固定超参数下训练并评估 `NSR_Dynamic.py` 或 `NSR_Simplified.py`。
- **支持数据集**：NATIONS、KINSHIP、KINSHIP_1990、WN18RR、FB15K、FB15K-237、YAGO3-10。
- **常用命令示例**：
  ```bash
  # 从项目根目录执行
  python NSR/NSR_assistant/evaluate.py \
    --model-file NSR/NSR_Simplified.py \
    --dataset KINSHIP_1990 \
    --params-json NSR/Results/fixed_params/kinship1990_v2_best.json \
    --output-dir NSR/Results/fixed_params/kinship1990 \
    --kinship1990-dir KINSHIP_1990
  ```
- **输出**：`{output-dir}/{dataset_lower()}_result.json`，包含训练耗时、C 层统计、no-C / with-C 指标、评估耗时。

### `NSR_assistant/compare_v2_simplified.py`
- **作用**：在 FB15K / FB15K-237 / YAGO3-10 上并排比较 `NSR_Fake_C_Dynamic_V2.py` 与 `NSR_Simplified.py` 的训练与评估指标、耗时。
- **特点**：
  - FB15K（K=2690）上 V2 因稠密 `K^3` 内存限制跳过 C 层，Simplified 使用混合稀疏 C 层仍可运行。
  - 通过 `--max-test-queries` 控制测试规模，便于在大数据集上快速对比。
- **常用命令示例**：
  ```bash
  cd NSR/NSR_assistant
  python compare_v2_simplified.py \
    --dataset YAGO3-10 \
    --max-test-queries 1000 \
    --v2-model-file ../../NSR_CloseTo_Final/NSR_Fake_C_Dynamic_V2.py \
    --simplified-model-file ../../NSR/NSR_Simplified.py \
    --output-dir ../../NSR/Results/fb15k_yago
  ```
- **输出**：`{output-dir}/{dataset}_comparison.json`。

### `NSR_assistant/tune_wrr_with_cached_c.py`
- **作用**：在 FB15K-237 / WN18RR / YAGO3-10 等大数据集上只调 `NSR_Simplified.py` 的 W_RR + fallback 超参数，C 层只学一次并缓存。
- **特点**：
  - C 层使用 `_learn_C_layer_matrix()`（`C_learning_algorithm="matrix"`），并强制使用稀疏 per-pair 存储。
  - 使用 Optuna 搜索 `W_RR_learning_algorithm`、`T_thresh`、`alpha`、`agg_mode`、`tail_fallback_weight`、`concepts_fallback_weight` 等。
  - 将结果追加到 `NSR/Final_Results.md`。
- **常用命令示例**：
  ```bash
  python NSR/NSR_assistant/tune_wrr_with_cached_c.py \
    --dataset FB15K-237 --n-trials 20
  ```

### `NSR_assistant/tune_dynamic.py`
- **作用**：对 `NSR_Dynamic.py`（忠实动力学版）做完整训练 + 超参数搜索，适用于 NATIONS / KINSHIP / KINSHIP_1990 等小数据集。
- **特点**：
  - 每 trial 都重新训练 W_RR 和 C 层。
  - 搜索 `T_thresh`、`W_RR_sigmoid_k`、`C_activity_threshold`、`C_rule_threshold`、`C_min_support`、`alpha`、`agg_mode`。
  - **2026-07-23 更新**：额外搜索 `top_rules_fraction`（0.01~1.0，NATIONS 上限制为 0.05~0.30，否则 faithful dynamics 推理会超时）。NATIONS 等活跃规则极多的数据集若保留全部规则，忠实动力学推理会超时；通过 `top_rules_fraction` 剪枝可在速度/coverage 之间权衡。
  - 将结果追加到 `NSR/Final_Results.md`。
- **常用命令示例**：
  ```bash
  python NSR/NSR_assistant/tune_dynamic.py \
    --dataset KINSHIP_1990 --n-trials 20
  ```

### `Interpretability/`
- **作用**：对 NSR 与 NeuralLP 做“单事实学习更新局部性”评估，量化一个新三元组对规则权重和内部学习坐标的影响是否局部、可审计。
- **指标**：
  - Rule-update locality：规则分数变化能量中落在图结构相关规则集合内的比例；
  - Update participation ratio：基于逆参与率 \(N_{\mathrm{eff}}=1/\sum_i p_i^2\) 的有效更新坐标数及其归一化比例。
- **文件**：
  - `Introduction.md`：实验动机、公式、共同规则空间、公平性设置和局限；
  - `evaluate_single_fact_locality.py`：在 NATIONS / KINSHIP 上运行 NSR 与 NeuralLP 的完整评估；
  - `Single_Fact_Update_Audit.ipynb`：从原始 JSON 交互式重建主表、逐事件统计和案例表；
  - `Interpre_Results.md`：第一部分为可直接回应 reviewer 的英文学术化量化结果与 rebuttal 文本，第二部分为中文实验设置、案例核查和解读边界；
  - `single_fact_locality_results.json`：每个事实的原始结果与汇总统计。
- **复现**：
  ```bash
  python NSR/Interpretability/evaluate_single_fact_locality.py \
    --datasets NATIONS KINSHIP --num-events 30 --seed 42
  ```

---

## 4. 结果与报告

### `Results/`
- `fixed_params/`：使用 V2 最佳超参数对 `NSR_Simplified.py` 进行评估的结果。
  - `nations/nations_result.json`
  - `kinship/kinship_result.json`
  - `kinship1990/kinship1990_result.json`
  - `*_v2_best.json`：对应的最佳超参数。
- `simplified_search/`：简化版独立 10-trial Optuna 搜索结果。
- `simplified_search_nations30/` / `simplified_search_kinship30/`：30-trial 搜索结果。
- `wn18rr_default/` / `wn18rr_kinship1990_params/`：WN18RR 在默认参数与 KINSHIP_1990 迁移参数下的结果。
- `NSR_Simplified_Report.md`：中文对比总结报告，包含简化思路、指标、耗时、WN18RR 结果与分析。

### `Final_Results.md`
- **作用**：汇总 `tune_wrr_with_cached_c.py` 与 `tune_dynamic.py` 在不同数据集上的最佳验证 MRR、测试指标、训练/推理耗时及完整超参数。
- **内容**：按数据集分节，包含 C 层训练时间、W_RR 训练时间、搜索时间、no-C / with-C MRR、Hits@1/3/10 等。

---

## 5. 复现流程

1. 安装依赖：`numpy`、`scipy`、`pykeen`（KINSHIP、NATIONS、WN18RR 需要）。
2. 评估 `NSR_Simplified.py` 在 KINSHIP_1990 上的效果：
   ```bash
   python NSR/NSR_assistant/evaluate.py \
     --model-file NSR/NSR_Simplified.py \
     --dataset KINSHIP_1990 \
     --params-json NSR/Results/fixed_params/kinship1990_v2_best.json \
     --output-dir NSR/Results/fixed_params/kinship1990 \
     --kinship1990-dir KINSHIP_1990
   ```
3. 查看对比报告：`NSR/Results/NSR_Simplified_Report.md`。

---

## 6. 更新日志

| 日期 | 变更 |
|------|------|
| 2026-07-22 | 创建 `NSR_Dynamic.py`、`NSR_Simplified.py`、`NSR_Test.ipynb`、`NSR_assistant/evaluate.py`，以及 `Results/` 下的对比结果与 `NSR_Simplified_Report.md`。 |
| 2026-07-22 | 修复 `NSR_Dynamic.py` 中 `_query_with_C_faithful` 的 `r_i == r_k` fallback 缺陷（修正 τ=3 状态），使 NDF 推理与 V2 / Simplified 一致；将 `learn_cache_reverse` 默认值改为 `True`。 |
| 2026-07-22 | 优化 `NSR_Simplified.py`：预计算 directed `(entity, relation)` 邻居元组；给 `_query_single(use_WRR=False)` 加速路径；修复 `load()` 后未重建邻接表的问题。优化后 NATIONS C 学习 2.61 s，with-C 推理 1.59 s，均超过 V2。 |
| 2026-07-22 | 进一步优化 `NSR_Simplified.py` C 层学习：在 dense/sparse 两种存储模式下均只遍历实际有出边的关系（`_active_rels_by_entity`），并内联 Hebbian 更新；KINSHIP C 学习降至 9.52 s，指标不变。 |
| 2026-07-22 | 为 `NSR_Simplified.py` 加入混合稀疏/稠密 C 层存储（`C_dense_memory_limit_bytes`），支持关系数极大的数据集（如 FB15K-237）运行 C 层；`save/load`、`get_learned_rules` 已适配稀疏格式。 |
| 2026-07-22 | 新增 `NSRConfig.C_fast_batching`：对 `(r_i, r_j)` 前件对批量聚合 `(h, t)` 端点更新，缓解稠密图逐路径枚举开销；`running_mean` 下严格等价。 |
| 2026-07-22 | `evaluate.py` 新增 FB15K、FB15K-237、YAGO3-10 支持；新增 `compare_v2_simplified.py` 用于 V2 与 Simplified 在大数据集上的并排比较。 |
| 2026-07-22 | 更新 `NSR/Results/NSR_Simplified_Report.md`：加入 `learn_C_layer_Fast_Approx` 实现与测试结果；在 FB15K-237 上 Fast_Approx 与标准 C 学习时间相近（约 20 min），说明瓶颈已从权重更新转移到路径枚举；YAGO3-10 上 Fast_Approx 仍因内存爆炸无法完成。 |
| 2026-07-23 | 重构 `NSR_Simplified.py` 的 C 层学习入口：`learn_C_layer()` 改为 dispatcher，新增 `NSRConfig.C_learning_algorithm`（`"online" | "count" | "matrix"`）；原 `Fast_Approx` / `Fast_Approx_v2` 分别重命名为 `count` / `matrix` 并改为私有方法 `_learn_C_layer_count` / `_learn_C_layer_matrix`；保留 `C_fast_approx` / `C_fast_approx_v2` 作为向后兼容别名。新增稀疏矩阵版 `matrix` 使 YAGO3-10 C 层学习从 OOM 失败降到约 4.7 min；统一 `C_support` / `_C_hit_dense` 为 `int64` 避免大 support 溢出。 |
| 2026-07-23 | 新增 `NSR_Simplified.py` 中 `W_RR` 学习算法切换：`NSRConfig.W_RR_learning_algorithm`（`"online" | "cooccurrence"`）。默认 `"online"` 与 `NSR_Dynamic.py` 完全一致； `"cooccurrence"` 从有向实体对上的关系共现矩阵直接构造 `W_RR`，在 FB15K-237 上比 online 快约 215 倍，但与 online 不等价。保留向后兼容别名 `W_RR_fast=True -> "cooccurrence"`。 |
| 2026-07-23 | 为 `NSR_Simplified.py` 引入两种 fallback 机制（源自 `NSR_Simplified/NSR_Simplified_model.py`）：`tail_fallback_weight`（尾实体 popularity fallback）与 `concepts_fallback_weight` / `concepts_fallback_k`（结构等价 fallback，旧称 `re_fallback` 已更名）。 |
| 2026-07-23 | 新增 `NSR_assistant/tune_wrr_with_cached_c.py`：先学习并缓存 C 层，再单独对 `NSR_Simplified.py` 的 W_RR + fallback 超参数做 Optuna 搜索；新增 `NSR_assistant/tune_dynamic.py`：对 `NSR_Dynamic.py` 做完整 Optuna 搜索。两脚本均将结果写入 `NSR/Final_Results.md`。 |
| 2026-07-23 | 新增 `NSR/Test_On_KGs.ipynb`：多数据集 Optuna 搜索示例，展示如何缓存 C 层并只调 W_RR + fallback。 |
| 2026-07-23 | 调整 `NSR_assistant/tune_dynamic.py`：在搜索空间中加入 `top_rules_fraction`，并在 valid/test 评估中生效，以解决 NATIONS 上规则数过多导致动力学推理超时的问题。同步更新 `Introduction.md` 说明。 |
| 2026-07-26 | 新增 `Interpretability/` 单事实学习更新审计：比较 NSR 与 NeuralLP 的 Rule-update locality 和 Update participation ratio，在 NATIONS / KINSHIP 各评估 30 个事实，并提供脚本、交互式 notebook、原始 JSON 与 reviewer-ready 英文报告。 |
| 2026-07-26 | 重构 `Interpretability/Interpre_Results.md`：英文部分专门回应 reviewer 的量化可解释性问题，中文部分记录实验设置与 small audit case；Markdown 数学公式统一使用 `$` / `$$`。 |
| 2026-07-26 | 精简 `Interpretability/Interpre_Results.md` 的 reviewer-facing Part I，仅保留扩展 rebuttal、指标直观含义和主结果表；将指标公式及完整解释移入中文 Part II。 |
