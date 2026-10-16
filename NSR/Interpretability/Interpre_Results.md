# Quantitative Evaluation of Learning-Update Interpretability

## Part I. Reviewer-Facing Response

We agree that merely displaying relational rules does not by itself
demonstrate improved interpretability, since NeuralLP can also decode weighted
rule combinations from its learned attention distributions. We therefore
evaluate a more specific and testable property of the learning process:
**whether the effect of one newly observed fact can be localized to, and
audited through, the relational structures affected by that fact**.

We use a single-fact learning-update audit with two complementary metrics.
**Rule-update locality** measures the fraction of rule-score change assigned
to rules whose graph evidence is actually altered by the new fact; a higher
value indicates better semantic localization. **Update participation ratio**
measures the effective fraction of a model's native internal coordinates
involved in the update; a lower value indicates a more concentrated and
directly auditable change.

We independently evaluate 30 held-out facts on each of NATIONS and KINSHIP.
Every selected fact is absent from training and closes at least one existing
two-hop path. Each fact is evaluated from the same base graph or restored
checkpoint, preventing updates from accumulating across examples.

| Dataset | Method | Rule-update locality $L_{\mathrm{rule}}$ ↑ | Effective coordinates $N_{\mathrm{eff}}$ ↓ | Evaluated coordinates $P$ | Normalized participation $N_{\mathrm{eff}}/P$ ↓ |
|---|---:|---:|---:|---:|---:|
| NATIONS | **NSR** | **1.0000 ± 0.0000** | **220.8 ± 210.6** | 665,500 | **0.0332% ± 0.0317%** |
| NATIONS | NeuralLP | 0.0796 ± 0.1277 | 76,317.4 ± 7,996.8 | 159,982 | 47.7038% ± 4.9985% |
| KINSHIP | **NSR** | **1.0000 ± 0.0000** | **20.1 ± 22.5** | 62,500 | **0.0321% ± 0.0361%** |
| KINSHIP | NeuralLP | 0.3092 ± 0.2633 | 48,077.0 ± 9,178.9 | 144,562 | 33.2570% ± 6.3495% |

Values are reported as mean ± sample standard deviation across the 30 learning
events. NSR obtains rule-update locality $1.000$ on both datasets, compared
with $0.080$ and $0.309$ for NeuralLP. Thus, all measured NSR rule-score
update energy is confined to rules whose graph evidence is changed by the new
fact, whereas NeuralLP exhibits substantial changes outside this set.
Consistently, NSR's normalized effective participation is only $0.033\%$ and
$0.032\%$, compared with $47.704\%$ and $33.257\%$ for NeuralLP.

These results show that the distinction is not simply whether a method can
output a weighted rule. In NeuralLP, an individual fact updates a broadly
shared parameterization; in NSR, the same learning event produces a
concentrated change in identifiable C-layer connections associated with the
affected relational compositions. This provides quantitative evidence that
NSR offers more semantically localized and directly auditable learning
updates than NeuralLP.

We restrict this conclusion to **learning-update interpretability** rather
than claiming improvement under every definition of interpretability. The two
methods use their native atomic learning operations—an exact
count/running-mean update for NSR and one Adam step from a released checkpoint
for NeuralLP—so the participation ratio compares update concentration rather
than coordinate-wise parameter equivalence.

---

## 第二部分：实验设置与案例核查

### 1. 比较对象与共同规则空间

实验只比较两跳组合规则：

$$
\rho:\quad
r_k(X,Z)\Leftarrow r_i(X,Y)\wedge r_j(Y,Z).
$$

规则体中的关系包含正向关系和逆关系。选择两跳规则的原因是：NSR 的 C
层原生表示这类组合规则，而 NeuralLP 的 operator attention 和 memory
attention 也能够被解码到同一个规则空间。这样，Rule-update locality
比较的是相同语义空间中的规则分数变化，而不是直接比较两个形状不同的内部参数张量。

NSR 使用 count/running-mean 形式的 C 层规则分数：

$$
c^{\mathrm{NSR}}_{kij}=
\frac{H_{kij}}{S_{ij}}.
$$

其中，$S_{ij}$ 是两跳规则体 $(r_i,r_j)$ 的路径数量，$H_{kij}$
是在这些路径端点之间同时观察到 $r_k$ 的次数。

NeuralLP 并不显式保存一个离散规则表，但可以从 attention 中解码出带权规则组合。
对于长度为二的路径，其分数为：

$$
c^{\mathrm{NLP}}_{kij}
=
a^{\mathrm{mem}}_{0,0}
a^{\mathrm{op}}_{0,i}
a^{\mathrm{mem}}_{1,1}
a^{\mathrm{op}}_{1,j}
a^{\mathrm{mem}}_{2,2}.
$$

这里使用的是 NeuralLP 原始 `list_rules` 枚举所对应的精确乘积，并且没有使用
top-$k$ 或阈值剪枝。需要注意：这个值是 attention-derived rule score，
可以理解为带权规则组合的分数，但不应称为经过校准的概率，也不等价于 NSR
基于观测计数得到的经验置信度。

### 2. 评估指标及公式解释

#### A. Rule-update locality

对一个新事实 $e=(h,r,t)$，先根据图结构定义与该事实相关的规则集合：

$$
\mathcal R(e)=
\left\{
(r_k,r_i,r_j):
S_{ij}\ \text{或}\ H_{kij}
\text{在加入 }e\text{ 后发生变化}
\right\}.
$$

该定义不依赖 NSR 或 NeuralLP 的模型参数。如果某条规则的两跳规则体 support
$S_{ij}$ 或 consequent hit count $H_{kij}$ 因为新事实而改变，就将其视为
graph-relevant rule。

设规则 $\rho$ 在观察新事实前后的分数变化为

$$
\Delta c_\rho(e)=
c_\rho(G\cup\{e\})-c_\rho(G).
$$

Rule-update locality 定义为

$$
L_{\mathrm{rule}}(e)=
\frac{
\sum_{\rho\in\mathcal R(e)}\Delta c_\rho(e)^2
}{
\sum_{\rho}\Delta c_\rho(e)^2
}.
$$

$L_{\mathrm{rule}}\in[0,1]$。它表示全部规则分数变化能量中，有多少落在
graph-relevant rules 上。值为 $1$ 表示所有规则变化都能由该事实实际改变的图证据解释。
使用平方变化可以防止正负更新相互抵消。该指标在 NSR 和 NeuralLP 共享的两跳规则空间
中计算，因此衡量的是语义层面的规则更新局部性。

#### B. Update participation ratio

Rule-update locality 说明哪些规则发生变化，但不能说明内部更新是否集中。令
$\Delta\theta_i$ 表示第 $i$ 个被评估内部坐标的变化，并定义

$$
p_i=
\frac{\Delta\theta_i^2}
{\lVert\Delta\theta\rVert_2^2},
\qquad
N_{\mathrm{eff}}=
\frac{1}{\sum_i p_i^2},
\qquad
R_{\mathrm{eff}}=
\frac{N_{\mathrm{eff}}}{P},
$$

其中 $P$ 是被评估坐标总数。$N_{\mathrm{eff}}$ 是 inverse participation
ratio：如果全部更新能量集中在一个坐标上，则 $N_{\mathrm{eff}}=1$；如果更新在
$P$ 个坐标间均匀分布，则 $N_{\mathrm{eff}}$ 接近 $P$。

归一化比例 $R_{\mathrm{eff}}=N_{\mathrm{eff}}/P$ 用于比较不同参数规模的模型。
值越低，说明一次学习事件实际牵涉的内部坐标越少，更新越集中，也越容易审计。
该指标对整个更新向量的统一缩放保持不变，因此不直接比较两个方法的 learning-rate
magnitude。NSR 在 C-layer rule connections 上计算该指标，NeuralLP
则在一次原生 Adam step 改变的所有 trainable parameters 上计算。

### 3. 数据和单事实选择

- 数据集：NATIONS 和 KINSHIP。
- 每个数据集选择 30 个 validation facts，随机种子固定为 42。
- 每个事实必须不在训练图中。
- 每个事实必须使头实体和尾实体之间至少一条已有的两跳路径闭合；否则该事实无法为两跳规则提供有效的更新局部性测试。
- 每个事实都从同一个基础训练图或 checkpoint 独立评估，事实之间不累积更新。
- 两个方法都使用包含逆边的关系图。
- NeuralLP 数据目录将训练知识拆成 `facts.txt` 和 `train.txt`；二者的并集已经核对，与 NSR 使用的公共 training split 完全一致。

这种采样是有条件的 compositional-event 采样，不代表对所有可能三元组的无条件随机采样。

### 4. 两种方法的原子更新

#### NSR

对每个事实 $e$，分别计算 $G$ 和 $G\cup\{e\}$ 上完整的
count/running-mean C 层状态。两者之差就是该事实引起的规则连接变化。
这与对 C 层局部充分统计量进行精确增量更新等价。

NSR 的 participation ratio 在以下坐标上计算：

$$
P=M(2M)^2,
$$

即所有正向 consequent 对应的两跳 C 层连接。因此，NATIONS 的
$P=665{,}500$，KINSHIP 的 $P=62{,}500$。

#### NeuralLP

对每个事实都重新恢复同一个发布 checkpoint，包括其中的 Adam optimizer
状态，然后对事实 $e$ 及其逆事实执行一次原生 Adam update。更新过程中使用完整
training graph 作为 differentiable database。更新前后解码完整的两跳规则分数张量，
同时记录所有 trainable parameters 的变化。

- NATIONS 使用 `model-8`，参数量为 159,982。
- KINSHIP 使用 `model-10`，参数量为 144,562。

这种设置比较的是两种方法面对同一个新事实时各自的原生 atomic update。
它没有把 NSR 的精确计数更新和 NeuralLP 的梯度学习率强行设为相同数值。
Participation ratio 对整体缩放不敏感，因此不会因为原始更新幅度不同而直接受影响。

### 5. Small audit case

案例是固定随机种子下第一个满足条件的 NATIONS validation fact，并不是观察结果后挑选的：

$$
e=
(\texttt{E\_egypt},
\texttt{R\_intergovorgs},
\texttt{E\_israel}).
$$

该事实闭合了 4,032 条已有的有向两跳路径，并改变了 15,930 条规则的图证据。
下面列出两个方法绝对变化最大的三条规则：

| Method | 观察到 $e$ 后变化最大的规则 | Before → after（$\Delta$） | 是否属于 $\mathcal R(e)$ |
|---|---|---:|:---:|
| NSR | `R_intergovorgs <= inv_R_tourism3 o inv_R_lostterritory` | 0.0000 → 1.0000 (+1.0000) | 是 |
| NSR | `R_intergovorgs <= inv_R_tourism o inv_R_lostterritory` | 0.0000 → 1.0000 (+1.0000) | 是 |
| NSR | `R_intergovorgs <= R_severdiplomatic o inv_R_tourism3` | 0.5000 → 1.0000 (+0.5000) | 是 |
| NeuralLP | `R_ngo <= inv_R_ngo o inv_R_ngo` | 0.01851 → 0.02366 (+0.00514) | **否** |
| NeuralLP | `R_intergovorgs <= inv_R_reltourism o inv_R_reltourism` | 0.00836 → 0.00502 (−0.00334) | **否** |
| NeuralLP | `R_treaties <= inv_R_reltourism o inv_R_reltourism` | 0.03709 → 0.03465 (−0.00244) | **否** |

这个案例展示了 aggregate metrics 的具体含义：NSR 最大的规则分数变化都能追溯到
新事实实际影响的关系组合；NeuralLP 的最大变化可能出现在其他 query relation
对应的规则上，因为一次梯度更新会改变多个关系共享的参数。这并不说明 NeuralLP
解码出的规则无效，而是说明从“单个训练事实”到“具体规则分数变化”的审计链条更分散。

### 6. 实现核查

- NSR 的独立计数实现已经与公开的 `NSR_Simplified.py` count learner
  在两个完整训练图上交叉核对。
- NATIONS 上最大绝对误差为 $2.97\times10^{-8}$。
- KINSHIP 上最大绝对误差为 $2.97\times10^{-8}$。
- 该误差处于 float32 舍入误差范围。
- NeuralLP 的两跳规则张量使用完整 attention 分布解码，没有使用展示阶段的
  rule threshold 或 top-$k$ 筛选。

完整 aggregate 和 per-event 数据保存在
[`single_fact_locality_results.json`](single_fact_locality_results.json)，
交互式复核见
[`Single_Fact_Update_Audit.ipynb`](Single_Fact_Update_Audit.ipynb)。

### 7. 解读边界

1. 该实验量化的是学习过程中的 update locality 和 auditability，而不是所有形式的
   human interpretability。
2. 不能据此声称 NSR 在所有 inference-time explanation 上都优于 NeuralLP。
3. $\mathcal R(e)$ 在查看两个模型的变化之前由图结构独立定义，但这个定义天然对应
   NSR 使用的局部充分统计量，因此 rebuttal 中应主动说明。
4. 两种方法使用各自原生的原子更新语义：NSR 是精确统计更新，NeuralLP 是一步
   checkpointed Adam update。
5. 表格中的标准差来自 30 个不同事实，而不是多次独立训练的随机种子。
6. NSR 中数值上非零的连接可能很多，因为 support denominator 的变化会使多个很小的
   rule scores 同时变化。因此主指标应使用能量加权的 $N_{\mathrm{eff}}$，
   而不应使用简单的 nonzero parameter count。
