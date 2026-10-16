# Single-Fact Learning-Update Audit

## Purpose

This directory evaluates a deliberately narrow and testable form of
interpretability: **whether the effect of observing one new fact can be
localized and audited in the model's learned state**.  It does not claim that
every aspect of either model's reasoning process is human-interpretable.

For a held-out fact

$$
e=(h,r,t),
$$

we compare the model state immediately before and immediately after learning
that fact.  The experiment covers NSR and NeuralLP on NATIONS and KINSHIP and
reports:

1. **Rule-update locality**: how much of the change in two-hop rule scores is
   assigned to rules whose graph evidence is actually changed by $e$.
2. **Update participation ratio**: how many internal coordinates effectively
   participate in the atomic learning update.
3. A small audit trail listing the largest rule-score changes for one example
   fact.

The complete numerical results are in
[`Interpre_Results.md`](Interpre_Results.md), and the raw per-event records are
in [`single_fact_locality_results.json`](single_fact_locality_results.json).

## Common rule space

The comparison is restricted to length-two relational composition rules,

$$
\rho:\quad r_k(X,Z)\Leftarrow r_i(X,Y)\wedge r_j(Y,Z),
$$

including inverse relations in the body.  NSR stores one C-layer connection
for every such rule.  NeuralLP does not store a discrete rule table, but its
operator and memory attentions can be decoded into weights for the same rule
space.

For a length-two NeuralLP path, the decoded rule score is

$$
c^{\mathrm{NLP}}_{kij}
=
a^{\mathrm{mem}}_{0,0}
a^{\mathrm{op}}_{0,i}
a^{\mathrm{mem}}_{1,1}
a^{\mathrm{op}}_{1,j}
a^{\mathrm{mem}}_{2,2}.
$$

This is the exact multiplicative enumeration used by NeuralLP's
`list_rules`, evaluated without its display-time pruning threshold.  Thus,
NeuralLP can output weighted rule combinations; these weights are
attention-derived rule scores, not calibrated empirical probabilities or
independently stored rule confidences.

For NSR, the count/running-mean C-layer score is

$$
c^{\mathrm{NSR}}_{kij}
=
\frac{H_{kij}}{S_{ij}},
$$

where $S_{ij}$ is the number of observed two-hop bodies
$(r_i,r_j)$, and $H_{kij}$ is the number for which the consequent
$r_k$ is also observed.  This is the exact sufficient-statistic form of the
current NSR C-layer learner.

## Event selection

For each dataset, the script uses seed 42 to select 30 validation triples that:

- are absent from the training graph; and
- close at least one existing two-hop path from $h$ to $t$.

The second condition is intentional: a fact that touches no two-hop
composition provides no meaningful test of a two-hop rule learner.  Each event
is evaluated independently from the same base graph/checkpoint, so updates do
not accumulate across events.

The graph is augmented with inverse edges for both methods.  NATIONS and
KINSHIP use the same official training facts.  In the NeuralLP data layout,
these facts are split between `facts.txt` and `train.txt`; their union is
identical to the common training split used by NSR.

## A. Rule-update locality

Let $c_\rho$ be the score of rule $\rho$, and let

$$
\Delta c_\rho(e)=c_\rho(G\cup\{e\})-c_\rho(G).
$$

Before comparing the models, we define a model-independent, graph-theoretic
relevance set:

$$
\mathcal R(e)=
\left\{\rho=(r_k,r_i,r_j):
S_{ij}\ \text{or}\ H_{kij}\ \text{changes after adding }e
\right\}.
$$

In words, a rule is relevant when the new edge changes either its two-hop body
support or its consequent-hit count.  Rule-update locality is then

$$
\boxed{
L_{\mathrm{rule}}(e)=
\frac{\sum_{\rho\in\mathcal R(e)}\Delta c_\rho(e)^2}
     {\sum_{\rho}\Delta c_\rho(e)^2}
}
$$

and lies in $[0,1]$.  A value of 1 means that all rule-score update energy is
confined to rules whose observed graph evidence changed.  Squared changes are
used so that positive and negative updates cannot cancel.

## B. Update participation ratio

Let $\Delta\theta_i$ be the change in one evaluated internal coordinate.
Define

$$
p_i=\frac{\Delta\theta_i^2}{\lVert\Delta\theta\rVert_2^2},
\qquad
\boxed{
N_{\mathrm{eff}}=\frac{1}{\sum_i p_i^2}
},
\qquad
\boxed{
R_{\mathrm{eff}}=\frac{N_{\mathrm{eff}}}{P}
},
$$

where $P$ is the number of evaluated coordinates.  $N_{\mathrm{eff}}$ is
the inverse participation ratio: it equals 1 when one coordinate contains all
update energy and approaches $P$ for a uniformly distributed update.
$R_{\mathrm{eff}}$ makes the result comparable across different model
sizes.  Lower values indicate a more concentrated and easier-to-audit update.
The metric is invariant to multiplying the whole update by a scalar, so it
does not compare raw learning-rate magnitudes.

The evaluated coordinates follow each method's native learning
representation:

- **NSR:** all forward-consequent two-hop C-layer connections,
  $P=M(2M)^2$.
- **NeuralLP:** all trainable parameters changed by its native Adam update.

The supplementary nonzero-count statistic in the raw JSON is descriptive
only.  It uses a tiny relative numerical threshold and is not the primary
sparsity measure.

## Atomic update protocol

- **NSR:** recompute the exact count/running-mean C-layer state for
  $G\cup\{e\}$.  This is equivalent to applying the fact's local sufficient
  statistic updates to the base state.
- **NeuralLP:** restore the released trained checkpoint, then perform one
  native Adam update on $e$ and its inverse, using the full official training
  graph as the differentiable database.  The checkpoint, including optimizer
  state, is restored before every event.
- Decode the complete NeuralLP length-two rule tensor before and after the
  step; no top-$k$ or score-threshold pruning is used.

The independently implemented NSR counting routine was cross-checked against
the complete public `NSR_Simplified.py` count learner.  On the two full
training graphs, the maximum absolute rule-score difference was below
$3\times10^{-8}$.

## Reproduction

From the repository root:

```bash
python NSR/Interpretability/evaluate_single_fact_locality.py \
  --datasets NATIONS KINSHIP \
  --num-events 30 \
  --seed 42
```

The public driver runs in the current Python environment.  NeuralLP is
launched automatically in the existing `nsr_tf1` Conda environment because
the released implementation uses TensorFlow 1.  Its released checkpoints are
expected under:

```text
Other_Model/results/{DATASET}/neurallp/experiment/ckpt/
```

For interactive inspection, open
[`Single_Fact_Update_Audit.ipynb`](Single_Fact_Update_Audit.ipynb).  The
notebook loads the committed raw JSON, reconstructs the main table, shows
per-event distributions, and exposes the example rule changes.  A final
optional cell reruns the full experiment.

## Scope and limitations

- These metrics quantify **learning-update locality and auditability**, not
  global human interpretability, causal faithfulness of every prediction, or
  superiority on all notions of interpretability.
- The relevance set is graph-theoretic and fixed before looking at either
  model's updates.  It naturally corresponds to the evidence used by an
  explicit two-hop sufficient-statistic learner.
- The two models have different native atomic update semantics: exact
  sufficient-statistic incorporation for NSR versus one optimizer step for
  NeuralLP.  The experiment therefore asks how localized each model's native
  response to the same fact is; it does not equate their learning rates.
- Reported standard deviations are across 30 facts from one fixed
  checkpoint/seed, not across independent training seeds.
- Events are deliberately conditioned on closing an existing two-hop path and
  should not be interpreted as an unconditioned sample of all possible facts.

## Files

- `evaluate_single_fact_locality.py`: executable NSR/NeuralLP evaluation.
- `single_fact_locality_results.json`: complete aggregate and per-event output.
- `_worker_results/`: intermediate NeuralLP worker records.
- `Single_Fact_Update_Audit.ipynb`: interactive analysis and reproduction.
- `Interpre_Results.md`: reviewer-ready results and interpretation.
