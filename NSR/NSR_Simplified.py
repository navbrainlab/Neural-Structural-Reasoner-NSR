"""Simplified Dynamic NSR with adjacency-table ERZ semantics.

The model contains only the mechanisms needed for:
1. encoding triples into E-R-Z synaptic connectivity;
2. learning relation-relation weights W_RR with a local Oja/Hebbian rule;
3. running the five-step ERZ dynamics;
4. learning 2M x 2M compositional antecedents from ERZ activations;
5. answering direct and compositional tail queries.

Simplification note
-------------------
This version keeps the same compositional-rule learning and scoring as
NSR_Fake_C_Dynamic_V2 / NDF, but replaces the ERZ dynamics with equivalent
adjacency-table lookups.  The effective semantics of a closed-W_RR ERZ episode
are exactly the direct neighbors of the queried relation; the effective
semantics of the reverse consequent readout are exactly the forward relations
that directly connect (h, t).  Thus W_comp_agent, C_support and query scores
remain identical to the dynamic version while train/query time is much lower.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp

Triple = Tuple[int, int, int]


def gated_relu(x: np.ndarray, threshold: float) -> np.ndarray:
    """Clipped thresholded ReLU: clip(max(0, x-threshold), 0, 1)."""
    return np.clip(np.maximum(0.0, x - threshold), 0.0, 1.0)


def sigma_gate(x: np.ndarray, threshold: float) -> np.ndarray:
    """Keep continuous activity at or above threshold."""
    return x * (x >= threshold).astype(float)


def signed_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    """Map non-negative Hebbian weights to [0, 1)."""
    return 2.0 / (1.0 + np.exp(-k * x)) - 1.0


@dataclass
class NSRConfig:
    # ERZ dynamics
    eta_Z: float = 1.0
    eta_E: float = 0.0
    T_thresh: float = 0.0
    T_infer: int = 5
    b_bias: float = 1.0
    w_zz_strength: float = 1.0
    i_e_delay: int = 2
    hold_R_until_e_delay: bool = True
    inhibit_R_from: int = 3
    inhibit_value: float = 1e6
    use_rr_recurrence: bool = True

    # W_RR learning
    # Top-level algorithm switch (mutually exclusive):
    #   "online"        : local Hebbian/Oja update per directed entity pair.
    #                     Sub-options: delta_RR, mu_RR, n_passes (via argument).
    #   "cooccurrence"  : build relation co-occurrence matrix over directed
    #                     entity pairs and convert it to W_RR directly.
    #                     Much faster on large graphs; not mathematically
    #                     identical to "online" but produces a similar
    #                     relation-similarity matrix.
    # Deprecated alias: W_RR_fast=True is equivalent to "cooccurrence".
    W_RR_learning_algorithm: str = "online"  # "online" | "cooccurrence"
    W_RR_fast: bool = False

    delta_RR: float = 0.5
    mu_RR: float = 0.05
    W_RR_normalize: str = "sigmoid"  # sigmoid | max | none
    W_RR_sigmoid_k: float = 1.0

    # ERZ-driven C learning
    C_learning_T: int = 5
    C_activity_threshold: float = 0.5
    C_hebbian_mode: str = "ema"  # running_mean | ema
    delta_C: float = 0.1                    # EMA step when mode=ema
    C_n_passes: int = 1
    C_min_support: int = 2

    # C inference
    enable_C_layer_by_default: bool = True
    C_rule_threshold: float = 0.1
    alpha: float = 1.0

    # Fallback mechanisms (adapted from NSR_Simplified/NSR_Simplified_model.py)
    # tail_fallback_weight: boost tail entities by their popularity as tails of r_q
    #   when rule-based scores are empty.
    # concepts_fallback_weight / concepts_fallback_k: structural-equivalence fallback.
    #   When rule-based scores are empty, find the top-k most structurally similar
    #   training heads for relation r_q, run inference for each, and aggregate their
    #   tail scores weighted by similarity.  Formerly called re_fallback.
    tail_fallback_weight: float = 0.0
    concepts_fallback_weight: float = 0.0
    concepts_fallback_k: int = 3

    # Performance only: cached values are outputs of ERZ runs, not graph tables.
    enable_erz_cache: bool = True

    # If True, cache _reverse_consequent_activity(h, t) during C learning to avoid
    # recomputing the same (h, t) return hop for different intermediate entities x.
    learn_cache_reverse: bool = True

    # Memory threshold for keeping the full K^3 W_comp_agent dense.
    # Below this limit the dense array is faster; above it we use per-pair
    # dense vectors to avoid O(K^3) memory (e.g. FB15K would need ~78 GB).
    C_dense_memory_limit_bytes: int = 512 * 1024 * 1024

    # ------------------------------------------------------------------
    # C learning algorithm selection (mutually exclusive top-level switch)
    #
    #   "online"  : standard per-path Hebbian/running_mean/EMA updates.
    #               Sub-options: C_fast_batching, C_hebbian_mode, delta_C,
    #               C_n_passes, learn_cache_reverse.
    #   "count"   : one-pass path counting via explicit Python enumeration.
    #               Mathematically equivalent to running_mean; faster on
    #               small/medium graphs; no online-update hyperparameters.
    #   "matrix"  : one-pass path counting via scipy.sparse matrix multiplication.
    #               Same result as "count" but much faster and more memory
    #               efficient on large/dense graphs such as FB15K-237/YAGO3-10.
    #
    # The deprecated booleans C_fast_approx / C_fast_approx_v2 still work for
    # backward compatibility and override C_learning_algorithm as follows:
    #   C_fast_approx_v2=True  -> "matrix"
    #   C_fast_approx=True     -> "count"
    # ------------------------------------------------------------------
    C_learning_algorithm: str = "online"  # "online" | "count" | "matrix"

    # Second-level switches, only effective when C_learning_algorithm == "online".
    # If True, group (h,t) endpoints per (r_i,r_j) and apply one batched update
    # per (h,t) instead of one per path. Exact for running_mean, approximate for ema.
    C_fast_batching: bool = False

    # Deprecated backward-compatibility aliases for C_learning_algorithm.
    # Prefer setting C_learning_algorithm directly.
    C_fast_approx: bool = False     # equivalent to "count"
    C_fast_approx_v2: bool = False  # equivalent to "matrix"

    def to_dict(self) -> Dict[str, int | float | bool | str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, int | float | bool | str]) -> "NSRConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NSRDynamicERZC:
    """Dynamic NSR whose C statistics are generated entirely by ERZ activity."""

    def __init__(self, num_entities: int, num_relations: int, cfg: Optional[NSRConfig] = None):
        self.N = int(num_entities)
        self.M = int(num_relations)
        self.K = 2 * self.M
        self.NZ = self.N * self.K
        self.cfg = cfg if cfg is not None else NSRConfig()

        if self.N <= 0 or self.M <= 0:
            raise ValueError("num_entities and num_relations must be positive")

        # Fixed E -> Z projection: each entity projects to all of its relation-indexed Z cells.
        z_rows = np.arange(self.NZ, dtype=np.int64)
        e_cols = np.repeat(np.arange(self.N, dtype=np.int64), self.K)
        self.W_EZ = sp.csr_matrix(
            (np.ones(self.NZ, dtype=float), (z_rows, e_cols)),
            shape=(self.NZ, self.N),
        )

        # Learned/encoded sparse ERZ synapses are accumulated as unique entries.
        self._zr_entries: Dict[Tuple[int, int], float] = {}
        self._zz_entries: Dict[Tuple[int, int], float] = {}
        self._sparse_dirty = True
        self._W_ZR: Optional[sp.csr_matrix] = None
        self._W_ZZ: Optional[sp.csr_matrix] = None

        self.W_RR = np.zeros((self.K, self.K), dtype=float)
        self._encoded_triples: List[Triple] = []
        self._erz_cache: Dict[Tuple[int, int, int, bool, float], Dict[int, float]] = {}

        # Adjacency tables that replace closed-W_RR ERZ dynamics.
        self._tails_by_hr: Dict[Tuple[int, int], set] = {}
        self._heads_by_tr: Dict[Tuple[int, int], set] = {}
        self._rels_ht: Dict[Tuple[int, int], set] = {}
        # Precomputed sorted tuple of active targets for every directed (entity, relation).
        self._active_targets: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        # Relations that have at least one target from each entity (used to skip
        # empty r_i / r_j iterations during C learning).
        self._active_rels_by_entity: Dict[int, Tuple[int, ...]] = {}

        # C-layer storage.
        # For small K we keep the full K^3 dense array (fast numpy updates).
        # For large K (e.g. FB15K) we fall back to per-antecedent-pair dense
        # vectors, which only consumes memory proportional to active pairs.
        self._use_dense_c = (self.K ** 3 * 4) <= int(self.cfg.C_dense_memory_limit_bytes)
        if self._use_dense_c:
            self._W_dense = np.zeros((self.K, self.K, self.K), dtype=np.float32)
            self._C_hit_dense = np.zeros((self.K, self.K, self.K), dtype=np.int64)
            self._W_comp: Dict[Tuple[int, int], np.ndarray] = {}
        else:
            self._W_dense = None
            self._C_hit_dense = None
            self._W_comp: Dict[Tuple[int, int], np.ndarray] = {}
        self.C_support = np.zeros((self.K, self.K), dtype=np.int64)
        self._C_hits: Dict[Tuple[int, int, int], int] = {}
        self._rules_by_consequent: Dict[int, List[Tuple[int, int, float]]] = {}
        self.C_learning_stats: Dict[str, int | float | str] = {}

        # Fallback data structures (populated by encode_triples / _rebuild_adjacency_tables).
        self._tail_pop: Dict[int, Dict[int, int]] = {}
        self._tail_pop_max: Dict[int, float] = {}
        self._heads_by_rel: List[set] = [set() for _ in range(self.M)]
        self._entity_deg: Optional[np.ndarray] = None
        self._rel_adj: List[sp.csr_matrix] = []
        self._concepts_sim_cache: Dict[int, np.ndarray] = {}

    @property
    def W_comp_agent(self) -> np.ndarray:
        """Backward-compatible dense view of the rule tensor."""
        if self._use_dense_c:
            return self._W_dense
        if self.K ** 3 * 4 > 1024 * 1024 * 1024:
            raise MemoryError(
                f"Dense W_comp_agent view would need {self.K**3*4/1e9:.1f} GB. "
                "Use get_learned_rules() or self._W_comp instead."
            )
        arr = np.zeros((self.K, self.K, self.K), dtype=np.float32)
        for (r_i, r_j), vec in self._W_comp.items():
            for r_k in np.nonzero(vec > 0.0)[0]:
                arr[int(r_k), int(r_i), int(r_j)] = float(vec[int(r_k)])
        return arr

    @property
    def C_hit_count(self) -> np.ndarray:
        """Backward-compatible dense view of the hit counts."""
        if self._use_dense_c:
            return self._C_hit_dense
        if self.K ** 3 * 4 > 1024 * 1024 * 1024:
            raise MemoryError(
                f"Dense C_hit_count view would need {self.K**3*4/1e9:.1f} GB. "
                "Use self._C_hits instead."
            )
        arr = np.zeros((self.K, self.K, self.K), dtype=np.int64)
        for (r_k, r_i, r_j), c in self._C_hits.items():
            arr[int(r_k), int(r_i), int(r_j)] = int(c)
        return arr

    # ------------------------------------------------------------------
    # Sparse ERZ connectivity
    # ------------------------------------------------------------------
    def z_idx(self, entity: int, relation: int) -> int:
        return int(entity) * self.K + int(relation)

    @staticmethod
    def _csr_from_entries(
        entries: Mapping[Tuple[int, int], float],
        shape: Tuple[int, int],
    ) -> sp.csr_matrix:
        if not entries:
            return sp.csr_matrix(shape, dtype=float)
        rows = np.fromiter((key[0] for key in entries), dtype=np.int64)
        cols = np.fromiter((key[1] for key in entries), dtype=np.int64)
        vals = np.fromiter(entries.values(), dtype=float)
        return sp.csr_matrix((vals, (rows, cols)), shape=shape)

    def _ensure_sparse(self) -> None:
        if not self._sparse_dirty and self._W_ZR is not None and self._W_ZZ is not None:
            return
        self._W_ZR = self._csr_from_entries(self._zr_entries, (self.K, self.NZ))
        self._W_ZZ = self._csr_from_entries(self._zz_entries, (self.NZ, self.NZ))
        self._sparse_dirty = False

    @property
    def W_ZR(self) -> sp.csr_matrix:
        self._ensure_sparse()
        assert self._W_ZR is not None
        return self._W_ZR

    @property
    def W_ZZ(self) -> sp.csr_matrix:
        self._ensure_sparse()
        assert self._W_ZZ is not None
        return self._W_ZZ

    # ------------------------------------------------------------------
    # Triple encoding
    # ------------------------------------------------------------------
    def encode_triples(self, triples: Iterable[Triple]) -> "NSRDynamicERZC":
        for raw_h, raw_r, raw_t in triples:
            h, r, t = int(raw_h), int(raw_r), int(raw_t)
            if not (0 <= h < self.N and 0 <= t < self.N and 0 <= r < self.M):
                raise ValueError(f"Triple out of range: {(h, r, t)}")

            r_inv = r + self.M
            z_head = self.z_idx(h, r)
            z_tail_inv = self.z_idx(t, r_inv)

            # Keep the sparse ERZ weights for compatibility with save/load.
            self._zr_entries[(r, z_head)] = 1.0
            self._zr_entries[(r_inv, z_tail_inv)] = 1.0
            strength = float(self.cfg.w_zz_strength)
            self._zz_entries[(z_head, z_tail_inv)] = strength
            self._zz_entries[(z_tail_inv, z_head)] = strength
            self._encoded_triples.append((h, r, t))

            # Build adjacency tables that give the same effective semantics as
            # closed-W_RR ERZ dynamics.
            self._tails_by_hr.setdefault((h, r), set()).add(t)
            self._heads_by_tr.setdefault((t, r), set()).add(h)
            self._rels_ht.setdefault((h, t), set()).add(r)

        self._build_active_targets()
        self._build_fallback_structures()
        self._sparse_dirty = True
        self._erz_cache.clear()
        return self

    def _build_active_targets(self) -> None:
        """Cache sorted tuples of direct neighbors for every directed (entity, rel)."""
        self._active_targets = {}
        active_rels_by_entity: Dict[int, List[int]] = {}
        for (h, r), s in self._tails_by_hr.items():
            self._active_targets[(h, r)] = tuple(sorted(int(t) for t in s))
            active_rels_by_entity.setdefault(h, []).append(r)
        for (t, r), s in self._heads_by_tr.items():
            inv_r = r + self.M
            self._active_targets[(t, inv_r)] = tuple(sorted(int(h) for h in s))
            active_rels_by_entity.setdefault(t, []).append(inv_r)
        self._active_rels_by_entity = {
            e: tuple(sorted(set(rs))) for e, rs in active_rels_by_entity.items()
        }

    def _rebuild_adjacency_tables(self) -> None:
        """Rebuild adjacency tables from encoded triples (used after load)."""
        self._tails_by_hr.clear()
        self._heads_by_tr.clear()
        self._rels_ht.clear()
        for h, r, t in self._encoded_triples:
            self._tails_by_hr.setdefault((h, r), set()).add(t)
            self._heads_by_tr.setdefault((t, r), set()).add(h)
            self._rels_ht.setdefault((h, t), set()).add(r)
        self._build_active_targets()
        self._build_fallback_structures()

    def _build_fallback_structures(self) -> None:
        """Pre-compute data for tail-popularity and Concepts (SE) fallbacks.

        Tail-population fallback needs, for each forward relation r, the count of
        times each entity appears as tail of r.  Concepts fallback needs relation-
        level adjacency matrices and entity degrees so that structural similarity
        between two entities can be computed as in NSR_Simplified_model.py.
        """
        self._tail_pop = {r: defaultdict(int) for r in range(self.M)}
        self._tail_pop_max = {}
        self._heads_by_rel = [set() for _ in range(self.M)]

        rel_rows: List[List[int]] = [[] for _ in range(self.M)]
        rel_cols: List[List[int]] = [[] for _ in range(self.M)]
        out_deg = np.zeros(self.N, dtype=np.int64)
        in_deg = np.zeros(self.N, dtype=np.int64)

        for h, r, t in self._encoded_triples:
            self._tail_pop[r][t] += 1
            self._heads_by_rel[r].add(h)
            out_deg[h] += 1
            in_deg[t] += 1
            rel_rows[r].append(int(h))
            rel_cols[r].append(int(t))

        for r in range(self.M):
            counts = self._tail_pop[r]
            self._tail_pop_max[r] = float(max(counts.values())) if counts else 1.0

        self._entity_deg = (out_deg + in_deg).astype(np.float32)
        self._rel_adj = [
            sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32),
                 (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                shape=(self.N, self.N),
            )
            for rows, cols in zip(rel_rows, rel_cols)
        ]
        self._concepts_sim_cache.clear()

    # ------------------------------------------------------------------
    # W_RR learning: online Hebbian/Oja or co-occurrence shortcut
    # ------------------------------------------------------------------
    def learn_WRR_hebbian(
        self,
        triples: Optional[Sequence[Triple]] = None,
        *,
        n_passes: int = 1,
        normalize: Optional[str] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """Learn W_RR from co-active directed relation neurons.

        Dispatches to either the standard online Hebbian/Oja update or the
        co-occurrence-based shortcut according to
        NSRConfig.W_RR_learning_algorithm.

        Args:
            reset: If True (default), zero W_RR before learning. Set to False only
                when intentionally accumulating multiple passes.
        """
        algo = self.cfg.W_RR_learning_algorithm
        if self.cfg.W_RR_fast:
            algo = "cooccurrence"

        if algo == "cooccurrence":
            return self._learn_WRR_cooccurrence(
                triples=triples,
                normalize=normalize,
                seed=seed,
                reset=reset,
                verbose=verbose,
            )
        return self._learn_WRR_online(
            triples=triples,
            n_passes=n_passes,
            normalize=normalize,
            seed=seed,
            reset=reset,
            verbose=verbose,
        )

    def _learn_WRR_online(
        self,
        triples: Optional[Sequence[Triple]] = None,
        *,
        n_passes: int = 1,
        normalize: Optional[str] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """Standard online Hebbian/Oja update for W_RR.

        Each directed entity pair is one experience. Its K-dimensional relation
        activity vector has a 1 for every relation neuron active on that pair.
        The update for each experience is local:

            dW_ij = delta_RR * x_i*x_j
                    - mu_RR * (x_i^2 + x_j^2) * W_ij.
        """
        if reset:
            self.W_RR.fill(0.0)
        source = self._encoded_triples if triples is None else triples
        activity_by_pair: Dict[Tuple[int, int], set[int]] = defaultdict(set)
        for raw_h, raw_r, raw_t in source:
            h, r, t = int(raw_h), int(raw_r), int(raw_t)
            activity_by_pair[(h, t)].add(r)
            activity_by_pair[(t, h)].add(r + self.M)

        samples = sorted(activity_by_pair.items())
        rng = np.random.default_rng(seed)
        delta = float(self.cfg.delta_RR)
        mu = float(self.cfg.mu_RR)

        for _ in range(max(1, int(n_passes))):
            for sample_idx in rng.permutation(len(samples)):
                _, active_relations = samples[int(sample_idx)]
                x = np.zeros(self.K, dtype=float)
                if active_relations:
                    x[np.fromiter(active_relations, dtype=np.int64)] = 1.0
                hebb = np.outer(x, x)
                decay = (x[:, None] ** 2 + x[None, :] ** 2) * self.W_RR
                self.W_RR += delta * hebb - mu * decay

        self.W_RR = np.clip(self.W_RR, 0.0, None)
        np.fill_diagonal(self.W_RR, 0.0)

        method = self.cfg.W_RR_normalize if normalize is None else normalize
        if method == "sigmoid":
            self.W_RR = signed_sigmoid(self.W_RR, float(self.cfg.W_RR_sigmoid_k))
        elif method == "max":
            max_value = float(self.W_RR.max())
            if max_value > 1e-12:
                self.W_RR /= max_value
        elif method == "none":
            pass
        else:
            raise ValueError("normalize must be 'sigmoid', 'max', or 'none'")

        self._erz_cache.clear()
        if verbose:
            print(
                f"[W_RR/online] samples={len(samples)} passes={max(1, int(n_passes))} "
                f"nnz={int(np.count_nonzero(self.W_RR > 1e-9))}/{self.K*self.K} "
                f"max={float(self.W_RR.max()):.4f}"
            )
        return self

    def _learn_WRR_cooccurrence(
        self,
        triples: Optional[Sequence[Triple]] = None,
        *,
        normalize: Optional[str] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """Fast co-occurrence-based W_RR construction.

        Instead of applying a local Hebbian/Oja update for every directed entity
        pair, this method counts how often each pair of directed relations
        co-occurs on the same directed entity pair and converts the raw counts
        into a lift score:

            lift[r_i, r_j] = C_rr[r_i, r_j] * N / (C_r[r_i] * C_r[r_j])

        where C_rr is the co-occurrence count and C_r is the per-relation pair
        count. The lift matrix is then min-max normalized to [0, 1] and passed
        through the same sigmoid/max/none normalization as the online version.

        This is NOT mathematically identical to the online Hebbian/Oja update
        (order-dependence and decay dynamics are lost), but it is much faster on
        large graphs and produces a relation-similarity matrix of comparable
        quality.
        """
        if reset:
            self.W_RR.fill(0.0)
        _ = seed  # deterministic; kept for API compatibility
        source = self._encoded_triples if triples is None else triples
        activity_by_pair: Dict[Tuple[int, int], set[int]] = defaultdict(set)
        for raw_h, raw_r, raw_t in source:
            h, r, t = int(raw_h), int(raw_r), int(raw_t)
            activity_by_pair[(h, t)].add(r)
            activity_by_pair[(t, h)].add(r + self.M)

        N_total = len(activity_by_pair)
        C_r = np.zeros(self.K, dtype=np.float64)
        C_rr = np.zeros((self.K, self.K), dtype=np.float64)

        for rels in activity_by_pair.values():
            rels_list = list(rels)
            for r in rels_list:
                C_r[r] += 1.0
            n = len(rels_list)
            for i in range(n):
                ri = rels_list[i]
                for j in range(n):
                    if i == j:
                        continue
                    rj = rels_list[j]
                    C_rr[ri, rj] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.outer(C_r, C_r)
            lift = (C_rr * float(N_total)) / denom
        lift = np.nan_to_num(lift, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(lift, 0.0)

        # Scale to [0, 1] so downstream normalization has a comparable range.
        max_val = float(lift.max())
        min_val = float(lift.min())
        if max_val > min_val:
            self.W_RR = (lift - min_val) / (max_val - min_val)
        else:
            self.W_RR = lift.astype(float)

        method = self.cfg.W_RR_normalize if normalize is None else normalize
        if method == "sigmoid":
            self.W_RR = signed_sigmoid(self.W_RR, float(self.cfg.W_RR_sigmoid_k))
        elif method == "max":
            max_value = float(self.W_RR.max())
            if max_value > 1e-12:
                self.W_RR /= max_value
        elif method == "none":
            pass
        else:
            raise ValueError("normalize must be 'sigmoid', 'max', or 'none'")

        self._erz_cache.clear()
        if verbose:
            print(
                f"[W_RR/cooccurrence] samples={N_total} "
                f"nnz={int(np.count_nonzero(self.W_RR > 1e-9))}/{self.K*self.K} "
                f"max={float(self.W_RR.max()):.4f}"
            )
        return self

    # ------------------------------------------------------------------
    # ERZ dynamics
    # ------------------------------------------------------------------
    def _run_erz(
        self,
        start_entity: int,
        relation: int,
        *,
        T: int,
        use_WRR: bool,
    ) -> Dict[int, float]:
        """Adjacency-based equivalent of a relation-conditioned ERZ episode.

        With W_RR off, the episode returns exactly the direct neighbors of
        `start_entity` under `relation` with score 1.0.  With W_RR on, the
        effective active relation set after the first two steps is
        {relation} ∪ {r_p | W_RR[r_p, relation] >= T_thresh}.  Each neighbor
        receives the maximum activity among the relations that reach it:
        1.0 for `relation` itself and W_RR[r_p, relation] for a similar
        relation r_p, exactly matching the dynamic E-readout.
        """
        h, r = int(start_entity), int(relation)
        if not (0 <= h < self.N and 0 <= r < self.K):
            raise ValueError(f"Query out of range: entity={h}, relation={r}")

        active_rels = {r: 1.0}
        if use_WRR and self.cfg.use_rr_recurrence:
            T_thresh = float(self.cfg.T_thresh)
            for r_p in range(self.K):
                w = self.W_RR[r_p, r]
                if r_p != r and w > 0.0 and w >= T_thresh:
                    active_rels[r_p] = float(w)

        scores: Dict[int, float] = {}
        for ar, weight in active_rels.items():
            if ar < self.M:
                for t in self._tails_by_hr.get((h, ar), set()):
                    if t != h:
                        scores[int(t)] = max(scores.get(int(t), 0.0), weight)
            else:
                for h2 in self._heads_by_tr.get((h, ar - self.M), set()):
                    if h2 != h:
                        scores[int(h2)] = max(scores.get(int(h2), 0.0), weight)
        return scores

    def _query_single(
        self,
        h: int,
        r: int,
        *,
        T: Optional[int] = None,
        use_WRR: bool = True,
    ) -> Dict[int, float]:
        h, r = int(h), int(r)
        steps = int(self.cfg.T_infer if T is None else T)
        key = (h, r, steps, bool(use_WRR), float(self.cfg.T_thresh))
        if self.cfg.enable_erz_cache:
            cached = self._erz_cache.get(key)
            if cached is not None:
                return cached
        if not use_WRR:
            # Fast path: direct neighbors with score 1.0.
            scores = {
                int(t): 1.0
                for t in self._active_targets.get((h, r), ())
                if int(t) != h
            }
        else:
            scores = self._run_erz(h, r, T=steps, use_WRR=True)
        if self.cfg.enable_erz_cache:
            self._erz_cache[key] = scores
        return scores

    # ------------------------------------------------------------------
    # ERZ-driven C learning
    # ------------------------------------------------------------------
    def _active_erz_targets(self, h: int, r: int, T: int) -> Tuple[int, ...]:
        """Return direct neighbors under relation r (ERZ with W_RR off).

        With W_RR disabled, the ERZ dynamics only propagates one step through
        the encoded triple.  Direct edges have activity 1.0, so any threshold
        ≤ 1.0 returns all cached neighbors.
        """
        if float(self.cfg.C_activity_threshold) > 1.0:
            return ()
        return self._active_targets.get((int(h), int(r)), ())

    def _reverse_consequent_activity(self, h: int, t: int) -> np.ndarray:
        """Return forward relations that directly connect h to t.

        The dynamic version activates all inverse-R neurons at t, runs one Z->Z
        step, and reads the relation activity at h.  Because Z-Z links encode
        exactly the train triples, the result is the set of forward relations r
        such that (h, r, t) is in the training set.
        """
        h, t = int(h), int(t)
        if not (0 <= h < self.N and 0 <= t < self.N):
            raise ValueError(f"Entity out of range in C return hop: h={h}, t={t}")

        y = np.zeros(self.K, dtype=np.float32)
        threshold = float(self.cfg.C_activity_threshold)
        if threshold <= 1.0:
            for r in self._rels_ht.get((h, t), set()):
                y[int(r)] = 1.0
        return y

    def _c_hebbian_update(self, r_i: int, r_j: int, y: np.ndarray, seen: int) -> None:
        mode = self.cfg.C_hebbian_mode
        if mode == "running_mean":
            eta = 1.0 / float(seen)
        elif mode == "ema":
            eta = float(self.cfg.delta_C)
            if not (0.0 < eta <= 1.0):
                raise ValueError("delta_C must be in (0,1] for EMA C learning")
        else:
            raise ValueError("C_hebbian_mode must be 'running_mean' or 'ema'")

        if y.shape != (self.K,):
            raise ValueError(f"C target activity must have shape ({self.K},), got {y.shape}")

        # Use full dense array when it fits in memory; otherwise per-pair vectors.
        if self._use_dense_c:
            w = self._W_dense[:, int(r_i), int(r_j)]
            np.clip(w + eta * (y - w), 0.0, 1.0, out=w)
        else:
            key = (int(r_i), int(r_j))
            d = self._W_comp.get(key)
            if d is None:
                d = np.zeros(self.K, dtype=np.float32)
                self._W_comp[key] = d
            np.clip(d + eta * (y - d), 0.0, 1.0, out=d)

    def _c_batched_update(
        self,
        r_i: int,
        r_j: int,
        y: np.ndarray,
        count: int,
        seen_before: int,
    ) -> None:
        """Apply `count` identical EMA updates at once.

        Repeating W <- clip(W + eta * (y - W)) `count` times with the same y
        is equivalent to:
            running_mean : W <- (seen * W + count * y) / (seen + count)
            ema          : W <- (1-eta)^count * W + (1 - (1-eta)^count) * y
        """
        if count <= 0:
            return
        mode = self.cfg.C_hebbian_mode
        if mode == "running_mean":
            total = float(seen_before + count)
            eta = float(count) / total
        elif mode == "ema":
            eta = 1.0 - (1.0 - float(self.cfg.delta_C)) ** count
        else:
            raise ValueError("C_hebbian_mode must be 'running_mean' or 'ema'")

        if y.shape != (self.K,):
            raise ValueError(f"C target activity must have shape ({self.K},), got {y.shape}")

        if self._use_dense_c:
            w = self._W_dense[:, int(r_i), int(r_j)]
            np.clip(w + eta * (y - w), 0.0, 1.0, out=w)
        else:
            key = (int(r_i), int(r_j))
            d = self._W_comp.get(key)
            if d is None:
                d = np.zeros(self.K, dtype=np.float32)
                self._W_comp[key] = d
            np.clip(d + eta * (y - d), 0.0, 1.0, out=d)

    def learn_C_layer(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """Learn C weights in the requested first-hop/second-hop/return-hop order.

        For each r_i in the 2M directed relation bank and each head h:
        1. activate r_i as external R input and run ERZ(h, r_i) to obtain x;
        2. for each selected x, activate every r_j one at a time and run
           ERZ(x, r_j) to obtain t; c_i and c_j jointly activate c_ij;
        3. for each selected t, extinguish c_i/c_j, activate all inverse R
           neurons at once, run one recurrent Z iteration, and read at h which
           forward r_k neurons return from t to h; these activities become c_k;
        4. update every W[k,i,j] locally toward the resulting K-vector y.

        Every concrete h -> x -> t neural path is one learning experience.
        Paths are not collapsed into a unique (h,t) endpoint set. W_RR remains
        disabled in the first and second ERZ hops, and no graph lookup is used.

        The actual implementation is selected by C_learning_algorithm:
            "online" -> per-path Hebbian updates
            "count"  -> one-pass explicit path counting
            "matrix" -> one-pass sparse-matrix path counting
        """
        algo = self.cfg.C_learning_algorithm
        if self.cfg.C_fast_approx_v2:
            algo = "matrix"
        elif self.cfg.C_fast_approx:
            algo = "count"

        if algo == "matrix":
            return self._learn_C_layer_matrix(T=T, seed=seed, reset=reset, verbose=verbose)
        if algo == "count":
            return self._learn_C_layer_count(T=T, seed=seed, reset=reset, verbose=verbose)
        return self._learn_C_layer_online(T=T, seed=seed, reset=reset, verbose=verbose)

    def _learn_C_layer_online(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """Standard C-layer learning with per-path Hebbian updates.

        For each r_i in the 2M directed relation bank and each head h:
        1. activate r_i as external R input and run ERZ(h, r_i) to obtain x;
        2. for each selected x, activate every r_j one at a time and run
           ERZ(x, r_j) to obtain t; c_i and c_j jointly activate c_ij;
        3. for each selected t, extinguish c_i/c_j, activate all inverse R
           neurons at once, run one recurrent Z iteration, and read at h which
           forward r_k neurons return from t to h; these activities become c_k;
        4. update every W[k,i,j] locally toward the resulting K-vector y.

        Sub-options (only used in this branch):
            C_fast_batching     : batch updates by (h,t) endpoint
            C_hebbian_mode      : "running_mean" or "ema"
            delta_C             : EMA step size
            C_n_passes          : number of passes over the training paths
            learn_cache_reverse : cache (h,t) -> forward relations
        """
        steps = int(self.cfg.C_learning_T if T is None else T)
        if steps <= 0:
            raise ValueError("C learning T must be positive")

        if reset:
            if self._use_dense_c:
                self._W_dense.fill(0.0)
                self._C_hit_dense.fill(0)
            self._W_comp.clear()
            self.C_support.fill(0)
            self._C_hits.clear()
            self._rules_by_consequent.clear()

        # Kept for API compatibility. The requested learning sequence is an
        # explicit deterministic set of nested for-loops, so it is not shuffled.
        _ = seed
        n_dynamic_calls_before = len(self._erz_cache)
        n_passes = max(1, int(self.cfg.C_n_passes))
        if self.cfg.C_hebbian_mode == "running_mean":
            n_passes = 1

        seen_by_pair = np.zeros((self.K, self.K), dtype=np.int64)
        total_path_samples = 0
        hebbian_mode = self.cfg.C_hebbian_mode
        delta_c = float(self.cfg.delta_C)

        # Cache for reverse consequent activity. The result only depends on
        # (h, t) and the fixed encoded triples, so the same (h, t) reached via
        # different intermediate entities x can reuse the cached vector.
        reverse_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = (
            {} if self.cfg.learn_cache_reverse else None
        )

        for pass_index in range(n_passes):
            count_statistics = pass_index == 0

            # Required ordering: choose r_i, then h, then each ERZ-generated x;
            # only after x is selected do we loop through r_j one at a time.
            # Use active-relation iteration in both dense and sparse modes.
            # Relations with no targets from the current entity contribute no
            # paths, so skipping them is result-identical and much faster when
            # the relation graph is sparse.
            activity_threshold = float(self.cfg.C_activity_threshold)
            use_dense = self._use_dense_c
            W_dense = self._W_dense
            C_hit_dense = self._C_hit_dense
            W_comp = self._W_comp
            C_hits = self._C_hits
            C_support = self.C_support

            if self.cfg.C_fast_batching:
                # Fast batching: for each (r_i, r_j) collect unique (h, t) pairs
                # and the number of intermediate x that connect them, then apply
                # one batched weight update per (h, t). This is exact for
                # running_mean and a close approximation for EMA; it avoids the
                # per-path overhead and unbounded reverse-cache growth that
                # makes exhaustive enumeration impractical on dense graphs.
                for r_i in range(self.K):
                    for r_j in range(self.K):
                        pair_counts: Dict[Tuple[int, int], int] = {}
                        for h in range(self.N):
                            if r_i not in self._active_rels_by_entity.get(h, ()):
                                continue
                            for x in self._active_erz_targets(h, r_i, steps):
                                if r_j not in self._active_rels_by_entity.get(x, ()):
                                    continue
                                for t in self._active_erz_targets(x, r_j, steps):
                                    if t == h:
                                        continue
                                    key = (h, t)
                                    pair_counts[key] = pair_counts.get(key, 0) + 1

                        if not pair_counts:
                            continue

                        n_samples = sum(pair_counts.values())
                        if count_statistics:
                            C_support[r_i, r_j] += n_samples
                            total_path_samples += n_samples

                        seen_before = int(seen_by_pair[r_i, r_j])
                        for (h, t), count in pair_counts.items():
                            y = self._reverse_consequent_activity(h, t)
                            if count_statistics:
                                if use_dense:
                                    C_hit_dense[:, r_i, r_j] += (
                                        (y >= activity_threshold).astype(np.int64) * count
                                    )
                                elif y.max() > 0.0:
                                    for r_k in np.nonzero(y >= activity_threshold)[0]:
                                        r_k = int(r_k)
                                        C_hits[(r_k, r_i, r_j)] = (
                                            C_hits.get((r_k, r_i, r_j), 0) + count
                                        )
                            self._c_batched_update(r_i, r_j, y, count, seen_before)
                            seen_before += count
                        seen_by_pair[r_i, r_j] = seen_before
            else:
                for h in range(self.N):
                    for r_i in self._active_rels_by_entity.get(h, ()):
                        intermediates = self._active_erz_targets(h, r_i, steps)
                        for x in intermediates:
                            for r_j in self._active_rels_by_entity.get(x, ()):
                                tails = self._active_erz_targets(x, r_j, steps)
                                for t in tails:
                                    if t == h:
                                        continue

                                    # Third hop: all inverse R neurons are co-active;
                                    # c_i and c_j are not carried into this readout.
                                    if reverse_cache is not None:
                                        key = (h, t)
                                        y = reverse_cache.get(key)
                                        if y is None:
                                            y = self._reverse_consequent_activity(h, t)
                                            reverse_cache[key] = y
                                    else:
                                        y = self._reverse_consequent_activity(h, t)

                                    if count_statistics:
                                        C_support[r_i, r_j] += 1
                                        if use_dense:
                                            C_hit_dense[:, r_i, r_j] += (
                                                y >= activity_threshold
                                            ).astype(np.int64)
                                        elif y.max() > 0.0:
                                            for r_k in np.nonzero(y >= activity_threshold)[0]:
                                                r_k = int(r_k)
                                                C_hits[(r_k, r_i, r_j)] = (
                                                    C_hits.get((r_k, r_i, r_j), 0) + 1
                                                )
                                        total_path_samples += 1

                                    seen = int(seen_by_pair[r_i, r_j] + 1)
                                    seen_by_pair[r_i, r_j] = seen
                                    if hebbian_mode == "running_mean":
                                        eta = 1.0 / float(seen)
                                    else:
                                        eta = delta_c
                                    if use_dense:
                                        w = W_dense[:, r_i, r_j]
                                        w[:] = w + eta * (y - w)
                                        w.clip(0.0, 1.0, out=w)
                                    else:
                                        d = W_comp.get((r_i, r_j))
                                        if d is None:
                                            d = np.zeros(self.K, dtype=np.float32)
                                            W_comp[(r_i, r_j)] = d
                                        d[:] = d + eta * (y - d)
                                        d.clip(0.0, 1.0, out=d)

        active_antecedent_pairs = int(np.count_nonzero(self.C_support > 0))
        active_rules = self._build_rules_by_consequent()
        self.C_learning_stats = {
            "source": "online_per_path_hebbian_update",
            "T": steps,
            "activity_threshold": float(self.cfg.C_activity_threshold),
            "antecedent_pairs_with_support": active_antecedent_pairs,
            "path_experience_samples": total_path_samples,
            # Retained for compatibility with callers that used the old key.
            "unique_endpoint_samples": total_path_samples,
            "active_rules": active_rules,
            "erz_cache_entries_added": max(0, len(self._erz_cache) - n_dynamic_calls_before),
        }

        if verbose:
            all_ws = [w for d in self._W_comp.values() for w in d.values()]
            print(
                "[C/online] "
                f"pairs={active_antecedent_pairs}/{self.K*self.K} "
                f"samples={total_path_samples} active_rules={active_rules} "
                f"mean_w={float(sum(all_ws)/len(all_ws)) if all_ws else 0.0:.4f} "
                f"max_w={float(max(all_ws)) if all_ws else 0.0:.4f}"
            )
        return self

    def _learn_C_layer_count(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """One-pass C-layer learning via explicit path counting.

        Instead of applying an EMA update for every concrete h -> x -> t path,
        this method counts paths and directly computes the path-weighted mean
        confidence for each consequent r_k:

            W[r_k, r_i, r_j] = (#paths h -(r_i)-> x -(r_j)-> t with h -(r_k)-> t)
                               / (#paths h -(r_i)-> x -(r_j)-> t)

        This is mathematically identical to the running_mean branch of the
        standard `_learn_C_layer_online` but avoids per-path weight-vector
        read/write and EMA arithmetic. It is therefore significantly faster on
        large graphs while producing exactly the same active rules and
        confidences.

        The method uses explicit Python enumeration over two-hop paths. For very
        large or dense graphs, prefer `_learn_C_layer_matrix`.
        """
        steps = int(self.cfg.C_learning_T if T is None else T)
        if steps <= 0:
            raise ValueError("C learning T must be positive")
        _ = seed  # deterministic; kept for API compatibility
        n_dynamic_calls_before = len(self._erz_cache)

        if reset:
            if self._use_dense_c:
                self._W_dense.fill(0.0)
                self._C_hit_dense.fill(0)
            self._W_comp.clear()
            self.C_support.fill(0)
            self._C_hits.clear()
            self._rules_by_consequent.clear()

        activity_threshold = float(self.cfg.C_activity_threshold)
        use_dense = self._use_dense_c
        W_dense = self._W_dense
        C_hit_dense = self._C_hit_dense
        W_comp = self._W_comp
        C_hits = self._C_hits
        C_support = self.C_support

        total_path_samples = 0

        # Precompute transposed adjacency tables: for each directed relation r,
        # x_to_h[r][x] = heads h such that h -(r)-> x, and
        # x_to_t[r][x] = tails  t such that x -(r)-> t.
        # These are exactly the information in _active_targets, transposed for
        # fast intersection-based enumeration of two-hop paths per (r_i, r_j).
        x_to_h_by_rel: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(self.K)
        ]
        x_to_t_by_rel: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(self.K)
        ]
        for h in range(self.N):
            for r_i in self._active_rels_by_entity.get(h, ()):
                for x in self._active_targets.get((h, r_i), ()):
                    x_to_h_by_rel[r_i][x].append(h)
        for x in range(self.N):
            for r_j in self._active_rels_by_entity.get(x, ()):
                for t in self._active_targets.get((x, r_j), ()):
                    x_to_t_by_rel[r_j][x].append(t)

        for r_i in range(self.K):
            x_to_h = x_to_h_by_rel[r_i]
            if not x_to_h:
                continue
            for r_j in range(self.K):
                x_to_t = x_to_t_by_rel[r_j]
                if not x_to_t:
                    continue
                common_x = x_to_h.keys() & x_to_t.keys()
                if not common_x:
                    continue

                n_samples = 0
                if use_dense:
                    conf_vec = np.zeros(self.K, dtype=np.float32)
                    hit_vec = np.zeros(self.K, dtype=np.int64)
                else:
                    conf_vec = np.zeros(self.K, dtype=np.float32)
                    hit_dict: Dict[Tuple[int, int, int], int] = {}

                # Cache the set of forward r_k that directly connect (h, t).
                # Within one (r_i, r_j) pair many paths can share the same (h, t),
                # so this avoids repeated lookups in self._rels_ht.
                reverse_cache: Dict[Tuple[int, int], Set[int]] = {}

                for x in common_x:
                    for h in x_to_h[x]:
                        for t in x_to_t[x]:
                            if h == t:
                                continue
                            n_samples += 1
                            if activity_threshold <= 1.0:
                                key = (h, t)
                                y = reverse_cache.get(key)
                                if y is None:
                                    y = set(self._rels_ht.get(key, ()))
                                    reverse_cache[key] = y
                                for r_k in y:
                                    r_k = int(r_k)
                                    conf_vec[r_k] += 1.0
                                    if use_dense:
                                        hit_vec[r_k] += 1
                                    else:
                                        hit_dict[(r_k, r_i, r_j)] = (
                                            hit_dict.get((r_k, r_i, r_j), 0) + 1
                                        )

                if n_samples == 0:
                    continue

                C_support[r_i, r_j] = n_samples
                total_path_samples += n_samples

                if use_dense:
                    conf_vec /= float(n_samples)
                    W_dense[:, r_i, r_j] = conf_vec
                    C_hit_dense[:, r_i, r_j] = hit_vec
                else:
                    conf_vec /= float(n_samples)
                    W_comp[(r_i, r_j)] = conf_vec
                    C_hits.update(hit_dict)

        active_antecedent_pairs = int(np.count_nonzero(C_support > 0))
        active_rules = self._build_rules_by_consequent()
        self.C_learning_stats = {
            "source": "count_one_pass_path_weighted_running_mean",
            "T": steps,
            "activity_threshold": activity_threshold,
            "antecedent_pairs_with_support": active_antecedent_pairs,
            "path_experience_samples": total_path_samples,
            "unique_endpoint_samples": total_path_samples,
            "active_rules": active_rules,
            "erz_cache_entries_added": max(0, len(self._erz_cache) - n_dynamic_calls_before),
        }

        if verbose:
            all_ws = [w for d in self._W_comp.values() for w in d.values()]
            print(
                "[C/count] "
                f"pairs={active_antecedent_pairs}/{self.K*self.K} "
                f"samples={total_path_samples} active_rules={active_rules} "
                f"mean_w={float(sum(all_ws)/len(all_ws)) if all_ws else 0.0:.4f} "
                f"max_w={float(max(all_ws)) if all_ws else 0.0:.4f}"
            )
        return self

    def _learn_C_layer_matrix(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZC":
        """One-pass C-layer learning via sparse-matrix path counting.

        This is mathematically identical to `_learn_C_layer_count` (and therefore
        to the running_mean branch of the full online update): it computes, for
        each antecedent pair (r_i, r_j) and each forward consequent r_k, the
        path-weighted conditional probability

            W[r_k, r_i, r_j] = |{(h,x,t): h-(r_i)->x-(r_j)->t and (h,r_k,t) in T}|
                               / |{(h,x,t): h-(r_i)->x-(r_j)->t}|

        using scipy.sparse matrix multiplication. The result is written into the
        same internal structures (`_W_dense` or `_W_comp`) so that inference is
        unchanged.

        The speedup comes from avoiding the Python triple loop over paths; for
        very large or dense graphs (e.g. YAGO3-10) this is orders of magnitude
        faster than the explicit enumeration in `_learn_C_layer_count`.
        """
        steps = int(self.cfg.C_learning_T if T is None else T)
        if steps <= 0:
            raise ValueError("C learning T must be positive")
        _ = seed  # deterministic; kept for API compatibility
        n_dynamic_calls_before = len(self._erz_cache)

        if reset:
            if self._use_dense_c:
                self._W_dense.fill(0.0)
                self._C_hit_dense.fill(0)
            self._W_comp.clear()
            self.C_support.fill(0)
            self._C_hits.clear()
            self._rules_by_consequent.clear()

        activity_threshold = float(self.cfg.C_activity_threshold)
        if activity_threshold > 1.0:
            # No active targets; leave weights zero.
            self.C_learning_stats = {
                "source": "matrix_sparse_path_weighted_running_mean",
                "T": steps,
                "activity_threshold": activity_threshold,
                "antecedent_pairs_with_support": 0,
                "path_experience_samples": 0,
                "unique_endpoint_samples": 0,
                "active_rules": 0,
                "erz_cache_entries_added": max(0, len(self._erz_cache) - n_dynamic_calls_before),
            }
            return self

        use_dense = self._use_dense_c
        W_dense = self._W_dense
        C_hit_dense = self._C_hit_dense
        W_comp = self._W_comp
        C_hits = self._C_hits
        C_support = self.C_support

        # Build one sparse adjacency matrix per directed relation r in [0, K).
        # A_r[h, t] = 1 iff h -(r)-> t in the encoded graph.
        row_inds: List[List[int]] = [[] for _ in range(self.K)]
        col_inds: List[List[int]] = [[] for _ in range(self.K)]
        data_vals: List[List[int]] = [[] for _ in range(self.K)]

        for h, r, t in self._encoded_triples:
            # forward relation r
            row_inds[r].append(int(h))
            col_inds[r].append(int(t))
            data_vals[r].append(1)
            # inverse relation r+M
            r_inv = int(r) + self.M
            row_inds[r_inv].append(int(t))
            col_inds[r_inv].append(int(h))
            data_vals[r_inv].append(1)

        A: List[sp.csr_matrix] = []
        for r in range(self.K):
            if row_inds[r]:
                A.append(sp.csr_matrix(
                    (data_vals[r], (row_inds[r], col_inds[r])),
                    shape=(self.N, self.N), dtype=np.int32
                ))
            else:
                A.append(sp.csr_matrix((self.N, self.N), dtype=np.int32))

        total_path_samples = 0

        for r_i in range(self.K):
            Ai = A[r_i]
            if Ai.nnz == 0:
                continue
            for r_j in range(self.K):
                Aj = A[r_j]
                if Aj.nnz == 0:
                    continue
                # S[h, t] = number of intermediate x such that
                # h -(r_i)-> x -(r_j)-> t.
                S = Ai @ Aj
                S.setdiag(0)
                S.eliminate_zeros()
                support = int(S.sum())
                if support == 0:
                    continue

                C_support[r_i, r_j] = support
                total_path_samples += support

                if use_dense:
                    hit_vec = np.zeros(self.K, dtype=np.int64)
                    conf_vec = None
                else:
                    conf_vec = np.zeros(self.K, dtype=np.float32)

                # For each forward consequent r_k, count paths whose endpoints
                # (h, t) are also connected by r_k.
                for r_k in range(self.M):
                    H = A[r_k].multiply(S)
                    hits = int(H.sum())
                    if hits == 0:
                        continue
                    conf = float(hits) / float(support)
                    if use_dense:
                        W_dense[r_k, r_i, r_j] = conf
                        hit_vec[r_k] = hits
                    else:
                        conf_vec[r_k] = conf
                        C_hits[(r_k, r_i, r_j)] = hits

                if use_dense:
                    C_hit_dense[:, r_i, r_j] = hit_vec
                else:
                    W_comp[(r_i, r_j)] = conf_vec

        active_antecedent_pairs = int(np.count_nonzero(C_support > 0))
        active_rules = self._build_rules_by_consequent()
        self.C_learning_stats = {
            "source": "matrix_sparse_path_weighted_running_mean",
            "T": steps,
            "activity_threshold": activity_threshold,
            "antecedent_pairs_with_support": active_antecedent_pairs,
            "path_experience_samples": total_path_samples,
            "unique_endpoint_samples": total_path_samples,
            "active_rules": active_rules,
            "erz_cache_entries_added": max(0, len(self._erz_cache) - n_dynamic_calls_before),
        }

        if verbose:
            all_ws = [w for d in self._W_comp.values() for w in d.values()]
            print(
                "[C/matrix] "
                f"pairs={active_antecedent_pairs}/{self.K*self.K} "
                f"samples={total_path_samples} active_rules={active_rules} "
                f"mean_w={float(sum(all_ws)/len(all_ws)) if all_ws else 0.0:.4f} "
                f"max_w={float(max(all_ws)) if all_ws else 0.0:.4f}"
            )
        return self

    def _build_rules_by_consequent(self) -> int:
        """Build the sparse rule index used for inference and return active rule count."""
        threshold = float(self.cfg.C_rule_threshold)
        min_support = int(self.cfg.C_min_support)
        self._rules_by_consequent = {}
        active_rules = 0

        if self._use_dense_c:
            mask = (
                (self._W_dense >= threshold)
                & (self.C_support[None, :, :] >= min_support)
            )
            for r_k, r_i, r_j in zip(*np.nonzero(mask)):
                self._rules_by_consequent.setdefault(int(r_k), []).append(
                    (int(r_i), int(r_j), float(self._W_dense[r_k, r_i, r_j]))
                )
                active_rules += 1
        else:
            for (r_i, r_j), vec in self._W_comp.items():
                if self.C_support[r_i, r_j] < min_support:
                    continue
                active_indices = np.nonzero(vec >= threshold)[0]
                for r_k in active_indices:
                    r_k = int(r_k)
                    self._rules_by_consequent.setdefault(r_k, []).append(
                        (int(r_i), int(r_j), float(vec[r_k]))
                    )
                    active_rules += 1
        return active_rules

    # ------------------------------------------------------------------
    # Fallback mechanisms (adapted from NSR_Simplified/NSR_Simplified_model.py)
    # ------------------------------------------------------------------
    def _compute_concepts_similarity(self, head_ids: np.ndarray) -> np.ndarray:
        """Structural-equivalence cosine similarity for a batch of query heads.

        For each forward relation r we count common out-neighbors (A_r[h,:] @ A_r.T)
        and common in-neighbors (A_r[:,h].T @ A_r).  Counts are normalised by
        sqrt(deg(h) * deg(e)), matching the original Concepts_Fallback (formerly
        re_fallback_v2) implementation.
        """
        head_ids = np.asarray(head_ids, dtype=np.int32)
        U = head_ids.shape[0]
        S = np.zeros((U, self.N), dtype=np.float32)
        temp = np.zeros((U, self.N), dtype=np.float32)

        if self._entity_deg is None:
            return S
        deg = self._entity_deg
        deg_safe = np.where(deg > 0, deg, 1.0)
        h_deg = deg_safe[head_ids][:, None]

        for r in range(self.M):
            A = self._rel_adj[r]
            if A.nnz == 0:
                continue
            C = A[head_ids, :] @ A.T
            if C.nnz:
                C.toarray(out=temp)
                np.add(S, temp, out=S)
            C = A[:, head_ids].T @ A
            if C.nnz:
                C.toarray(out=temp)
                np.add(S, temp, out=S)

        norm = np.sqrt(h_deg * deg_safe[None, :])
        np.divide(S, norm, out=S, where=norm != 0)
        return S.astype(np.float64, copy=False)

    def _concepts_fallback_infer(self, h: int, r_k: int) -> Dict[int, float]:
        """Concepts_Fallback (formerly re_fallback_v2).

        When rule-based scores are empty, find the top-k most structurally
        similar training heads for relation r_k, run inference for each of them,
        and aggregate the resulting tail scores weighted by similarity.
        """
        if r_k >= self.M or self.cfg.concepts_fallback_weight <= 0.0:
            return {}

        h = int(h)
        if h in self._concepts_sim_cache:
            sims = self._concepts_sim_cache[h]
        else:
            sims = self._compute_concepts_similarity(np.array([h], dtype=np.int32))[0]
            self._concepts_sim_cache[h] = sims

        cand_heads = np.array(sorted(self._heads_by_rel[r_k]), dtype=np.int32)
        if cand_heads.shape[0] == 0:
            return {}

        cand_sims = sims[cand_heads].copy()
        self_pos = np.where(cand_heads == h)[0]
        if self_pos.size:
            cand_sims[self_pos[0]] = -1.0

        top_k = min(int(self.cfg.concepts_fallback_k), cand_heads.shape[0])
        top_idx = np.argpartition(cand_sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-cand_sims[top_idx])]

        aggregated: Dict[int, float] = {}
        weight = float(self.cfg.concepts_fallback_weight)
        for h_i, w_i in zip(cand_heads[top_idx], cand_sims[top_idx]):
            if w_i <= 0.0:
                continue
            # Neighbor inference must NOT recurse into Concepts_Fallback again.
            neighbor_scores = self._query_with_C(
                int(h_i), r_k, T=self.cfg.T_infer, use_WRR=True,
                agg_mode="max", alpha=1.0, top_rules_fraction=1.0,
                _allow_concepts_fallback=False,
            )
            if not neighbor_scores:
                continue
            scale = weight * float(w_i)
            for e, s in neighbor_scores.items():
                aggregated[int(e)] = aggregated.get(int(e), 0.0) + scale * float(s)
        return aggregated

    def _tail_fallback_infer(self, h: int, r_k: int) -> Dict[int, float]:
        """Relation-specific tail-popularity fallback.

        When rule-based scores are empty, boost entities that frequently appear
        as tails of r_k.  The score is proportional to the tail's empirical
        popularity for this relation.
        """
        if r_k >= self.M or self.cfg.tail_fallback_weight <= 0.0:
            return {}
        max_count = self._tail_pop_max.get(r_k, 1.0)
        weight = float(self.cfg.tail_fallback_weight)
        scores: Dict[int, float] = {}
        for t, c in self._tail_pop[r_k].items():
            scores[int(t)] = weight * (float(c) / max_count)
        return scores

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _query_with_C(
        self,
        h: int,
        r_k: int,
        *,
        T: int,
        use_WRR: bool,
        agg_mode: str,
        alpha: float,
        top_rules_fraction: float = 1.0,
        _allow_concepts_fallback: bool = True,
    ) -> Dict[int, float]:
        scores_direct = self._query_single(h, r_k, T=T, use_WRR=use_WRR)

        # Sparse rule lookup: only rules whose consequent is r_k and which already
        # passed the support/threshold filter during learning.
        active_rules = self._rules_by_consequent.get(int(r_k), [])

        # Optionally keep only the highest-confidence rules.
        if top_rules_fraction is not None and 0.0 < top_rules_fraction < 1.0:
            if len(active_rules) > 0:
                keep_count = max(1, int(np.ceil(len(active_rules) * top_rules_fraction)))
                # active_rules is (r_i, r_j, confidence); sort by confidence.
                active_rules = sorted(active_rules, key=lambda x: x[2])[-keep_count:]

        scores_comp: Dict[int, float] = {}
        for r_i, r_j, confidence in active_rules:
            # Rule execution is dynamic but exact with respect to its selected
            # relation neurons: W_RR is deliberately off in both antecedent hops.
            for x, score_hx in self._query_single(h, r_i, T=T, use_WRR=False).items():
                if score_hx <= 0.0:
                    continue
                for t, score_xt in self._query_single(x, r_j, T=T, use_WRR=False).items():
                    if t == h or score_xt <= 0.0:
                        continue
                    path_score = float(score_hx) * float(score_xt) * confidence
                    scores_comp[int(t)] = max(scores_comp.get(int(t), 0.0), path_score)

        scores = dict(scores_direct)
        if agg_mode == "max":
            for t, score in scores_comp.items():
                scores[t] = max(scores.get(t, 0.0), score)
        elif agg_mode == "add":
            for t, score in scores_comp.items():
                scores[t] = scores.get(t, 0.0) + float(alpha) * score
        else:
            raise ValueError("agg_mode must be 'max' or 'add'")

        # Optional fallbacks (order: Concepts_Fallback first, then tail-popularity).
        apply_tail_fallback = (
            self.cfg.tail_fallback_weight > 0.0
            and r_k < self.M
        )
        if (
            _allow_concepts_fallback
            and not scores
            and self.cfg.concepts_fallback_weight > 0.0
            and r_k < self.M
        ):
            concepts_scores = self._concepts_fallback_infer(int(h), int(r_k))
            if concepts_scores:
                scores = concepts_scores
                apply_tail_fallback = False

        if apply_tail_fallback and not scores:
            scores = self._tail_fallback_infer(int(h), int(r_k))

        return scores

    def query(
        self,
        h: int,
        r: int,
        *,
        T: Optional[int] = None,
        use_WRR: bool = True,
        use_C_layer: Optional[bool] = None,
        agg_mode: str = "max",
        alpha: Optional[float] = None,
        top_rules_fraction: float = 1.0,
    ) -> Dict[int, float]:
        steps = int(self.cfg.T_infer if T is None else T)
        use_c = self.cfg.enable_C_layer_by_default if use_C_layer is None else bool(use_C_layer)
        alpha_value = float(self.cfg.alpha if alpha is None else alpha)
        if use_c:
            return self._query_with_C(
                int(h), int(r), T=steps, use_WRR=bool(use_WRR),
                agg_mode=agg_mode, alpha=alpha_value,
                top_rules_fraction=top_rules_fraction,
            )
        return self._query_single(int(h), int(r), T=steps, use_WRR=bool(use_WRR))

    # ------------------------------------------------------------------
    # Model persistence (avoids re-learning across reasoning experiments)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save learned weights and encoded triples to an .npz file."""
        self._ensure_sparse()
        kwargs: Dict[str, Any] = {
            "N": self.N,
            "M": self.M,
            "W_RR": self.W_RR,
            "C_support": self.C_support,
            "use_dense_c": self._use_dense_c,
            "zr_keys": np.array(list(self._zr_entries.keys()), dtype=np.int64),
            "zr_vals": np.array(list(self._zr_entries.values()), dtype=float),
            "zz_keys": np.array(list(self._zz_entries.keys()), dtype=np.int64),
            "zz_vals": np.array(list(self._zz_entries.values()), dtype=float),
            "encoded_triples": np.array(self._encoded_triples, dtype=np.int64),
            "cfg_dict": np.array(str(self.cfg.to_dict()), dtype=object),
        }
        hit_keys = list(self._C_hits.keys())
        hit_vals = [self._C_hits[k] for k in hit_keys]
        kwargs["hit_keys"] = (
            np.array(hit_keys, dtype=np.int64) if hit_keys else np.zeros((0, 3), dtype=np.int64)
        )
        kwargs["hit_vals"] = (
            np.array(hit_vals, dtype=np.int64) if hit_vals else np.zeros(0, dtype=np.int64)
        )

        if self._use_dense_c:
            kwargs["W_comp_agent_dense"] = self._W_dense
            kwargs["C_hit_dense"] = self._C_hit_dense
        else:
            comp_keys = []
            comp_vals = []
            for (r_i, r_j), vec in self._W_comp.items():
                for r_k in np.nonzero(vec > 0.0)[0]:
                    comp_keys.append((int(r_k), int(r_i), int(r_j)))
                    comp_vals.append(float(vec[int(r_k)]))
            kwargs["comp_keys"] = (
                np.array(comp_keys, dtype=np.int64) if comp_keys else np.zeros((0, 3), dtype=np.int64)
            )
            kwargs["comp_vals"] = (
                np.array(comp_vals, dtype=np.float32) if comp_vals else np.zeros(0, dtype=np.float32)
            )
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path: str) -> "NSRDynamicERZC":
        """Load a model previously saved with ``save``."""
        data = np.load(path, allow_pickle=True)
        N = int(data["N"])
        M = int(data["M"])
        cfg = NSRConfig.from_dict(eval(str(data["cfg_dict"])))
        model = cls(N, M, cfg)

        model.W_RR = data["W_RR"]
        model.C_support = data["C_support"]

        use_dense = bool(data.get("use_dense_c", False))
        if use_dense and "W_comp_agent_dense" in data:
            model._W_dense = data["W_comp_agent_dense"]
            model._C_hit_dense = data.get("C_hit_dense", np.zeros((model.K, model.K, model.K), dtype=np.int64))
            model._use_dense_c = True
            model._W_comp = {}
        else:
            model._W_comp = {}
            comp_keys = data.get("comp_keys", np.zeros((0, 3), dtype=np.int64))
            comp_vals = data.get("comp_vals", np.zeros(0, dtype=np.float32))
            if len(comp_keys):
                for key, val in zip(comp_keys, comp_vals):
                    r_k, r_i, r_j = int(key[0]), int(key[1]), int(key[2])
                    d = model._W_comp.get((r_i, r_j))
                    if d is None:
                        d = np.zeros(model.K, dtype=np.float32)
                        model._W_comp[(r_i, r_j)] = d
                    d[r_k] = float(val)
            if use_dense and model._use_dense_c:
                # Migrate sparse save back into dense representation.
                model._W_dense.fill(0.0)
                for (r_i, r_j), vec in model._W_comp.items():
                    model._W_dense[:, r_i, r_j] = vec
                model._W_comp.clear()
            else:
                model._use_dense_c = False

        model._C_hits = {}
        hit_keys = data.get("hit_keys", np.zeros((0, 3), dtype=np.int64))
        hit_vals = data.get("hit_vals", np.zeros(0, dtype=np.int64))
        if len(hit_keys):
            for key, val in zip(hit_keys, hit_vals):
                r_k, r_i, r_j = int(key[0]), int(key[1]), int(key[2])
                model._C_hits[(r_k, r_i, r_j)] = int(val)

        if len(data["zr_keys"]):
            model._zr_entries = {
                tuple(int(k) for k in key): float(val)
                for key, val in zip(data["zr_keys"], data["zr_vals"])
            }
        if len(data["zz_keys"]):
            model._zz_entries = {
                tuple(int(k) for k in key): float(val)
                for key, val in zip(data["zz_keys"], data["zz_vals"])
            }
        model._encoded_triples = [tuple(int(x) for x in row) for row in data["encoded_triples"]]
        model._rebuild_adjacency_tables()
        model._build_rules_by_consequent()
        model._sparse_dirty = True
        model._erz_cache.clear()
        return model

    # ------------------------------------------------------------------
    # Lightweight rule inspection (not used for inference)
    # ------------------------------------------------------------------
    def _relation_name(self, relation: int, id2rel: Optional[Sequence[str]]) -> str:
        relation = int(relation)
        base = relation if relation < self.M else relation - self.M
        name = str(id2rel[base]) if id2rel is not None else f"r{base}"
        return name if relation < self.M else f"{name}^-1"

    def _make_rule_dict(
        self,
        r_i: int,
        r_j: int,
        r_k: int,
        id2rel: Optional[Sequence[str]],
    ) -> Dict[str, int | float | str]:
        if self._use_dense_c:
            confidence = float(self._W_dense[r_k, r_i, r_j])
        else:
            confidence = float(self._W_comp[(r_i, r_j)][r_k])
        return {
            "rule": (
                f"{self._relation_name(r_i, id2rel)} + "
                f"{self._relation_name(r_j, id2rel)} => "
                f"{self._relation_name(r_k, id2rel)}"
            ),
            "r_i": r_i,
            "r_j": r_j,
            "r_k": r_k,
            "confidence": confidence,
            "support": int(self.C_support[r_i, r_j]),
            "hits": int(
                self._C_hit_dense[r_k, r_i, r_j]
                if self._use_dense_c
                else self._C_hits.get((r_k, r_i, r_j), 0)
            ),
        }

    def get_learned_rules(
        self,
        id2rel: Optional[Sequence[str]] = None,
        *,
        threshold: Optional[float] = None,
        min_support: Optional[int] = None,
    ) -> List[Dict[str, int | float | str]]:
        threshold_value = float(self.cfg.C_rule_threshold if threshold is None else threshold)
        support_value = int(self.cfg.C_min_support if min_support is None else min_support)
        rules: List[Dict[str, int | float | str]] = []

        if self._use_dense_c:
            for r_k in range(self.K):
                active = np.argwhere(
                    (self._W_dense[r_k] >= threshold_value)
                    & (self.C_support >= support_value)
                )
                for raw_i, raw_j in active:
                    r_i, r_j = int(raw_i), int(raw_j)
                    rules.append(self._make_rule_dict(r_i, r_j, r_k, id2rel))
        else:
            for (r_i, r_j), vec in self._W_comp.items():
                if self.C_support[r_i, r_j] < support_value:
                    continue
                active_indices = np.nonzero(vec >= threshold_value)[0]
                for r_k in active_indices:
                    r_k = int(r_k)
                    rules.append(self._make_rule_dict(r_i, r_j, r_k, id2rel))

        rules.sort(
            key=lambda rule: (
                -float(rule["confidence"]),
                -int(rule["support"]),
                int(rule["r_k"]),
                int(rule["r_i"]),
                int(rule["r_j"]),
            )
        )
        return rules
