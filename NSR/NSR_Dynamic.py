"""Minimal Dynamic NSR with ERZ-driven compositional-rule learning.

The model contains only the mechanisms needed for:
1. encoding triples into E-R-Z synaptic connectivity;
2. learning relation-relation weights W_RR with a local Oja/Hebbian rule;
3. running the five-step ERZ dynamics;
4. learning 2M x 2M compositional antecedents from ERZ activations;
5. answering direct and compositional tail queries.

Important design decision
-------------------------
C_support and C_hit_count are NOT built from adjacency lists or direct triple
lookups. Every first hop, second hop, and direct consequent check is obtained by
running the same ERZ dynamics with W_RR disabled. A small query cache may reuse
an already-computed ERZ readout, but the cached value is a neural-dynamics
result, not an adjacency-table result.
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
    C_min_support: int = 0

    # Faithful C dynamics extra threshold
    C_comp_threshold: float = 1.5

    # C inference
    enable_C_layer_by_default: bool = True
    C_rule_threshold: float = 0.1
    alpha: float = 1.0

    # Performance only: cached values are outputs of ERZ runs, not graph tables.
    enable_erz_cache: bool = True

    # If True, cache _reverse_consequent_activity(h, t) during C learning to avoid
    # recomputing the same (h, t) return hop for different intermediate entities x.
    learn_cache_reverse: bool = True

    def to_dict(self) -> Dict[str, int | float | bool | str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, int | float | bool | str]) -> "NSRConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NSRDynamicERZCBase:
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

        # Both the C-agent bank and each axis of the C-composition bank use 2M
        # directed relations. W_comp_agent[k, i, j] connects c_k to c_ij.
        self.W_comp_agent = np.zeros((self.K, self.K, self.K), dtype=np.float32)
        self.C_support = np.zeros((self.K, self.K), dtype=np.int32)
        self.C_hit_count = np.zeros((self.K, self.K, self.K), dtype=np.int32)
        self.C_learning_stats: Dict[str, int | float | str] = {}

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
    def encode_triples(self, triples: Iterable[Triple]) -> "NSRDynamicERZCBase":
        for raw_h, raw_r, raw_t in triples:
            h, r, t = int(raw_h), int(raw_r), int(raw_t)
            if not (0 <= h < self.N and 0 <= t < self.N and 0 <= r < self.M):
                raise ValueError(f"Triple out of range: {(h, r, t)}")

            r_inv = r + self.M
            z_head = self.z_idx(h, r)
            z_tail_inv = self.z_idx(t, r_inv)

            # R selects a relation-indexed Z cell only where that relation occurs.
            self._zr_entries[(r, z_head)] = 1.0
            self._zr_entries[(r_inv, z_tail_inv)] = 1.0

            # Bidirectional Z-Z link supports both forward and inverse retrieval.
            strength = float(self.cfg.w_zz_strength)
            self._zz_entries[(z_head, z_tail_inv)] = strength
            self._zz_entries[(z_tail_inv, z_head)] = strength
            self._encoded_triples.append((h, r, t))

        self._sparse_dirty = True
        self._erz_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Local W_RR Hebbian/Oja learning
    # ------------------------------------------------------------------
    def learn_WRR_hebbian(
        self,
        triples: Optional[Sequence[Triple]] = None,
        *,
        n_passes: int = 1,
        normalize: Optional[str] = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> "NSRDynamicERZCBase":
        """Learn W_RR from co-active directed relation neurons.

        Each directed entity pair is one experience. Its K-dimensional relation
        activity vector has a 1 for every relation neuron active on that pair.
        The update for each experience is local:

            dW_ij = delta_RR * x_i*x_j
                    - mu_RR * (x_i^2 + x_j^2) * W_ij.
        """
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
                f"[W_RR] samples={len(samples)} passes={max(1, int(n_passes))} "
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
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        """Run one relation-conditioned ERZ episode and return final E activity."""
        h, r = int(start_entity), int(relation)
        if not (0 <= h < self.N and 0 <= r < self.K):
            raise ValueError(f"Query out of range: entity={h}, relation={r}")

        I_E = np.zeros(self.N, dtype=float)
        I_R = np.zeros(self.K, dtype=float)
        I_E[h] = 1.0
        I_R[r] = 1.0

        X_E = np.zeros(self.N, dtype=float)
        X_R = np.zeros(self.K, dtype=float)
        X_Z = np.zeros(self.NZ, dtype=float)

        cfg = self.cfg
        zero_R = np.zeros(self.K, dtype=float)
        zero_E = np.zeros(self.N, dtype=float)

        for step in range(1, int(T) + 1):
            if cfg.hold_R_until_e_delay and step <= max(1, int(cfg.i_e_delay)):
                inp_R = I_R.copy()
            elif step == 1:
                inp_R = I_R.copy()
            else:
                inp_R = zero_R.copy()
            if step >= int(cfg.inhibit_R_from):
                inp_R -= float(cfg.inhibit_value)

            inp_E = I_E if step == int(cfg.i_e_delay) else zero_E

            # Z(t+1) from E(t), R(t), Z(t)
            zz_base = self.W_ZZ @ X_Z
            zz_drive = zz_base + float(cfg.b_bias) * (zz_base > 0.0).astype(float)
            ez_drive = self.W_EZ @ X_E
            rz_drive = self.W_ZR.T @ sigma_gate(X_R, float(cfg.T_thresh))
            X_Z_new = gated_relu(zz_drive + ez_drive + rz_drive, float(cfg.eta_Z))

            # E(t+1) reads Z(t), preserving the synchronous one-step delay.
            readout = X_Z.reshape(self.N, self.K).max(axis=1)
            X_E_new = np.clip(
                gated_relu(readout, float(cfg.eta_E)) + inp_E,
                0.0,
                1.0,
            )

            # R(t+1) receives recurrent relation activity only when explicitly enabled.
            if use_WRR and cfg.use_rr_recurrence:
                rr_drive = self.W_RR @ X_R
            else:
                rr_drive = zero_R
            X_R_new = np.clip(rr_drive + inp_R, 0.0, 1.0)

            X_E, X_R, X_Z = X_E_new, X_R_new, X_Z_new

            if recorder is not None:
                recorder.setdefault("erz_steps", []).append(
                    {
                        "step": step,
                        "X_E": X_E.copy(),
                        "X_R": X_R.copy(),
                        "X_Z": X_Z.copy(),
                    }
                )

        return {
            entity: float(X_E[entity])
            for entity in range(self.N)
            if entity != h and X_E[entity] > 0.0
        }

    def _query_single(
        self,
        h: int,
        r: int,
        *,
        T: Optional[int] = None,
        use_WRR: bool = True,
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        steps = int(self.cfg.T_infer if T is None else T)
        if recorder is not None:
            return self._run_erz(
                int(h), int(r), T=steps, use_WRR=bool(use_WRR), recorder=recorder
            )
        key = (int(h), int(r), steps, bool(use_WRR), float(self.cfg.T_thresh))
        if self.cfg.enable_erz_cache:
            cached = self._erz_cache.get(key)
            if cached is not None:
                return cached
        scores = self._run_erz(int(h), int(r), T=steps, use_WRR=bool(use_WRR))
        if self.cfg.enable_erz_cache:
            self._erz_cache[key] = scores
        return scores

    # ------------------------------------------------------------------
    # ERZ-driven C learning
    # ------------------------------------------------------------------
    def _active_erz_targets(self, h: int, r: int, T: int) -> Tuple[int, ...]:
        """Threshold the output of an ERZ episode with W_RR strictly disabled."""
        threshold = float(self.cfg.C_activity_threshold)
        scores = self._query_single(h, r, T=T, use_WRR=False)
        return tuple(sorted(int(t) for t, score in scores.items() if float(score) >= threshold))

    def _reverse_consequent_activity(self, h: int, t: int) -> np.ndarray:
        """Read consequent relation activity by a simultaneous inverse-R return hop.

        The third learning hop follows the requested neural sequence:
        1. clamp entity t in E and activate all M inverse-relation neurons in R;
        2. form the E/R conjunction in the relation-indexed Z cells of t;
        3. run one recurrent Z -> Z iteration;
        4. retain only the Z population belonging to h and read it back to R.

        No adjacency lookup and no per-r_k direct ERZ query is used. The returned
        vector has K=2M entries because the C-agent population has size 2M. With
        the requested inverse-neuron stimulation, the readout normally occupies
        the forward-relation half of that vector.
        """
        h, t = int(h), int(t)
        if not (0 <= h < self.N and 0 <= t < self.N):
            raise ValueError(f"Entity out of range in C return hop: h={h}, t={t}")

        cfg = self.cfg
        I_E = np.zeros(self.N, dtype=float)
        I_R = np.zeros(self.K, dtype=float)
        I_E[t] = 1.0
        I_R[self.M:self.K] = 1.0

        # E(t) and all inverse R neurons jointly select only the inverse-indexed
        # Z cells belonging to t. With eta_Z=1, an E-only or R-only input is
        # subthreshold, whereas their conjunction reaches activity 1.
        ez_drive = self.W_EZ @ I_E
        rz_drive = self.W_ZR.T @ sigma_gate(I_R, float(cfg.T_thresh))
        X_Z_tail = gated_relu(ez_drive + rz_drive, float(cfg.eta_Z))

        # One recurrent Z-layer iteration carries each active (t, r_k^-1) cell
        # through the encoded Z-Z synapse to the corresponding (h', r_k) cell.
        zz_base = self.W_ZZ @ X_Z_tail
        zz_drive = zz_base + float(cfg.b_bias) * (zz_base > 0.0).astype(float)
        X_Z_return = gated_relu(zz_drive, float(cfg.eta_Z))

        # Read only relations whose returned Z activity lands on the selected h.
        X_Z_at_h = np.zeros(self.NZ, dtype=float)
        h_start = h * self.K
        h_stop = h_start + self.K
        X_Z_at_h[h_start:h_stop] = X_Z_return[h_start:h_stop]
        X_R_read = np.clip(self.W_ZR @ X_Z_at_h, 0.0, 1.0)

        threshold = float(cfg.C_activity_threshold)
        return sigma_gate(np.asarray(X_R_read).reshape(self.K), threshold).astype(np.float32)

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

        w = self.W_comp_agent[:, r_i, r_j]
        self.W_comp_agent[:, r_i, r_j] = np.clip(w + eta * (y - w), 0.0, 1.0)

    def learn_C_layer(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicERZCBase":
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
        """
        steps = int(self.cfg.C_learning_T if T is None else T)
        if steps <= 0:
            raise ValueError("C learning T must be positive")

        if reset:
            self.W_comp_agent.fill(0.0)
            self.C_support.fill(0)
            self.C_hit_count.fill(0)

        # Kept for API compatibility. The requested learning sequence is an
        # explicit deterministic set of nested for-loops, so it is not shuffled.
        _ = seed
        n_dynamic_calls_before = len(self._erz_cache)
        n_passes = max(1, int(self.cfg.C_n_passes))
        if self.cfg.C_hebbian_mode == "running_mean":
            n_passes = 1

        seen_by_pair = np.zeros((self.K, self.K), dtype=np.int64)
        total_path_samples = 0

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
            for r_i in range(self.K):
                for h in range(self.N):
                    intermediates = self._active_erz_targets(h, r_i, steps)
                    for x in intermediates:
                        for r_j in range(self.K):
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
                                    self.C_support[r_i, r_j] += 1
                                    self.C_hit_count[:, r_i, r_j] += (
                                        y >= float(self.cfg.C_activity_threshold)
                                    ).astype(np.int32)
                                    total_path_samples += 1

                                seen_by_pair[r_i, r_j] += 1
                                self._c_hebbian_update(
                                    r_i,
                                    r_j,
                                    y,
                                    int(seen_by_pair[r_i, r_j]),
                                )

        active_antecedent_pairs = int(np.count_nonzero(self.C_support > 0))
        active_rules = int(
            np.count_nonzero(
                (self.W_comp_agent >= float(self.cfg.C_rule_threshold))
                & (self.C_support[None, :, :] >= int(self.cfg.C_min_support))
            )
        )
        self.C_learning_stats = {
            "source": "ERZ_first_second_hops_plus_simultaneous_inverse_R_return",
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
            positive = self.W_comp_agent[self.W_comp_agent > 0.0]
            print(
                "[C/ERZ] "
                f"pairs={active_antecedent_pairs}/{self.K*self.K} "
                f"samples={total_path_samples} active_rules={active_rules} "
                f"mean_w={float(positive.mean()) if len(positive) else 0.0:.4f} "
                f"max_w={float(positive.max()) if len(positive) else 0.0:.4f}"
            )
        return self

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
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        scores_direct = self._query_single(h, r_k, T=T, use_WRR=use_WRR, recorder=recorder)

        # One-hot consequent C-agent activity projected through learned synapses.
        X_C_agent = np.zeros(self.K, dtype=np.float32)
        X_C_agent[r_k] = 1.0
        X_C_comp = np.tensordot(X_C_agent, self.W_comp_agent, axes=([0], [0]))

        active_rules = np.argwhere(
            (X_C_comp >= float(self.cfg.C_rule_threshold))
            & (self.C_support >= int(self.cfg.C_min_support))
        )

        # Optionally keep only the highest-confidence rules.
        if top_rules_fraction is not None and 0.0 < top_rules_fraction < 1.0:
            if len(active_rules) > 0:
                confidences = np.array([float(X_C_comp[r_i, r_j]) for r_i, r_j in active_rules])
                keep_count = max(1, int(np.ceil(len(active_rules) * top_rules_fraction)))
                top_indices = np.argsort(confidences)[-keep_count:]
                active_rules = active_rules[top_indices]

        scores_comp: Dict[int, float] = {}
        for raw_i, raw_j in active_rules:
            r_i, r_j = int(raw_i), int(raw_j)
            confidence = float(X_C_comp[r_i, r_j])

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
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        steps = int(self.cfg.T_infer if T is None else T)
        use_c = self.cfg.enable_C_layer_by_default if use_C_layer is None else bool(use_C_layer)
        alpha_value = float(self.cfg.alpha if alpha is None else alpha)
        if use_c:
            return self._query_with_C(
                int(h), int(r), T=steps, use_WRR=bool(use_WRR),
                agg_mode=agg_mode, alpha=alpha_value,
                top_rules_fraction=top_rules_fraction,
                recorder=recorder,
            )
        return self._query_single(int(h), int(r), T=steps, use_WRR=bool(use_WRR), recorder=recorder)

    # ------------------------------------------------------------------
    # Model persistence (avoids re-learning across reasoning experiments)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save learned weights and encoded triples to an .npz file."""
        self._ensure_sparse()
        np.savez(
            path,
            N=self.N,
            M=self.M,
            W_RR=self.W_RR,
            W_comp_agent=self.W_comp_agent,
            C_support=self.C_support,
            C_hit_count=self.C_hit_count,
            zr_keys=np.array(list(self._zr_entries.keys()), dtype=np.int64),
            zr_vals=np.array(list(self._zr_entries.values()), dtype=float),
            zz_keys=np.array(list(self._zz_entries.keys()), dtype=np.int64),
            zz_vals=np.array(list(self._zz_entries.values()), dtype=float),
            encoded_triples=np.array(self._encoded_triples, dtype=np.int64),
            cfg_dict=np.array(str(self.cfg.to_dict()), dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "NSRDynamicERZCBase":
        """Load a model previously saved with ``save``."""
        data = np.load(path, allow_pickle=True)
        N = int(data["N"])
        M = int(data["M"])
        cfg = NSRConfig.from_dict(eval(str(data["cfg_dict"])))
        model = cls(N, M, cfg)

        model.W_RR = data["W_RR"]
        model.W_comp_agent = data["W_comp_agent"]
        model.C_support = data["C_support"]
        model.C_hit_count = data["C_hit_count"]

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

        for r_k in range(self.K):
            active = np.argwhere(
                (self.W_comp_agent[r_k] >= threshold_value)
                & (self.C_support >= support_value)
            )
            for raw_i, raw_j in active:
                r_i, r_j = int(raw_i), int(raw_j)
                rules.append(
                    {
                        "rule": (
                            f"{self._relation_name(r_i, id2rel)} + "
                            f"{self._relation_name(r_j, id2rel)} => "
                            f"{self._relation_name(r_k, id2rel)}"
                        ),
                        "r_i": r_i,
                        "r_j": r_j,
                        "r_k": r_k,
                        "confidence": float(self.W_comp_agent[r_k, r_i, r_j]),
                        "support": int(self.C_support[r_i, r_j]),
                        "hits": int(self.C_hit_count[r_k, r_i, r_j]),
                    }
                )

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


"""Faithful C-layer dynamics for NSR Dynamic ERZ.

This module subclasses NSR_Fake_C_Dynamic_V2 and replaces its
"fake" C-layer statistics gathering with a genuine neural dynamics that
maintains X_C_agent and X_C_comp states across tau steps.

Learning dynamics (per sample path h --r_i--> x --r_j--> t):

    tau=1 : I_R = r_i, I_C_agent = c_i, I_C_comp = 0.
            X_C_agent gets c_i; X_C_comp stays silent.
    tau=2 : I_R = r_j, I_C_agent = c_j, I_C_comp still 0.
            W_agent_agent_fix keeps c_i; W_CR adds c_j.
            Both c_i and c_j project to c_{ij}, crossing eta_Comp.
    tau=3 : I_C_agent = 0 (silence agent), I_C_comp = c_{ij} (hold it).
            All inverse R neurons fire; one Z->Z step reads forward {r_k}
            at h via the reverse readout mechanism already present in V2.
    tau=4 : Forward {r_k} in R maps to {c_k} in C_agent.
            W_comp_agent[k, i, j] is updated by delta-rule toward X_C_agent.

Fixed matrices:

    W_comp_agent_fix : (K*K, K). For comp neuron (i, j), it receives
                       unit input from agent i and agent j.
    W_comp_comp_fix  : identity, keeps an active c_{ij} alive.
    W_agent_agent_fix: identity, keeps an active c_i / c_j alive.
    W_CR             : identity, maps R neuron r to C_agent neuron c_r.

The only learned weights are W_comp_agent (K, K, K).
"""




class NSRDynamicERZC(NSRDynamicERZCBase):
    """NSR Dynamic whose C-layer is a real multi-step neural dynamics."""

    def __init__(self, num_entities: int, num_relations: int, cfg=None):
        super().__init__(num_entities, num_relations, cfg)
        (
            self.W_comp_agent_fix,
            self.W_comp_comp_fix,
            self.W_agent_agent_fix,
            self.W_CR,
        ) = self._build_c_fixed_weights()
        (
            self.W_agent_comp,
            self.W_agent_agent,
            self.eta_C_agent,
            self.eta_thresh,
            self.k_sigmoid,
        ) = self._build_c_reasoning_weights()

    # ------------------------------------------------------------------
    # C-layer faithful reasoning (WTA dynamics with user-provided formulas)
    # ------------------------------------------------------------------
    def _build_c_reasoning_weights(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Return fixed weights and thresholds used during faithful WTA reasoning.

        W_agent_comp is Naturally Encoded: c_{ij} -> c_i is strong, c_{ij} -> c_j
        is weak.  W_agent_agent is a diffusion matrix: it spreads an active
        agent's state to all *other* agents (diagonal = 0), but the diffusion
        alone is below threshold and cannot activate any agent without the
        comp/R inputs.
        """
        K = self.K
        K2 = K * K

        w_strong = 1.0    # c_{ij} -> c_i
        w_weak = 0.6      # c_{ij} -> c_j
        w_diffuse = 0.5   # off-diagonal agent-agent diffusion
        eta_C = 0.55      # agent input threshold
        eta_thresh = 0.5  # Heaviside threshold for comp projection
        k_sigmoid = 5.0   # steepness of sigma_c_comp

        W_agent_comp = np.zeros((K, K2), dtype=float)
        for i in range(K):
            for j in range(K):
                comp_idx = i * K + j
                W_agent_comp[i, comp_idx] = w_strong
                W_agent_comp[j, comp_idx] = w_weak

        # Diffusion matrix: off-diagonal = w_diffuse, diagonal = 0.
        W_agent_agent = np.full((K, K), w_diffuse, dtype=float)
        np.fill_diagonal(W_agent_agent, 0.0)

        return W_agent_comp, W_agent_agent, eta_C, eta_thresh, k_sigmoid

    @staticmethod
    def _sigma_c_comp(x: np.ndarray, k: float) -> np.ndarray:
        """Signed sigmoid mapping rule confidence to [0, 1)."""
        return 2.0 / (1.0 + np.exp(-k * x)) - 1.0

    @staticmethod
    def _wta(x: np.ndarray) -> np.ndarray:
        """Hard Winner-Take-All: max entry becomes 1, others 0."""
        y = np.zeros_like(x)
        max_val = float(x.max())
        if max_val > 0.0:
            y[int(x.argmax())] = 1.0
        return y

    def _c_reasoning_dynamic(
        self,
        X_C_agent: np.ndarray,
        X_C_comp: np.ndarray,
        X_R: np.ndarray,
        I_C_agent: np.ndarray,
        *,
        eta_thresh: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One faithful C-layer step using the user's WTA equations.

        Equations:
            X_C_agent(τ+1) = WTA{
                W_agent_agent X_C_agent(τ)
                + W_agent_comp X_C_comp(τ)
                + W_CR X_R(τ)
                - η_C
            } + I_C_agent

            X_C_comp(τ+1) = W_C_compC_comp X_C_comp(τ)
                + σ_c_comp(W_comp_agent @ H(X_C_agent(τ+1) - η_thresh))

        W_comp_agent is the learned rule confidence tensor; it is used as the
        agent -> composition projection during reasoning.

        The external I_C_comp input is omitted because the user specifies it
        should always be 0; c_{ij} persistence is handled by the self-loop
        W_C_compC_comp.
        """
        if eta_thresh is None:
            eta_thresh = self.eta_thresh

        # Agent update: WTA on the combined input.
        agent_input = (
            self.W_agent_agent @ X_C_agent
            + self.W_agent_comp @ X_C_comp
            + self.W_CR @ X_R
            - self.eta_C_agent
        )
        X_C_agent_new = self._wta(agent_input)
        X_C_agent_new = np.clip(X_C_agent_new + I_C_agent, 0.0, 1.0)

        # Composition update: project active agent through learned W_comp_agent.
        active = (X_C_agent_new > eta_thresh).astype(float)
        proj = np.tensordot(active, self.W_comp_agent, axes=([0], [0]))
        comp_drive = self._sigma_c_comp(proj.ravel(), self.k_sigmoid)
        X_C_comp_new = np.clip(X_C_comp + comp_drive, 0.0, 1.0)

        return X_C_agent_new, X_C_comp_new

    def _query_with_C_faithful(
        self,
        h: int,
        r_k: int,
        *,
        T: int,
        use_WRR: bool,
        agg_mode: str,
        alpha: float,
        top_rules_fraction: float = 1.0,
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        """Faithful WTA-based C-layer reasoning for a query (h, r_k, ?).

        The C-layer state is updated in a single ``for tau in range(...)`` loop
        per active rule, exactly like ``learn_C_layer`` does during learning.
        ERZ readouts are inserted between C-layer steps (τ=2→3 and τ=4→5).
        """
        scores_direct = self._query_single(h, r_k, T=T, use_WRR=use_WRR, recorder=recorder)

        K = self.K
        K2 = K * K
        k_sigmoid = self.k_sigmoid

        zero_comp = np.zeros(K2, dtype=float)
        zero_agent = np.zeros(K, dtype=float)

        # τ=1: query relation activates consequent agent c_k.
        X_R_t1 = np.zeros(K, dtype=float)
        X_R_t1[r_k] = 1.0
        X_C_agent_t1, X_C_comp_all = self._c_reasoning_dynamic(
            zero_agent, zero_comp, X_R_t1, zero_agent
        )

        # Active rules are selected from the learned weights, exactly like V2.
        # The C-composition activity X_C_comp_all is used only to drive the
        # dynamics; selection uses raw rule confidence to avoid float-ordering
        # differences between V2 (float32) and Faithful (float64).
        active_rules = np.argwhere(
            (self.W_comp_agent[r_k, :, :] >= float(self.cfg.C_rule_threshold))
            & (self.C_support >= int(self.cfg.C_min_support))
        )

        # Optionally keep only the highest-confidence rules.
        if top_rules_fraction is not None and 0.0 < top_rules_fraction < 1.0:
            if len(active_rules) > 0:
                confidences = np.array(
                    [float(self.W_comp_agent[r_k, r_i, r_j]) for r_i, r_j in active_rules]
                )
                keep_count = max(1, int(np.ceil(len(active_rules) * top_rules_fraction)))
                top_indices = np.argsort(confidences)[-keep_count:]
                active_rules = active_rules[top_indices]

        scores_comp: Dict[int, float] = {}

        # For τ>1 we raise the Heaviside threshold so that a single active agent
        # (c_i or c_j) can no longer activate new c_{ij} neurons; only the
        # self-loop keeps the originally activated c_{ij} alive.
        eta_thresh_tau_gt1 = 1.1

        for raw_i, raw_j in active_rules:
            r_i, r_j = int(raw_i), int(raw_j)
            comp_idx = r_i * K + r_j

            # Raw confidence (used for final scoring, must match V2).
            confidence_raw = float(self.W_comp_agent[r_k, r_i, r_j])
            if confidence_raw <= 0.0:
                continue

            # C-comp activity for this rule comes from the actual τ=1 dynamics.
            confidence_comp = float(X_C_comp_all[comp_idx])

            # Isolated C-comp memory for this rule.
            X_C_comp_rule = zero_comp.copy()
            X_C_comp_rule[comp_idx] = confidence_comp

            # Two-step C-layer episode for this rule (τ=2 and τ=3), with ERZ
            # readouts inserted between the steps.  The loop body is the same
            # ``_c_reasoning_dynamic`` call at every τ, just like learning.
            X_R_series = np.zeros((2, K), dtype=float)
            I_C_agent_series = np.zeros((2, K), dtype=float)

            X_C_agent = X_C_agent_t1.copy()
            X_C_comp = X_C_comp_rule.copy()

            # Collect agent states after τ=2 and τ=3.
            agent_states = []
            for tau in range(2):
                X_C_agent, X_C_comp = self._c_reasoning_dynamic(
                    X_C_agent,
                    X_C_comp,
                    X_R_series[tau],
                    I_C_agent_series[tau],
                    eta_thresh=eta_thresh_tau_gt1,
                )
                agent_states.append(X_C_agent.copy())

            # Fallback: when r_i == r_k the WTA diffusion matrix has no self-loop,
            # so c_i (which equals the previous winner c_k) cannot sustain itself.
            # In this case we directly set the τ=2 winner to c_i, and recompute
            # τ=3 from that corrected state; otherwise the erroneous τ=2 winner
            # (c_j) propagates into τ=3 and prevents c_j from winning there.
            X_C_agent_t2 = agent_states[0]
            if r_i == r_k:
                X_C_agent_t2 = zero_agent.copy()
                X_C_agent_t2[r_i] = 1.0
                X_C_agent_t3, _ = self._c_reasoning_dynamic(
                    X_C_agent_t2,
                    X_C_comp_rule,
                    X_R_series[1],
                    I_C_agent_series[1],
                    eta_thresh=eta_thresh_tau_gt1,
                )
            else:
                X_C_agent_t3 = agent_states[1]
            if recorder is not None:
                recorder.setdefault("c_rule_states", []).append(
                    {
                        "r_i": r_i,
                        "r_j": r_j,
                        "confidence": confidence_raw,
                        "X_C_agent_t2": X_C_agent_t2.copy(),
                        "X_C_agent_t3": X_C_agent_t3.copy(),
                    }
                )
            if X_C_agent_t2[r_i] <= 0.0:
                continue

            # First ERZ hop: (h, r_i, ?)
            for x, score_hx in self._query_single(h, r_i, T=T, use_WRR=False).items():
                if score_hx <= 0.0:
                    continue

                if X_C_agent_t3[r_j] <= 0.0:
                    continue

                # Second ERZ hop: (x, r_j, ?)
                for t, score_xt in self._query_single(x, r_j, T=T, use_WRR=False).items():
                    if t == h or score_xt <= 0.0:
                        continue
                    path_score = float(score_hx) * float(score_xt) * confidence_raw
                    scores_comp[int(t)] = max(scores_comp.get(int(t), 0.0), path_score)

        scores = dict(scores_direct)
        if agg_mode == "max":
            for t, score in scores_comp.items():
                scores[t] = max(scores.get(t, 0.0), score)
        elif agg_mode == "add":
            for t, score in scores_comp.items():
                scores[t] = scores.get(t, 0.0) + float(alpha) * score
        if recorder is not None:
            recorder["c_layer"] = {
                "X_C_agent_t1": X_C_agent_t1.copy(),
                "X_C_comp_t1": X_C_comp_all.copy(),
                "active_rules": [[int(i), int(j)] for i, j in active_rules],
                "scores_comp": {int(k): float(v) for k, v in scores_comp.items()},
            }
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
        recorder: Optional[Dict[str, object]] = None,
    ) -> Dict[int, float]:
        steps = int(self.cfg.T_infer if T is None else T)
        use_c = self.cfg.enable_C_layer_by_default if use_C_layer is None else bool(use_C_layer)
        alpha_value = float(self.cfg.alpha if alpha is None else alpha)
        if use_c:
            return self._query_with_C_faithful(
                int(h), int(r), T=steps, use_WRR=bool(use_WRR),
                agg_mode=agg_mode, alpha=alpha_value,
                top_rules_fraction=top_rules_fraction,
                recorder=recorder,
            )
        return self._query_single(int(h), int(r), T=steps, use_WRR=bool(use_WRR), recorder=recorder)

    # ------------------------------------------------------------------
    # C-layer faithful learning
    # ------------------------------------------------------------------
    def _build_c_fixed_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the fixed C-layer connectivity matrices."""
        K = self.K
        K2 = K * K

        # W_comp_agent_fix[(i, j), k]: comp (i, j) receives from agent i and agent j.
        W_comp_agent_fix = np.zeros((K2, K), dtype=float)
        for i in range(K):
            row_start = i * K
            for j in range(K):
                comp_idx = row_start + j
                W_comp_agent_fix[comp_idx, i] = 1.0
                W_comp_agent_fix[comp_idx, j] = 1.0

        # Self-loops keep active neurons alive.
        W_comp_comp_fix = np.eye(K2, dtype=float)
        W_agent_agent_fix = np.eye(K, dtype=float)

        # One-to-one mapping between R and C_agent.
        W_CR = np.eye(K, dtype=float)

        return W_comp_agent_fix, W_comp_comp_fix, W_agent_agent_fix, W_CR

    def _c_comp_threshold(self) -> float:
        """Return the C-composition threshold."""
        return float(getattr(self.cfg, "C_comp_threshold", 1.5))

    @staticmethod
    def _heaviside(x: np.ndarray, threshold: float) -> np.ndarray:
        return (x >= threshold).astype(float)

    @staticmethod
    def _gated_relu(x: np.ndarray, threshold: float) -> np.ndarray:
        return np.clip(np.maximum(0.0, x - threshold), 0.0, 1.0)

    def _c_learning_dynamic(
        self,
        X_C_agent: np.ndarray,
        X_C_comp: np.ndarray,
        X_R: np.ndarray,
        I_C_agent: np.ndarray,
        eta_comp: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One unified C-layer dynamics step.

        Applies the same equations at every tau, exactly like `_run_erz`
        applies the same ERZ equations at every time step.

        Equations
        ---------
        X_C_agent(t+1) = sigma_cagent(
            W_agent_agent_fix @ X_C_agent(t) + W_CR @ X_R(t) - eta_cagent
        ) + I_C_agent

        X_C_comp(t+1) = sigma_c(
            H(W_comp_agent_fix @ X_C_agent(t) - eta_comp)
            + W_comp_comp_fix @ X_C_comp(t)
        )

        Simplification due to fixed matrices (marked below)
        ----------------------------------------------------
        - W_agent_agent_fix is the identity and W_CR is the identity, and
          eta_cagent = 0, so the agent update reduces to:
              X_C_agent_new = clip(X_C_agent + X_R + I_C_agent, 0, 1)
        - W_comp_comp_fix is the identity, so the comp self-loop is just
          X_C_comp.
        - W_comp_agent_fix[(i,j), k] = 1 iff k == i or k == j (the same entry
          when i == j). Therefore, for i != j,
          (W_comp_agent_fix @ X_C_agent)[i,j] = X_C_agent[i] + X_C_agent[j].
          We compute this for every (i,j) pair with broadcasting and rely on
          eta_Comp in (1, 2) to avoid spurious i == i compositions.
        - I_C_comp is always 0, so it is omitted from the update.
        """
        # --- Agent update ---
        # Faithful matrix form: W_agent_agent_fix @ X_C_agent + W_CR @ X_R.
        # Simplification: both matrices are identity, eta_cagent = 0.
        # sigma_cagent with threshold 0 is clip(x, 0, 1), so:
        #   X_C_agent_new = clip(clip(X_C_agent + X_R, 0, 1) + I_C_agent, 0, 1)
        # Since all values are binary, this is equivalent to element-wise OR.
        # NOTE: negative I_C_agent is used at tau=3 to silence the agent layer.
        X_C_agent_new = np.clip(X_C_agent + X_R + I_C_agent, 0.0, 1.0)

        # --- Composition update ---
        # Faithful matrix form: H(W_comp_agent_fix @ X_C_agent - eta_comp).
        # Simplification: W_comp_agent_fix has unit weights from agent i and j
        # to comp neuron (i, j), so each comp neuron's drive is
        # X_C_agent[i] + X_C_agent[j] for i != j.
        # We use the NEW agent state because the C-layer evolves agent first,
        # then composition (see user notes: tau=2 needs both c_i and c_j).
        pair_sum = X_C_agent_new[:, None] + X_C_agent_new[None, :]  # (K, K)
        comp_activation = self._heaviside(pair_sum.ravel() - eta_comp, 0.0)

        # Faithful matrix form: comp_activation + W_comp_comp_fix @ X_C_comp.
        # Simplification: W_comp_comp_fix is the identity.
        X_C_comp_new = np.clip(comp_activation + X_C_comp, 0.0, 1.0)

        return X_C_agent_new, X_C_comp_new

    def learn_C_layer(
        self,
        *,
        T: Optional[int] = None,
        seed: int = 0,
        reset: bool = True,
        verbose: bool = False,
    ) -> "NSRDynamicFaithfulC":
        """Learn W_comp_agent with the faithful C-layer dynamics.

        For every concrete two-hop path h --r_i--> x --r_j--> t we run a
        four-step C-layer episode.  The same ``_c_step`` is called at each
        tau; only the external input I_C_agent and the R-layer input X_R change
        from step to step, analogous to ``_run_erz``.  I_C_comp is always 0.
        """
        steps = int(self.cfg.C_learning_T if T is None else T)
        if steps <= 0:
            raise ValueError("C learning T must be positive")

        if reset:
            self.W_comp_agent.fill(0.0)
            self.C_support.fill(0)
            self.C_hit_count.fill(0)

        eta_comp = self._c_comp_threshold()
        delta_C = float(self.cfg.delta_C)

        # V2 ignores seed in its deterministic loops; kept for API parity.
        _ = seed

        n_dynamic_calls_before = len(self._erz_cache)
        n_passes = max(1, int(self.cfg.C_n_passes))
        if self.cfg.C_hebbian_mode == "running_mean":
            n_passes = 1

        seen_by_pair = np.zeros((self.K, self.K), dtype=np.int64)
        total_path_samples = 0

        zero_agent = np.zeros(self.K, dtype=float)
        zero_comp = np.zeros(self.K * self.K, dtype=float)

        # Cache reverse readouts when requested (same speedup as V2).
        reverse_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = (
            {} if getattr(self.cfg, "learn_cache_reverse", False) else None
        )

        for pass_index in range(n_passes):
            count_statistics = pass_index == 0

            for r_i in range(self.K):
                for h in range(self.N):
                    intermediates = self._active_erz_targets(h, r_i, steps)
                    for x in intermediates:
                        for r_j in range(self.K):
                            tails = self._active_erz_targets(x, r_j, steps)
                            for t in tails:
                                if t == h:
                                    continue

                                # Pre-compute the reverse readout that will be
                                # injected into R (and thus C_agent) at tau=4.
                                if reverse_cache is not None:
                                    key = (h, t)
                                    y = reverse_cache.get(key)
                                    if y is None:
                                        y = self._reverse_consequent_activity(h, t)
                                        reverse_cache[key] = y
                                else:
                                    y = self._reverse_consequent_activity(h, t)

                                # Build the external input time-series for the
                                # four C-layer steps tau = 1, 2, 3, 4.
                                # All arrays have shape (4, ...) so the loop body
                                # is just _c_step(...); no per-tau branching.
                                X_R_series = np.zeros((4, self.K), dtype=float)
                                I_C_agent_series = np.zeros((4, self.K), dtype=float)

                                # tau=1: external r_i activates c_i.
                                X_R_series[0, r_i] = 1.0
                                I_C_agent_series[0, r_i] = 1.0

                                # tau=2: external r_j activates c_j.
                                # c_i persists via W_agent_agent_fix identity self-loop.
                                X_R_series[1, r_j] = 1.0
                                I_C_agent_series[1, r_j] = 1.0

                                # tau=3: silence C_agent.  c_{ij} is held by the
                                # W_comp_comp_fix identity self-loop (eta_Comp>1).
                                I_C_agent_series[2, :] = -1.0

                                # tau=4: forward consequent relations {r_k} in R
                                # map one-to-one to {c_k} in C_agent; c_{ij} held.
                                X_R_series[3, :] = y

                                # Four-tau C-layer episode for this path.
                                # The same _c_learning_dynamic is invoked at every tau.
                                X_C_agent = np.zeros(self.K, dtype=float)
                                X_C_comp = np.zeros(self.K * self.K, dtype=float)
                                for tau in range(4):
                                    X_C_agent, X_C_comp = self._c_learning_dynamic(
                                        X_C_agent,
                                        X_C_comp,
                                        X_R_series[tau],
                                        I_C_agent_series[tau],
                                        eta_comp,
                                    )

                                # Delta / EMA update of the learned weights.
                                seen_by_pair[r_i, r_j] += 1
                                w = self.W_comp_agent[:, r_i, r_j]
                                self.W_comp_agent[:, r_i, r_j] = np.clip(
                                    w + delta_C * (X_C_agent - w),
                                    0.0,
                                    1.0,
                                )

                                if count_statistics:
                                    self.C_support[r_i, r_j] += 1
                                    self.C_hit_count[:, r_i, r_j] += (
                                        X_C_agent >= float(self.cfg.C_activity_threshold)
                                    ).astype(np.int32)
                                    total_path_samples += 1

        active_antecedent_pairs = int(np.count_nonzero(self.C_support > 0))
        active_rules = int(
            np.count_nonzero(
                (self.W_comp_agent >= float(self.cfg.C_rule_threshold))
                & (self.C_support[None, :, :] >= int(self.cfg.C_min_support))
            )
        )
        self.C_learning_stats = {
            "source": "faithful_C_layer_dynamics",
            "T": steps,
            "C_comp_threshold": eta_comp,
            "C_activity_threshold": float(self.cfg.C_activity_threshold),
            "antecedent_pairs_with_support": active_antecedent_pairs,
            "path_experience_samples": total_path_samples,
            "unique_endpoint_samples": total_path_samples,
            "active_rules": active_rules,
            "erz_cache_entries_added": max(0, len(self._erz_cache) - n_dynamic_calls_before),
        }

        if verbose:
            positive = self.W_comp_agent[self.W_comp_agent > 0.0]
            print(
                "[C/faithful] "
                f"pairs={active_antecedent_pairs}/{self.K*self.K} "
                f"samples={total_path_samples} active_rules={active_rules} "
                f"mean_w={float(positive.mean()) if len(positive) else 0.0:.4f} "
                f"max_w={float(positive.max()) if len(positive) else 0.0:.4f}"
            )
        return self


