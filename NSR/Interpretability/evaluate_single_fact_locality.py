#!/usr/bin/env python3
"""Single-fact learning-update locality for NSR and NeuralLP.

The public entry point runs under the current Python environment and launches a
small TensorFlow-1 worker in the existing ``nsr_tf1`` Conda environment for
NeuralLP.  The experiment is deliberately restricted to two-hop composition
rules, which are shared by NSR's C layer and NeuralLP's decoded rule space.

For every selected validation fact e=(h, r, t):

1. The base model is trained on the official training graph only.
2. The fact must close at least one existing two-hop path h -> x -> t.
3. NSR rule scores are recomputed after G <- G union {e}; this is the exact
   count/running-mean implementation of the C-layer rule weights.
4. NeuralLP is restored to the same released checkpoint, receives one native
   Adam update on e and its inverse, and its complete length-two rule-score
   tensor is decoded before and after the update.
5. Rule-update locality and update participation ratio are computed.

Run:
    python NSR/Interpretability/evaluate_single_fact_locality.py \
        --datasets NATIONS KINSHIP --num-events 30
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
INTERPRETABILITY_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OTHER_MODEL_ROOT = PROJECT_ROOT / "Other_Model"
COMMON_DATA_ROOT = OTHER_MODEL_ROOT / "data"
NEURALLP_ROOT = OTHER_MODEL_ROOT / "baselines" / "neurallp"
NEURALLP_RESULTS_ROOT = OTHER_MODEL_ROOT / "results"
RAW_RESULTS_PATH = INTERPRETABILITY_DIR / "single_fact_locality_results.json"
SUPPORTED_DATASETS = ("NATIONS", "KINSHIP")


def load_named_triples(path):
    triples = []
    with open(str(path), "r", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) == 3:
                triples.append(tuple(fields))
    return triples


def load_vocabulary(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_dataset(dataset):
    """Load the common split using NeuralLP's deterministic symbol ordering."""
    dataset = dataset.upper()
    neural_dir = NEURALLP_ROOT / "datasets" / ("benchmark_" + dataset.lower())
    relation_names = load_vocabulary(neural_dir / "relations.txt")
    entity_names = load_vocabulary(neural_dir / "entities.txt")
    relation_to_id = {name: idx for idx, name in enumerate(relation_names)}
    entity_to_id = {name: idx for idx, name in enumerate(entity_names)}

    def encode(rows):
        return [
            (entity_to_id[h], relation_to_id[r], entity_to_id[t])
            for h, r, t in rows
        ]

    common_dir = COMMON_DATA_ROOT / dataset
    return {
        "dataset": dataset,
        "relation_names": relation_names,
        "entity_names": entity_names,
        "train": encode(load_named_triples(common_dir / "train.txt")),
        "valid": encode(load_named_triples(common_dir / "valid.txt")),
    }


def directed_graph(triples, num_entities, num_relations):
    """Return deduplicated extended-relation adjacency and forward endpoints."""
    outgoing = [set() for _ in range(num_entities)]
    forward_relations = defaultdict(set)
    seen = set()
    for raw_h, raw_r, raw_t in triples:
        h, r, t = int(raw_h), int(raw_r), int(raw_t)
        if (h, r, t) in seen:
            continue
        seen.add((h, r, t))
        outgoing[h].add((r, t))
        outgoing[t].add((r + num_relations, h))
        forward_relations[(h, t)].add(r)
    return outgoing, forward_relations


def count_rule_statistics(triples, num_entities, num_relations):
    """Count NSR two-hop support and consequent hits.

    The returned rule bank has shape [M, 2M, 2M].  Consequents are restricted
    to the forward M relations because this is the part of NSR's current
    count/running-mean learner that is updated, and it aligns with NeuralLP's
    tail-query rule bank after canonicalizing its inverse-query convention.
    """
    k_rel = 2 * num_relations
    support = np.zeros((k_rel, k_rel), dtype=np.int64)
    hits = np.zeros((num_relations, k_rel, k_rel), dtype=np.int64)
    outgoing, forward_relations = directed_graph(
        triples, num_entities, num_relations
    )

    for h in range(num_entities):
        for r_i, x in outgoing[h]:
            for r_j, t in outgoing[x]:
                if t == h:
                    continue
                support[r_i, r_j] += 1
                for r_k in forward_relations.get((h, t), ()):
                    hits[r_k, r_i, r_j] += 1

    scores = np.zeros(hits.shape, dtype=np.float64)
    positive = support > 0
    if np.any(positive):
        scores[:, positive] = (
            hits[:, positive].astype(np.float64)
            / support[positive].astype(np.float64)[None, :]
        )
    return support, hits, scores


def count_closing_paths(event, triples, num_entities, num_relations):
    """Count base-graph two-hop paths from the event head to its tail."""
    h_event, _, t_event = event
    outgoing, _ = directed_graph(triples, num_entities, num_relations)
    count = 0
    for _, x in outgoing[h_event]:
        for _, t in outgoing[x]:
            if t == t_event and t != h_event:
                count += 1
    return count


def relevance_mask(base_support, base_hits, after_support, after_hits):
    """Graph-theoretic rule evidence changed by the newly observed fact."""
    support_changed = after_support != base_support
    hits_changed = after_hits != base_hits
    return hits_changed | support_changed[None, :, :]


def rule_update_locality(delta_scores, relevant):
    """Fraction of squared rule-score change on graph-relevant rules."""
    squared = np.square(np.asarray(delta_scores, dtype=np.float64))
    total = float(squared.sum())
    if total <= 0.0:
        return float("nan"), 0.0
    return float(squared[relevant].sum() / total), total


def update_participation(delta_parameters):
    """Effective updated parameter count based on inverse participation ratio."""
    delta = np.asarray(delta_parameters, dtype=np.float64).ravel()
    squared = np.square(delta)
    total = float(squared.sum())
    parameter_count = int(delta.size)
    if total <= 0.0:
        return {
            "parameter_count": parameter_count,
            "effective_updated_parameters": 0.0,
            "effective_update_ratio": 0.0,
            "changed_parameter_count": 0,
        }
    probabilities = squared / total
    effective = float(1.0 / np.square(probabilities).sum())
    relative_threshold = float(np.max(np.abs(delta))) * 1e-12
    changed = int(np.count_nonzero(np.abs(delta) > relative_threshold))
    return {
        "parameter_count": parameter_count,
        "effective_updated_parameters": effective,
        "effective_update_ratio": effective / float(parameter_count),
        "changed_parameter_count": changed,
    }


def relation_label(relation, names):
    relation = int(relation)
    m_rel = len(names)
    if relation < m_rel:
        return names[relation]
    return "inv_" + names[relation - m_rel]


def rule_label(index, names):
    r_k, r_i, r_j = (int(x) for x in index)
    return "{} <= {} o {}".format(
        names[r_k],
        relation_label(r_i, names),
        relation_label(r_j, names),
    )


def top_rule_changes(before, after, relevant, names, limit=5):
    delta = np.asarray(after) - np.asarray(before)
    flat_order = np.argsort(np.abs(delta).ravel())[::-1]
    changes = []
    for flat_idx in flat_order:
        idx = tuple(int(x) for x in np.unravel_index(flat_idx, delta.shape))
        value = float(delta[idx])
        if abs(value) <= 0.0:
            break
        changes.append({
            "rule": rule_label(idx, names),
            "before": float(before[idx]),
            "after": float(after[idx]),
            "delta": value,
            "graph_relevant": bool(relevant[idx]),
        })
        if len(changes) >= int(limit):
            break
    return changes


def aggregate(values):
    array = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def prepare_nsr_events(dataset_data, num_events, seed):
    train = list(dataset_data["train"])
    valid = list(dataset_data["valid"])
    names = dataset_data["relation_names"]
    entities = dataset_data["entity_names"]
    m_rel = len(names)
    n_ent = len(entities)

    base_support, base_hits, base_scores = count_rule_statistics(
        train, n_ent, m_rel
    )
    candidates = list(valid)
    random.Random(int(seed)).shuffle(candidates)
    selected = []

    for event in candidates:
        if event in train:
            continue
        closing_paths = count_closing_paths(event, train, n_ent, m_rel)
        if closing_paths <= 0:
            continue
        after_support, after_hits, after_scores = count_rule_statistics(
            train + [event], n_ent, m_rel
        )
        relevant = relevance_mask(
            base_support, base_hits, after_support, after_hits
        )
        delta = after_scores - base_scores
        locality, energy = rule_update_locality(delta, relevant)
        if energy <= 0.0 or not math.isfinite(locality):
            continue
        participation = update_participation(delta)
        h, r, t = event
        selected.append({
            "event_index": len(selected),
            "h": int(h),
            "r": int(r),
            "t": int(t),
            "fact": [entities[h], names[r], entities[t]],
            "closing_path_count": int(closing_paths),
            "relevant_rule_count": int(np.count_nonzero(relevant)),
            "nsr": {
                "rule_update_locality": locality,
                "rule_update_energy": energy,
                **participation,
                "top_rule_changes": top_rule_changes(
                    base_scores, after_scores, relevant, names
                ),
            },
        })
        if len(selected) >= int(num_events):
            break

    if len(selected) < int(num_events):
        raise RuntimeError(
            "{} has only {} eligible validation facts; requested {}".format(
                dataset_data["dataset"], len(selected), num_events
            )
        )
    return selected


def summarize_method(events, method):
    records = [event[method] for event in events]
    return {
        "rule_update_locality": aggregate(
            [record["rule_update_locality"] for record in records]
        ),
        "effective_updated_parameters": aggregate(
            [record["effective_updated_parameters"] for record in records]
        ),
        "effective_update_ratio": aggregate(
            [record["effective_update_ratio"] for record in records]
        ),
        "changed_parameter_count": aggregate(
            [float(record["changed_parameter_count"]) for record in records]
        ),
        "parameter_count": sorted({
            int(record["parameter_count"]) for record in records
        }),
    }


def parse_option_file(path):
    options = {}
    with open(str(path), "r", encoding="utf-8") as handle:
        for raw in handle:
            if ", " not in raw:
                continue
            key, value = raw.rstrip("\n").split(", ", 1)
            options[key] = value
    return options


def neural_rule_tensor(learner, session, num_relations):
    """Decode all NeuralLP length-two tail-query rules without pruning."""
    queries = np.arange(num_relations, 2 * num_relations, dtype=np.int32)
    attention_ops, attention_mems = learner.get_attentions_given_queries(
        session, queries
    )

    def operator_matrix(step):
        return np.concatenate(
            [np.asarray(column) for column in attention_ops[step]], axis=1
        ).astype(np.float64)

    op0 = operator_matrix(0)
    op1 = operator_matrix(1)
    scalar = (
        np.asarray(attention_mems[0])[:, 0]
        * np.asarray(attention_mems[1])[:, 1]
        * np.asarray(attention_mems[2])[:, 2]
    ).astype(np.float64)
    native = (
        scalar[:, None, None]
        * op0[:, :, None]
        * op1[:, None, :]
    )

    # NeuralLP matrices are oriented for head prediction.  Tail prediction uses
    # inverse query/operator IDs.  Swap the two relation banks to express the
    # decoded rules in NSR's forward/inverse traversal convention.
    order = np.concatenate([
        np.arange(num_relations, 2 * num_relations),
        np.arange(0, num_relations),
    ])
    return native[:, order, :][:, :, order]


def neural_worker(args):
    """TensorFlow-1 worker; invoked inside the existing nsr_tf1 environment."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    dataset_data = load_dataset(args.dataset)
    events = prepare_nsr_events(dataset_data, args.num_events, args.seed)
    dataset = args.dataset.upper()
    experiment_dir = (
        NEURALLP_RESULTS_ROOT / dataset / "neurallp" / "experiment"
    )
    saved_options = parse_option_file(experiment_dir / "option.txt")
    datadir = NEURALLP_ROOT / "datasets" / ("benchmark_" + dataset.lower())

    sys.path.insert(0, str(NEURALLP_ROOT / "src"))
    import tensorflow as tf  # pylint: disable=import-error,import-outside-toplevel
    from data import Data  # pylint: disable=import-error,import-outside-toplevel
    from model import Learner  # pylint: disable=import-error,import-outside-toplevel

    class Option(object):
        pass

    option = Option()
    option.seed = int(saved_options.get("seed", args.seed))
    option.num_step = int(saved_options.get("num_step", 3))
    option.num_layer = int(saved_options.get("num_layer", 1))
    option.rnn_state_size = int(saved_options.get("rnn_state_size", 128))
    option.query_embed_size = int(saved_options.get("query_embed_size", 128))
    option.learning_rate = float(saved_options.get("learning_rate", 0.001))
    option.no_norm = saved_options.get("no_norm", "False") == "True"
    option.thr = float(saved_options.get("thr", 1e-20))
    option.dropout = 0.0
    option.accuracy = False
    option.top_k = int(saved_options.get("top_k", 10))
    option.query_is_language = False
    option.type_check = False
    option.domain_size = int(saved_options.get("domain_size", 128))
    option.no_extra_facts = False

    data = Data(
        str(datadir),
        option.seed,
        option.type_check,
        option.domain_size,
        option.no_extra_facts,
    )
    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_query = data.num_query
    m_rel = data.num_relation
    n_ent = data.num_entity

    learner = Learner(option)
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(str(experiment_dir / "ckpt"))
    if checkpoint is None:
        raise FileNotFoundError(
            "No NeuralLP checkpoint under {}".format(experiment_dir / "ckpt")
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True

    trainable = tf.trainable_variables()
    base_support, base_hits, _ = count_rule_statistics(
        dataset_data["train"], n_ent, m_rel
    )
    output_events = []

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint)
        before_rules = neural_rule_tensor(learner, session, m_rel)

        for event_record in events:
            saver.restore(session, checkpoint)
            parameter_before = session.run(trainable)

            h = int(event_record["h"])
            r = int(event_record["r"])
            t = int(event_record["t"])
            # Match NeuralLP's native bidirectional training convention.
            learner.update(
                session,
                [r, r + m_rel],
                [h, t],
                [t, h],
                data.augmented_mdb_test,
            )

            parameter_after = session.run(trainable)
            delta_parameters = np.concatenate([
                (after - before).ravel()
                for before, after in zip(parameter_before, parameter_after)
            ])
            participation = update_participation(delta_parameters)
            after_rules = neural_rule_tensor(learner, session, m_rel)

            event = (h, r, t)
            after_support, after_hits, _ = count_rule_statistics(
                dataset_data["train"] + [event], n_ent, m_rel
            )
            relevant = relevance_mask(
                base_support, base_hits, after_support, after_hits
            )
            locality, energy = rule_update_locality(
                after_rules - before_rules, relevant
            )
            output_events.append({
                "event_index": int(event_record["event_index"]),
                "neural_lp": {
                    "rule_update_locality": locality,
                    "rule_update_energy": energy,
                    **participation,
                    "top_rule_changes": top_rule_changes(
                        before_rules,
                        after_rules,
                        relevant,
                        dataset_data["relation_names"],
                    ),
                },
            })

    payload = {
        "dataset": dataset,
        "checkpoint": checkpoint,
        "events": output_events,
    }
    with open(str(args.worker_output), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "dataset": dataset,
        "checkpoint": checkpoint,
        "num_events": len(output_events),
        "output": str(args.worker_output),
    }, indent=2))


def run_neural_worker(dataset, num_events, seed, output_path):
    command = [
        "conda", "run", "--no-capture-output", "-n", "nsr_tf1", "python",
        str(SCRIPT_PATH),
        "--neurallp-worker",
        "--dataset", dataset,
        "--num-events", str(num_events),
        "--seed", str(seed),
        "--worker-output", str(output_path),
    ]
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["CUDA_VISIBLE_DEVICES"] = ""
    print("[CMD] {}".format(" ".join(command)), flush=True)
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def merge_dataset_results(dataset, num_events, seed, worker_output):
    dataset_data = load_dataset(dataset)
    events = prepare_nsr_events(dataset_data, num_events, seed)
    with open(str(worker_output), "r", encoding="utf-8") as handle:
        neural_payload = json.load(handle)
    neural_by_idx = {
        int(record["event_index"]): record["neural_lp"]
        for record in neural_payload["events"]
    }
    for event in events:
        event["neural_lp"] = neural_by_idx[int(event["event_index"])]

    case = events[0]
    return {
        "dataset": dataset,
        "num_events": len(events),
        "event_selection": (
            "Seeded validation facts that are absent from training and close "
            "at least one existing two-hop path."
        ),
        "neural_lp_checkpoint": neural_payload["checkpoint"],
        "methods": {
            "NSR": summarize_method(events, "nsr"),
            "NeuralLP": summarize_method(events, "neural_lp"),
        },
        "case_event": {
            "fact": case["fact"],
            "closing_path_count": case["closing_path_count"],
            "relevant_rule_count": case["relevant_rule_count"],
            "NSR": case["nsr"]["top_rule_changes"],
            "NeuralLP": case["neural_lp"]["top_rule_changes"],
        },
        "events": events,
    }


def main(args):
    datasets = [dataset.upper() for dataset in args.datasets]
    unknown = sorted(set(datasets) - set(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError("Unsupported datasets: {}".format(", ".join(unknown)))

    worker_dir = INTERPRETABILITY_DIR / "_worker_results"
    worker_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "Single-Fact Learning Update Audit",
        "seed": int(args.seed),
        "num_events_per_dataset": int(args.num_events),
        "metrics": {
            "rule_update_locality": (
                "sum_{rho in R(e)} Delta c_rho^2 / "
                "sum_{rho} Delta c_rho^2"
            ),
            "effective_updated_parameters": (
                "N_eff = 1 / sum_i p_i^2, "
                "p_i = Delta theta_i^2 / ||Delta theta||_2^2"
            ),
            "effective_update_ratio": "N_eff / number of evaluated parameters",
        },
        "datasets": {},
    }

    for dataset in datasets:
        worker_output = worker_dir / (
            "neurallp_{}_events{}.json".format(dataset.lower(), args.num_events)
        )
        run_neural_worker(
            dataset, args.num_events, args.seed, worker_output
        )
        results["datasets"][dataset] = merge_dataset_results(
            dataset, args.num_events, args.seed, worker_output
        )

    with open(str(args.output), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print("\nSaved {}".format(args.output))
    for dataset in datasets:
        print("\n{}".format(dataset))
        for method, summary in results["datasets"][dataset]["methods"].items():
            print(
                "  {:8s} locality={:.6f} +/- {:.6f} "
                "N_eff/P={:.6e} +/- {:.6e}".format(
                    method,
                    summary["rule_update_locality"]["mean"],
                    summary["rule_update_locality"]["std"],
                    summary["effective_update_ratio"]["mean"],
                    summary["effective_update_ratio"]["std"],
                )
            )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=list(SUPPORTED_DATASETS)
    )
    parser.add_argument("--num-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RAW_RESULTS_PATH)
    parser.add_argument("--neurallp-worker", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.neurallp_worker:
        if parsed.dataset is None or parsed.worker_output is None:
            raise ValueError("--dataset and --worker-output are required")
        neural_worker(parsed)
    else:
        main(parsed)
