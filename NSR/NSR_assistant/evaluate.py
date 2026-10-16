"""Fixed-parameter train + evaluate for NSR models.

Supports NATIONS, KINSHIP, KINSHIP_1990 and WN18RR.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

Triple = Tuple[int, int, int]


def load_module_from_file(name: str, path: Path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mapped_triples_to_list(mapped) -> List[Triple]:
    array = mapped.detach().cpu().numpy().astype(np.int64)
    return [tuple(map(int, row)) for row in array.tolist()]


def pykeen_dataset_to_dict(dataset) -> Dict[str, Any]:
    training = dataset.training
    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id
    entities = [None] * len(entity_to_id)
    relations = [None] * len(relation_to_id)
    for label, index in entity_to_id.items():
        entities[int(index)] = str(label)
    for label, index in relation_to_id.items():
        relations[int(index)] = str(label)
    return {
        "entities": entities,
        "relations": relations,
        "train": mapped_triples_to_list(dataset.training.mapped_triples),
        "valid": mapped_triples_to_list(dataset.validation.mapped_triples),
        "test": mapped_triples_to_list(dataset.testing.mapped_triples),
    }


def load_nations() -> Dict[str, Any]:
    try:
        from NSR_Old_Models.nations_utils import load_nations_pykeen

        data = load_nations_pykeen()
        return {
            "entities": list(data["entities"]),
            "relations": list(data["relations"]),
            "train": [tuple(map(int, triple)) for triple in data["train"]],
            "valid": [tuple(map(int, triple)) for triple in data["val"]],
            "test": [tuple(map(int, triple)) for triple in data["test"]],
        }
    except Exception as fallback_error:
        try:
            from pykeen.datasets import Nations

            return pykeen_dataset_to_dict(Nations())
        except Exception as pykeen_error:
            raise RuntimeError(
                "Unable to load NATIONS from nations_utils or PyKEEN. "
                f"Fallback error: {fallback_error}; PyKEEN error: {pykeen_error}"
            ) from pykeen_error


def load_kinship() -> Dict[str, Any]:
    try:
        from pykeen.datasets import Kinships
    except Exception as exc:
        raise RuntimeError("KINSHIP requires PyKEEN: pip install pykeen") from exc
    return pykeen_dataset_to_dict(Kinships())


def load_wn18rr() -> Dict[str, Any]:
    try:
        from pykeen.datasets import WN18RR
    except Exception as exc:
        raise RuntimeError("WN18RR requires PyKEEN: pip install pykeen") from exc
    return pykeen_dataset_to_dict(WN18RR())


def load_fb15k() -> Dict[str, Any]:
    try:
        from pykeen.datasets import FB15k
    except Exception as exc:
        raise RuntimeError("FB15k requires PyKEEN: pip install pykeen") from exc
    return pykeen_dataset_to_dict(FB15k())


def load_fb15k237() -> Dict[str, Any]:
    try:
        from pykeen.datasets import FB15k237
    except Exception as exc:
        raise RuntimeError("FB15k-237 requires PyKEEN: pip install pykeen") from exc
    return pykeen_dataset_to_dict(FB15k237())


def load_yago310() -> Dict[str, Any]:
    try:
        from pykeen.datasets import YAGO310
    except Exception as exc:
        raise RuntimeError("YAGO3-10 requires PyKEEN: pip install pykeen") from exc
    return pykeen_dataset_to_dict(YAGO310())


def load_kinship_1990(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    entity_to_id: Dict[str, int] = {}
    relation_to_id: Dict[str, int] = {}
    pattern = re.compile(r"([^\(]+)\(([^,]+),\s*([^\)]+)\)")

    def entity_id(label: str) -> int:
        label = label.strip()
        if label not in entity_to_id:
            entity_to_id[label] = len(entity_to_id)
        return entity_to_id[label]

    def relation_id(label: str) -> int:
        label = label.strip()
        if label not in relation_to_id:
            relation_to_id[label] = len(relation_to_id)
        return relation_to_id[label]

    def parse(path: Path) -> List[Triple]:
        if not path.exists():
            raise FileNotFoundError(path)
        triples: List[Triple] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.fullmatch(line)
            if match:
                relation, head, tail = match.group(1), match.group(2), match.group(3)
            else:
                parts = line.replace("\t", " ").split()
                if len(parts) != 3:
                    raise ValueError(f"Cannot parse {path}: {raw_line!r}")
                head, relation, tail = parts
            triples.append((entity_id(head), relation_id(relation), entity_id(tail)))
        return triples

    train = parse(directory / "train.data")
    valid = parse(directory / "valid.data")
    test = parse(directory / "test.data")
    entities = [None] * len(entity_to_id)
    relations = [None] * len(relation_to_id)
    for label, index in entity_to_id.items():
        entities[index] = label
    for label, index in relation_to_id.items():
        relations[index] = label
    return {
        "entities": entities,
        "relations": relations,
        "train": train,
        "valid": valid,
        "test": test,
    }


def load_dataset(name: str, kinship1990_dir: Path) -> Dict[str, Any]:
    key = name.upper()
    if key == "NATIONS":
        data = load_nations()
    elif key == "KINSHIP":
        data = load_kinship()
    elif key == "KINSHIP_1990":
        data = load_kinship_1990(kinship1990_dir)
    elif key == "WN18RR":
        data = load_wn18rr()
    elif key == "FB15K":
        data = load_fb15k()
    elif key == "FB15K-237":
        data = load_fb15k237()
    elif key == "YAGO3-10":
        data = load_yago310()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data["N"] = len(data["entities"])
    data["M"] = len(data["relations"])
    return data


def build_filter_known(*splits: Iterable[Triple]):
    known = defaultdict(set)
    for split in splits:
        for h, r, t in split:
            known[(int(h), int(r))].add(int(t))
    return known


def evaluate_filtered_tail(
    model,
    triples: Sequence[Triple],
    filter_known: Mapping[Tuple[int, int], set],
    *,
    use_c_layer: bool,
    agg_mode: str,
    alpha: float,
    seed: int,
    top_rules_fraction: float = 1.0,
) -> Dict[str, float]:
    if not triples:
        raise ValueError("No triples supplied for evaluation")

    rng = random.Random(seed)
    scores_array = np.zeros(model.N, dtype=np.float64)
    ranks: List[int] = []

    for h, r, t in triples:
        scores = model.query(
            int(h), int(r),
            use_C_layer=bool(use_c_layer),
            agg_mode=agg_mode,
            alpha=float(alpha),
            top_rules_fraction=float(top_rules_fraction),
        )
        scores_array.fill(0.0)
        for entity, score in scores.items():
            scores_array[int(entity)] = float(score)

        for entity in filter_known.get((int(h), int(r)), set()):
            if int(entity) != int(t):
                scores_array[int(entity)] = -np.inf

        scores_array += np.fromiter(
            (rng.uniform(0.0, 1e-6) for _ in range(model.N)),
            dtype=np.float64,
            count=model.N,
        )
        target = int(t)
        higher = scores_array > scores_array[target] + 1e-12
        ties_before_target = (
            (np.abs(scores_array - scores_array[target]) <= 1e-12)
            & (np.arange(model.N) < target)
        )
        ties_before_target[target] = False
        rank = 1 + int(np.count_nonzero(higher)) + int(np.count_nonzero(ties_before_target))
        ranks.append(rank)

    rank_array = np.asarray(ranks, dtype=float)
    return {
        "MR": float(rank_array.mean()),
        "MRR": float((1.0 / rank_array).mean()),
        "Hits@1": float(100.0 * np.mean(rank_array <= 1)),
        "Hits@3": float(100.0 * np.mean(rank_array <= 3)),
        "Hits@10": float(100.0 * np.mean(rank_array <= 10)),
        "n_eval": int(len(ranks)),
    }


def make_config(module, params: Mapping[str, Any]):
    config_class = module.NSRConfig
    kwargs = {
        "eta_Z": 1.0,
        "eta_E": 0.0,
        "T_thresh": float(params["T_thresh"]),
        "T_infer": 5,
        "b_bias": 1.0,
        "w_zz_strength": 1.0,
        "i_e_delay": 2,
        "hold_R_until_e_delay": True,
        "inhibit_R_from": 3,
        "inhibit_value": 1e6,
        "use_rr_recurrence": True,
        "delta_RR": 0.5,
        "mu_RR": 0.05,
        "W_RR_normalize": "sigmoid",
        "W_RR_sigmoid_k": float(params["W_RR_sigmoid_k"]),
        "C_learning_T": 5,
        "C_activity_threshold": float(params["C_activity_threshold"]),
        "C_hebbian_mode": "ema",
        "delta_C": 0.1,
        "C_n_passes": 1,
        "C_min_support": int(params["C_min_support"]),
        "enable_C_layer_by_default": True,
        "C_rule_threshold": float(params["C_rule_threshold"]),
        "alpha": float(params["alpha"]),
        "enable_erz_cache": True,
        "learn_cache_reverse": bool(params.get("learn_cache_reverse", False)),
        "C_dense_memory_limit_bytes": int(params.get("C_dense_memory_limit_bytes", 512 * 1024 * 1024)),
        "C_fast_batching": bool(params.get("C_fast_batching", False)),
        "C_learning_algorithm": str(params.get("C_learning_algorithm", "online")),
        # Deprecated backward-compatibility aliases.
        "C_fast_approx": bool(params.get("C_fast_approx", False)),
        "C_fast_approx_v2": bool(params.get("C_fast_approx_v2", False)),
        # Fallback mechanisms ( Concepts_Fallback is the renamed re_fallback ).
        "tail_fallback_weight": float(params.get("tail_fallback_weight", 0.0)),
        "concepts_fallback_weight": float(params.get("concepts_fallback_weight", 0.0)),
        "concepts_fallback_k": int(params.get("concepts_fallback_k", 3)),
    }
    if is_dataclass(config_class):
        allowed = {field.name for field in fields(config_class)}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return config_class(**kwargs)


def train_model(module, data: Mapping[str, Any], params: Mapping[str, Any], seed: int):
    timings: Dict[str, float] = {}
    start_total = time.perf_counter()

    start = time.perf_counter()
    model = module.NSRDynamicERZC(data["N"], data["M"], make_config(module, params))
    timings["model_init_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    model.encode_triples(data["train"])
    timings["encode_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    model.learn_WRR_hebbian(
        data["train"], n_passes=1, normalize="sigmoid", seed=seed, verbose=False
    )
    timings["wrr_train_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    model.learn_C_layer(seed=seed, reset=True, verbose=False)
    timings["c_train_sec"] = time.perf_counter() - start
    timings["total_train_sec"] = time.perf_counter() - start_total
    return model, timings


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=Path("../NSR_Simplified.py"))
    parser.add_argument("--dataset", required=True, choices=["NATIONS", "KINSHIP", "KINSHIP_1990", "WN18RR", "FB15K", "FB15K-237", "YAGO3-10"])
    parser.add_argument("--kinship1990-dir", type=Path, default=Path("../../KINSHIP_1990"))
    parser.add_argument("--params-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("../Results"))
    parser.add_argument("--top-rules-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    module = load_module_from_file("nsr_model", args.model_file)

    if args.params_json:
        params = json.loads(args.params_json.read_text(encoding="utf-8"))
    else:
        # Sensible defaults; override with --params-json for tuned values.
        params = {
            "T_thresh": 0.0,
            "W_RR_sigmoid_k": 1.0,
            "C_activity_threshold": 0.5,
            "C_rule_threshold": 0.1,
            "C_min_support": 2,
            "agg_mode": "max",
            "alpha": 1.0,
            "learn_cache_reverse": True,
        }

    data = load_dataset(args.dataset, args.kinship1990_dir)
    print(
        f"[{args.dataset}] N={data['N']} M={data['M']} "
        f"train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])}"
    )

    t0 = time.perf_counter()
    model, timing = train_model(module, data, params, args.seed)
    train_wall = time.perf_counter() - t0

    known = build_filter_known(data["train"], data["valid"], data["test"])
    test = data["test"]

    t0 = time.perf_counter()
    no_c = evaluate_filtered_tail(
        model, test, known,
        use_c_layer=False,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=args.seed,
        top_rules_fraction=args.top_rules_fraction,
    )
    no_c_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    with_c = evaluate_filtered_tail(
        model, test, known,
        use_c_layer=True,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=args.seed,
        top_rules_fraction=args.top_rules_fraction,
    )
    with_c_sec = time.perf_counter() - t0

    result = {
        "dataset": args.dataset,
        "params": params,
        "top_rules_fraction": args.top_rules_fraction,
        "train_timing": timing,
        "train_wall_sec": train_wall,
        "C_learning_stats": model.C_learning_stats,
        "no_C": no_c,
        "with_C": with_c,
        "delta_MRR": with_c["MRR"] - no_c["MRR"],
        "delta_Hits@1": with_c["Hits@1"] - no_c["Hits@1"],
        "no_C_eval_sec": no_c_sec,
        "with_C_eval_sec": with_c_sec,
        "total_wall_sec": train_wall + no_c_sec + with_c_sec,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.dataset.lower()}_result.json"
    json_path.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
