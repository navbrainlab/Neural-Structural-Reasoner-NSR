"""Compare NSR_Fake_C_Dynamic_V2 and NSR_Simplified on FB15K / YAGO3-10.

FB15K has K = 2*M = 2690 directed relations; V2's dense K^3 W_comp_agent would
require ~78 GB, so V2 skips the C layer.  NSR_Simplified.py uses a hybrid
sparse/dense C layer and can run FB15K with the C layer.  YAGO3-10 has only
M = 37 relations (K = 74), so both models run the C layer.

By default only the first --max-test-queries test triples are evaluated, to
keep timing reasonable while still giving a fair head-to-head comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluate import build_filter_known, evaluate_filtered_tail, load_dataset

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


def make_config(module, params: Mapping[str, Any]):
    cfg_class = module.NSRConfig
    kwargs = {k: v for k, v in params.items() if k in cfg_class.__dataclass_fields__}
    return cfg_class(**kwargs)


def train_model(module, data: Mapping[str, Any], params: Mapping[str, Any], seed: int, run_c: bool):
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

    if run_c:
        start = time.perf_counter()
        model.learn_C_layer(seed=seed, reset=True, verbose=False)
        timings["c_train_sec"] = time.perf_counter() - start
    else:
        timings["c_train_sec"] = 0.0

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
    parser.add_argument("--dataset", required=True, choices=["FB15K", "FB15K-237", "YAGO3-10"])
    parser.add_argument("--v2-model-file", type=Path, default=Path("../NSR_Fake_C_Dynamic_V2.py"))
    parser.add_argument("--simplified-model-file", type=Path, default=Path("../NSR_Simplified.py"))
    parser.add_argument("--params-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("../Results/fb15k_yago"))
    parser.add_argument("--max-test-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset, Path("../../KINSHIP_1990"))
    print(
        f"[{args.dataset}] N={data['N']} M={data['M']} "
        f"train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])}"
    )

    if args.params_json:
        params = json.loads(args.params_json.read_text(encoding="utf-8"))
    else:
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

    # FB15K: V2's dense K^3 tensor would require ~78 GB, so V2 skips C.
    # NSR_Simplified.py uses a hybrid sparse/dense C layer and can run FB15K C.
    # FB15K-237 (K=474) fits in V2's dense memory (~425 MB) so both run C there.
    run_c_for = {"V2": args.dataset != "FB15K", "Simplified": True}
    if args.dataset == "FB15K":
        print("[FB15K] V2 skips C layer (dense K^3 tensor ~78 GB); Simplified runs sparse C layer.")

    test = data["test"][: args.max_test_queries]
    known = build_filter_known(data["train"], data["valid"], data["test"])

    results: Dict[str, Any] = {
        "dataset": args.dataset,
        "params": params,
        "max_test_queries": args.max_test_queries,
        "run_c_layer": run_c_for,
        "test_size_used": len(test),
    }

    for label, model_file in [("V2", args.v2_model_file), ("Simplified", args.simplified_model_file)]:
        print(f"\n=== {label} ===")
        module = load_module_from_file(f"nsr_{label.lower()}", model_file)
        label_run_c = run_c_for[label]

        t0 = time.perf_counter()
        model, timing = train_model(module, data, params, args.seed, label_run_c)
        train_wall = time.perf_counter() - t0
        print(f"train wall: {train_wall:.2f}s")

        # no-C eval
        t0 = time.perf_counter()
        no_c = evaluate_filtered_tail(
            model, test, known,
            use_c_layer=False,
            agg_mode=params["agg_mode"],
            alpha=params["alpha"],
            seed=args.seed,
            top_rules_fraction=1.0,
        )
        no_c_sec = time.perf_counter() - t0
        print(f"no-C MRR: {no_c['MRR']:.4f} ({no_c_sec:.2f}s)")

        # with-C eval
        if label_run_c:
            t0 = time.perf_counter()
            with_c = evaluate_filtered_tail(
                model, test, known,
                use_c_layer=True,
                agg_mode=params["agg_mode"],
                alpha=params["alpha"],
                seed=args.seed,
                top_rules_fraction=1.0,
            )
            with_c_sec = time.perf_counter() - t0
            print(f"with-C MRR: {with_c['MRR']:.4f} ({with_c_sec:.2f}s)")
        else:
            with_c = None
            with_c_sec = 0.0

        results[label] = {
            "train_timing": timing,
            "train_wall_sec": train_wall,
            "no_C": no_c,
            "no_C_eval_sec": no_c_sec,
            "with_C": with_c,
            "with_C_eval_sec": with_c_sec,
        }
        if label_run_c and model.C_learning_stats:
            results[label]["C_learning_stats"] = model.C_learning_stats

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.dataset.lower().replace('-', '_')}_comparison.json"
    out_path.write_text(json.dumps(json_ready(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
