"""Hyperparameter search for NSR/NSR_Dynamic.py.

This script runs Optuna over the faithful dynamics implementation on smaller
datasets (NATIONS, KINSHIP, KINSHIP_1990) and writes results to JSON/Markdown.

Usage (from project root):
    python NSR/NSR_assistant/tune_dynamic.py \
        --dataset KINSHIP_1990 --n-trials 30 \
        --output-dir NSR/Results/tune_dynamic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise RuntimeError("optuna is required for hyperparameter search") from exc

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ASSISTANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ASSISTANT_DIR))
from evaluate import (
    build_filter_known,
    evaluate_filtered_tail,
    load_dataset,
    make_config,
    json_ready,
)

Triple = Tuple[int, int, int]


def load_module_from_file(name: str, path: Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_model(
    model: Any,
    data: Mapping[str, Any],
    params: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    known = build_filter_known(data["train"], data["valid"], data["test"])
    top_rules_fraction = float(params.get("top_rules_fraction", 1.0))

    t0 = time.perf_counter()
    no_c = evaluate_filtered_tail(
        model, data["test"], known,
        use_c_layer=False,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=seed,
        top_rules_fraction=top_rules_fraction,
    )
    no_c_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    with_c = evaluate_filtered_tail(
        model, data["test"], known,
        use_c_layer=True,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=seed,
        top_rules_fraction=top_rules_fraction,
    )
    with_c_sec = time.perf_counter() - t0

    return {
        "no_C": no_c,
        "with_C": with_c,
        "delta_MRR": with_c["MRR"] - no_c["MRR"],
        "no_C_eval_sec": no_c_sec,
        "with_C_eval_sec": with_c_sec,
    }


def get_search_space(trial: optuna.Trial, dataset: str) -> Dict[str, Any]:
    # NATIONS has a very large active-rule set in faithful dynamics; without a
    # tight top_rules_fraction the per-trial evaluation can exceed the budget.
    if dataset == "NATIONS":
        top_rules_fraction_low, top_rules_fraction_high = 0.05, 0.30
    else:
        top_rules_fraction_low, top_rules_fraction_high = 0.01, 1.00
    params: Dict[str, Any] = {
        "T_thresh": trial.suggest_float("T_thresh", 0.0, 1.0),
        "W_RR_sigmoid_k": trial.suggest_float("W_RR_sigmoid_k", 0.1, 5.0),
        "C_activity_threshold": trial.suggest_float("C_activity_threshold", 0.5, 1.0),
        "C_rule_threshold": trial.suggest_float("C_rule_threshold", 0.05, 0.5),
        "C_min_support": trial.suggest_int("C_min_support", 1, 30),
        "agg_mode": trial.suggest_categorical("agg_mode", ["max", "add"]),
        "alpha": trial.suggest_float("alpha", 0.0, 2.0),
        "learn_cache_reverse": trial.suggest_categorical("learn_cache_reverse", [True, False]),
        # For datasets with many active rules (e.g. NATIONS), faithful dynamics
        # inference is too slow without pruning.  Include top_rules_fraction in
        # the search to trade off speed vs. rule coverage.
        "top_rules_fraction": trial.suggest_float(
            "top_rules_fraction", top_rules_fraction_low, top_rules_fraction_high
        ),
    }
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=Path("NSR/NSR_Dynamic.py"))
    parser.add_argument("--dataset", required=True,
                        choices=["NATIONS", "KINSHIP", "KINSHIP_1990"])
    parser.add_argument("--kinship1990-dir", type=Path, default=Path("KINSHIP_1990"))
    parser.add_argument("--output-dir", type=Path, default=Path("NSR/Results/tune_dynamic"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-rules-fraction", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    module = load_module_from_file("nsr_dynamic", args.model_file)
    data = load_dataset(args.dataset, args.kinship1990_dir)
    print(
        f"[{args.dataset}] N={data['N']} M={data['M']} "
        f"train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])}"
    )

    known = build_filter_known(data["train"], data["valid"], data["test"])

    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(trial, args.dataset)
        cfg = make_config(module, params)
        model = module.NSRDynamicERZC(data["N"], data["M"], cfg)

        start = time.perf_counter()
        model.encode_triples(data["train"])
        encode_sec = time.perf_counter() - start

        start = time.perf_counter()
        model.learn_WRR_hebbian(
            data["train"], n_passes=1, normalize="sigmoid",
            seed=args.seed, verbose=False,
        )
        wrr_sec = time.perf_counter() - start

        start = time.perf_counter()
        model.learn_C_layer(seed=args.seed, reset=True, verbose=False)
        c_sec = time.perf_counter() - start

        trial.set_user_attr("encode_sec", encode_sec)
        trial.set_user_attr("wrr_train_sec", wrr_sec)
        trial.set_user_attr("c_train_sec", c_sec)

        metrics = evaluate_filtered_tail(
            model, data["valid"], known,
            use_c_layer=True,
            agg_mode=params["agg_mode"],
            alpha=params["alpha"],
            seed=args.seed,
            top_rules_fraction=params["top_rules_fraction"],
        )
        return metrics["MRR"]

    print(f"[{args.dataset}] Starting Optuna search ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials)
    search_sec = time.perf_counter() - t0

    best_params = study.best_params
    print(f"[{args.dataset}] Best valid MRR: {study.best_value:.4f}")
    print(f"[{args.dataset}] Best params: {best_params}")

    # Final test evaluation with best params.
    cfg = make_config(module, best_params)
    final_model = module.NSRDynamicERZC(data["N"], data["M"], cfg)
    t0 = time.perf_counter()
    final_model.encode_triples(data["train"])
    encode_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_model.learn_WRR_hebbian(
        data["train"], n_passes=1, normalize="sigmoid",
        seed=args.seed, verbose=False,
    )
    wrr_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_model.learn_C_layer(seed=args.seed, reset=True, verbose=False)
    c_sec = time.perf_counter() - t0

    test_results = evaluate_model(final_model, data, best_params, args.seed)
    print(
        f"[{args.dataset}] Test no-C MRR={test_results['no_C']['MRR']:.4f} "
        f"with-C MRR={test_results['with_C']['MRR']:.4f}"
    )

    result = {
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "best_valid_mrr": study.best_value,
        "best_params": best_params,
        "encode_sec": encode_sec,
        "wrr_train_sec": wrr_sec,
        "c_train_sec": c_sec,
        "search_sec": search_sec,
        **test_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.dataset.lower()}_dynamic.json"
    save_json(json_path, result)
    print(f"Saved JSON: {json_path}")

    # Append to NSR/Final_Results.md.
    md_path = Path("NSR/Final_Results.md")
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {args.dataset} (NSR_Dynamic)\n\n")
        f.write(f"- Encode: {result['encode_sec']:.2f}s\n")
        f.write(f"- W_RR train: {result['wrr_train_sec']:.2f}s\n")
        f.write(f"- C-layer train: {result['c_train_sec']:.2f}s\n")
        f.write(f"- Search time: {result['search_sec']:.2f}s ({args.n_trials} trials)\n")
        f.write(f"- Best valid MRR: {result['best_valid_mrr']:.4f}\n")
        f.write(f"- Test no-C MRR: {result['no_C']['MRR']:.4f}\n")
        f.write(f"- Test with-C MRR: {result['with_C']['MRR']:.4f}\n")
        f.write(f"- Test Hits@1/3/10: {result['with_C']['Hits@1']:.2f}% / "
                f"{result['with_C']['Hits@3']:.2f}% / {result['with_C']['Hits@10']:.2f}%\n")
        f.write(f"- Best params: `{json.dumps(best_params, ensure_ascii=False)}`\n")
    print(f"Appended Markdown: {md_path}")


if __name__ == "__main__":
    main()
