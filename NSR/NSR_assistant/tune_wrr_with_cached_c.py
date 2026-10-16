"""Tune W_RR + fallback hyperparameters with a cached C layer.

Workflow:
    1. Load dataset and encode triples.
    2. Learn the C layer once with the fast matrix algorithm and save the model.
    3. Run Optuna over W_RR and fallback hyperparameters, relearning only W_RR
       and running inference for each trial.
    4. Evaluate the best valid configuration on the test set and write a JSON
       result plus a Markdown summary.

Usage (from project root):
    python NSR/NSR_assistant/tune_wrr_with_cached_c.py \
        --dataset FB15K-237 \
        --n-trials 30 \
        --output-dir NSR/Results/tune_wrr

The cached C-layer model is written to
    {cache_dir}/{dataset}_c_layer.npz
and reused on subsequent runs unless --relearn-c is passed.
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

# Allow importing evaluate.py helpers without adding it to sys.path permanently.
_ASSISTANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ASSISTANT_DIR))
from evaluate import (
    build_filter_known,
    evaluate_filtered_tail,
    load_dataset,
    make_config,
    mapped_triples_to_list,
    pykeen_dataset_to_dict,
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


def learn_and_cache_c_layer(
    module: Any,
    data: Mapping[str, Any],
    cfg_kwargs: Mapping[str, Any],
    cache_path: Path,
    seed: int,
) -> Tuple[Any, Dict[str, float]]:
    """Build model, encode triples, learn C layer with matrix algorithm, save."""
    timings: Dict[str, float] = {}

    start = time.perf_counter()
    model = module.NSRDynamicERZC(data["N"], data["M"], make_config(module, cfg_kwargs))
    timings["model_init_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    model.encode_triples(data["train"])
    timings["encode_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    model.learn_C_layer(seed=seed, reset=True, verbose=False)
    timings["c_train_sec"] = time.perf_counter() - start

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_path))
    timings["save_sec"] = time.perf_counter() - start - timings["c_train_sec"]

    return model, timings


def load_cached_model(module: Any, cache_path: Path) -> Any:
    return module.NSRDynamicERZC.load(str(cache_path))


def get_search_space(trial: optuna.Trial, dataset: str) -> Dict[str, Any]:
    """Return a hyperparameter dict for one Optuna trial."""
    # W_RR learning: cooccurrence is much faster and works well on large graphs.
    # We still expose online as a categorical so the search can pick it.
    wrr_algo = trial.suggest_categorical("W_RR_learning_algorithm", ["cooccurrence", "online"])

    params: Dict[str, Any] = {
        "T_thresh": trial.suggest_float("T_thresh", 0.0, 1.0),
        "W_RR_learning_algorithm": wrr_algo,
        "W_RR_normalize": trial.suggest_categorical("W_RR_normalize", ["sigmoid", "max", "none"]),
        "W_RR_sigmoid_k": trial.suggest_float("W_RR_sigmoid_k", 0.1, 5.0),
        "agg_mode": trial.suggest_categorical("agg_mode", ["max", "add"]),
        "alpha": trial.suggest_float("alpha", 0.0, 2.0),
        "tail_fallback_weight": trial.suggest_float("tail_fallback_weight", 0.0, 0.05),
        "concepts_fallback_weight": trial.suggest_float("concepts_fallback_weight", 0.0, 1.0),
        "concepts_fallback_k": trial.suggest_int("concepts_fallback_k", 1, 10),
    }

    if wrr_algo == "online":
        params["delta_RR"] = trial.suggest_float("delta_RR", 0.05, 1.0)
        params["mu_RR"] = trial.suggest_float("mu_RR", 0.0, 0.2)
    else:
        # Cooccurrence ignores these, but keep them in the dict for make_config.
        params["delta_RR"] = 0.5
        params["mu_RR"] = 0.05

    # C-layer thresholds are learned once and reused; we still search them because
    # the cached C layer uses whatever thresholds were used when it was created.
    # To avoid mismatch, we keep C params fixed to the cached values in the
    # objective (see make_trial_cfg).  These suggestions are kept only for
    # completeness / if the user passes --relearn-c.
    params["C_activity_threshold"] = trial.suggest_float("C_activity_threshold", 0.5, 1.0)
    params["C_rule_threshold"] = trial.suggest_float("C_rule_threshold", 0.05, 0.5)
    params["C_min_support"] = trial.suggest_int("C_min_support", 1, 30)

    return params


def make_trial_cfg(
    module: Any,
    base_cfg: Mapping[str, Any],
    trial_params: Mapping[str, Any],
) -> Any:
    """Build an NSRConfig for a trial, keeping C-layer params equal to cached config."""
    merged = dict(base_cfg)
    merged.update(trial_params)
    # C-layer params must match the cached C layer, so we overwrite them with the
    # base (cached) values.  This guarantees that only W_RR + inference params vary.
    for key in ("C_activity_threshold", "C_rule_threshold", "C_min_support"):
        merged[key] = base_cfg[key]
    return make_config(module, merged)


def evaluate_model(
    model: Any,
    data: Mapping[str, Any],
    params: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a model on valid and test sets."""
    known = build_filter_known(data["train"], data["valid"], data["test"])

    t0 = time.perf_counter()
    no_c = evaluate_filtered_tail(
        model, data["test"], known,
        use_c_layer=False,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=seed,
    )
    no_c_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    with_c = evaluate_filtered_tail(
        model, data["test"], known,
        use_c_layer=True,
        agg_mode=params["agg_mode"],
        alpha=params["alpha"],
        seed=seed,
    )
    with_c_sec = time.perf_counter() - t0

    return {
        "no_C": no_c,
        "with_C": with_c,
        "delta_MRR": with_c["MRR"] - no_c["MRR"],
        "no_C_eval_sec": no_c_sec,
        "with_C_eval_sec": with_c_sec,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=Path("NSR/NSR_Simplified.py"))
    parser.add_argument("--dataset", required=True,
                        choices=["NATIONS", "KINSHIP", "KINSHIP_1990", "WN18RR",
                                 "FB15K", "FB15K-237", "YAGO3-10"])
    parser.add_argument("--kinship1990-dir", type=Path, default=Path("../../KINSHIP_1990"))
    parser.add_argument("--output-dir", type=Path, default=Path("../Results/tune_wrr"))
    parser.add_argument("--cache-dir", type=Path, default=Path("../Results/tune_wrr/cache"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--relearn-c", action="store_true",
                        help="Relearn the C layer even if a cached file exists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-rules-fraction", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    module = load_module_from_file("nsr_model", args.model_file)
    data = load_dataset(args.dataset, args.kinship1990_dir)
    print(
        f"[{args.dataset}] N={data['N']} M={data['M']} "
        f"train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])}"
    )

    cache_path = args.cache_dir / f"{args.dataset.lower()}_c_layer.npz"

    # Base C-layer config: force matrix algorithm and sparse storage for large K.
    base_cfg: Dict[str, Any] = {
        "T_thresh": 0.0,
        "W_RR_sigmoid_k": 1.0,
        "C_activity_threshold": 0.5,
        "C_rule_threshold": 0.1,
        "C_min_support": 2,
        "C_learning_algorithm": "matrix",
        "C_dense_memory_limit_bytes": 1,  # always use sparse per-pair storage
        "agg_mode": "max",
        "alpha": 1.0,
        "tail_fallback_weight": 0.0,
        "concepts_fallback_weight": 0.0,
        "concepts_fallback_k": 3,
    }

    if cache_path.exists() and not args.relearn_c:
        print(f"[{args.dataset}] Loading cached C layer from {cache_path}")
        model = load_cached_model(module, cache_path)
        c_timings = {"c_train_sec": 0.0, "cached": True}
    else:
        print(f"[{args.dataset}] Learning C layer with matrix algorithm...")
        model, c_timings = learn_and_cache_c_layer(
            module, data, base_cfg, cache_path, args.seed
        )
        print(
            f"[{args.dataset}] C layer done: {c_timings['c_train_sec']:.2f}s, "
            f"active_rules={model.C_learning_stats.get('active_rules', -1)}"
        )

    # Update base_cfg to match whatever was actually used for the C layer.
    base_cfg["C_activity_threshold"] = float(model.cfg.C_activity_threshold)
    base_cfg["C_rule_threshold"] = float(model.cfg.C_rule_threshold)
    base_cfg["C_min_support"] = int(model.cfg.C_min_support)

    known = build_filter_known(data["train"], data["valid"], data["test"])

    def objective(trial: optuna.Trial) -> float:
        trial_params = get_search_space(trial, args.dataset)
        # Reuse the cached model structure but relearn W_RR with trial params.
        trial_model = load_cached_model(module, cache_path)
        trial_model.cfg = make_trial_cfg(module, base_cfg, trial_params)

        start = time.perf_counter()
        trial_model.learn_WRR_hebbian(
            data["train"],
            n_passes=1,
            normalize=trial_params["W_RR_normalize"],
            seed=args.seed,
            reset=True,
            verbose=False,
        )
        wrr_time = time.perf_counter() - start

        metrics = evaluate_filtered_tail(
            trial_model, data["valid"], known,
            use_c_layer=True,
            agg_mode=trial_params["agg_mode"],
            alpha=trial_params["alpha"],
            seed=args.seed,
            top_rules_fraction=args.top_rules_fraction,
        )
        trial.set_user_attr("wrr_train_sec", wrr_time)
        return metrics["MRR"]

    print(f"[{args.dataset}] Starting Optuna search ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials)
    search_sec = time.perf_counter() - t0

    best_params = study.best_params
    print(f"[{args.dataset}] Best valid MRR: {study.best_value:.4f}")
    print(f"[{args.dataset}] Best params: {best_params}")

    # Final evaluation on test set with best valid params.
    final_model = load_cached_model(module, cache_path)
    final_model.cfg = make_trial_cfg(module, base_cfg, best_params)
    start = time.perf_counter()
    final_model.learn_WRR_hebbian(
        data["train"],
        n_passes=1,
        normalize=best_params["W_RR_normalize"],
        seed=args.seed,
        reset=True,
        verbose=False,
    )
    final_wrr_time = time.perf_counter() - start

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
        "c_learning_stats": model.C_learning_stats,
        "c_train_sec": c_timings.get("c_train_sec", 0.0),
        "search_sec": search_sec,
        "final_wrr_train_sec": final_wrr_time,
        **test_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.dataset.lower()}_tune_wrr.json"
    save_json(json_path, result)
    print(f"Saved JSON: {json_path}")

    # Append to NSR/Final_Results.md.
    md_path = Path("NSR/Final_Results.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {args.dataset}\n\n")
        f.write(f"- C-layer train time: {result['c_train_sec']:.2f}s\n")
        f.write(f"- W_RR train time (best trial): {result['final_wrr_train_sec']:.4f}s\n")
        f.write(f"- Search time: {result['search_sec']:.2f}s ({args.n_trials} trials)\n")
        f.write(f"- Best valid MRR: {result['best_valid_mrr']:.4f}\n")
        f.write(f"- Test no-C MRR: {result['no_C']['MRR']:.4f}\n")
        f.write(f"- Test with-C MRR: {result['with_C']['MRR']:.4f}\n")
        f.write(f"- Test Hits@1/3/10: {result['with_C']['Hits@1']:.2f}% / "
                f"{result['with_C']['Hits@3']:.2f}% / {result['with_C']['Hits@10']:.2f}%\n")
        f.write(f"- Test no-C eval time: {result['no_C_eval_sec']:.2f}s\n")
        f.write(f"- Test with-C eval time: {result['with_C_eval_sec']:.2f}s\n")
        f.write(f"- Best params: `{json.dumps(best_params, ensure_ascii=False)}`\n")
    print(f"Appended Markdown: {md_path}")


if __name__ == "__main__":
    main()
