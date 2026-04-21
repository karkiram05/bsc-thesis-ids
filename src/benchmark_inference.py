from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_FILE, NON_FEATURE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR

OUT = REPORTS_DIR / "benchmark"

WARMUP = 2
ROUNDS = 5
BATCH_SIZES = [1, 100, 1_000, 10_000, 100_000]


def _load_test_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    test = df[df["split_day"] == "test"].copy()
    feat_cols = [c for c in test.columns if c not in NON_FEATURE]
    return test[feat_cols]


def _predict_batch(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


def benchmark_model(name: str, model_path: Path, X_test: pd.DataFrame) -> dict:
    model = joblib.load(model_path)
    # keep DataFrame so sklearn does not warn about feature names
    X_df = X_test

    for _ in range(WARMUP):
        _predict_batch(model, X_df.iloc[:1000])

    results: dict[str, dict] = {}
    for bs in BATCH_SIZES:
        if bs > len(X_df):
            continue
        batch = X_df.iloc[:bs]
        elapsed = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            _predict_batch(model, batch)
            elapsed.append(time.perf_counter() - t0)
        mean_s = float(np.mean(elapsed))
        std_s = float(np.std(elapsed))
        per_flow_us = (mean_s / bs) * 1e6
        fps = bs / mean_s if mean_s > 0 else float("inf")
        results[str(bs)] = {
            "batch_size": bs,
            "mean_seconds": mean_s,
            "std_seconds": std_s,
            "per_flow_microseconds": per_flow_us,
            "flows_per_second": fps,
        }
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("[benchmark] loading test data ...")
    X_test = _load_test_data()
    print(f"[benchmark] test shape={X_test.shape}")

    # day-split is closer to real deployment than stratified
    model_dir = MODELS_DIR / "binary_day"
    model_files = {
        "logreg": model_dir / "logreg.joblib",
        "random_forest": model_dir / "random_forest.joblib",
        "xgboost": model_dir / "xgboost.joblib",
        "lightgbm": model_dir / "lightgbm.joblib",
    }

    all_results: dict[str, dict] = {}
    for name, path in model_files.items():
        if not path.exists():
            print(f"[benchmark] skip {name}: {path} missing")
            continue
        print(f"[benchmark] {name} ...")
        all_results[name] = benchmark_model(name, path, X_test)

    out_json = OUT / "inference_latency.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"[benchmark] wrote {out_json}")

    lines = [
        "# Inference Latency Benchmark (Binary Day Split)",
        "",
        "Measured on the held-out Friday test set. Warm-up: 2 rounds. Measured: 5 rounds per batch size.",
        "",
        "## Per-flow latency (microseconds, batch=10,000)",
        "",
        "| Model | μs / flow | Flows / second | Batch=10k elapsed (ms) |",
        "|---|---:|---:|---:|",
    ]
    bench_batch = "10000"
    for name, res in all_results.items():
        if bench_batch in res:
            r = res[bench_batch]
            lines.append(
                f"| {name} | {r['per_flow_microseconds']:.2f} | "
                f"{r['flows_per_second']:,.0f} | "
                f"{r['mean_seconds']*1000:.2f} |"
            )

    lines += [
        "",
        "## Scaling across batch sizes (flows/sec)",
        "",
        "| Model | bs=1 | bs=100 | bs=1k | bs=10k | bs=100k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, res in all_results.items():
        row = [f"| {name}"]
        for bs in ["1", "100", "1000", "10000", "100000"]:
            if bs in res:
                row.append(f"{res[bs]['flows_per_second']:,.0f}")
            else:
                row.append("—")
        lines.append(" | ".join(row) + " |")

    lines += [
        "",
        "## Operational interpretation",
        "",
        "Reference: a saturated 1 Gbps link at ~100k flows/min (≈1,667 flows/sec) needs any",
        "model above that rate to scan inline. All tree-based models comfortably exceed this",
        "with batch scoring. Per-flow latency at batch=10k is the relevant number for mirror",
        "mode / NIDS deployments where flows arrive in bursts from the collector (nProbe, Zeek).",
        "",
    ]
    out_md = OUT / "inference_latency.md"
    out_md.write_text("\n".join(lines))
    print(f"[benchmark] wrote {out_md}")

    fig, ax = plt.subplots(figsize=(9, 5))
    names, fps_vals = [], []
    for name, res in all_results.items():
        if bench_batch in res:
            names.append(name)
            fps_vals.append(res[bench_batch]["flows_per_second"])
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"][: len(names)]
    bars = ax.bar(names, fps_vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Flows / second (log scale)")
    ax.set_title("Inference throughput — binary IDS models (batch=10,000)")
    ax.axhline(1667, color="red", linestyle="--", alpha=0.6,
               label="1 Gbps line-rate (~1,667 flows/sec)")
    for b, v in zip(bars, fps_vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "inference_latency.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[benchmark] wrote {fig_path}")


if __name__ == "__main__":
    main()
