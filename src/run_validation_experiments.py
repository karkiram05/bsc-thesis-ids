"""Validation experiments: near-dup sensitivity, split policy, binary operating points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
)

from src.config import (
    DATA_FILE,
    REPORTS_DIR,
    RNG,
    feature_cols,
)

OUT = REPORTS_DIR / "validation"


def _train_eval_multiclass(Xt, yt, Xs, ys, n_classes):
    """Train and eval multi-class models, return macro_f1 and macro_recall per model."""
    import xgboost as xgb
    import lightgbm as lgb

    # drop rows where the encode() helper marked labels as unseen (-1), otherwise
    # every prediction for those rows is counted as a false positive for the
    # predicted class and precision is silently deflated.
    keep = ys != -1
    if (~keep).any():
        print(f"[val-exp] dropping {int((~keep).sum())} test rows with unseen labels")
        Xs = Xs.loc[keep] if hasattr(Xs, "loc") else Xs[keep]
        ys = ys[keep]
    keep_t = yt != -1
    if (~keep_t).any():
        Xt = Xt.loc[keep_t] if hasattr(Xt, "loc") else Xt[keep_t]
        yt = yt[keep_t]

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=2000, tol=1e-3,
                                       class_weight="balanced", random_state=RNG)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=24, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RNG, n_jobs=-1, verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200, max_depth=12, learning_rate=0.1,
            is_unbalance=True, random_state=RNG, n_jobs=-1, verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        # XGBoost and LightGBM need contiguous 0..N-1 labels
        if name in ("xgboost", "lightgbm"):
            train_classes = sorted(set(yt.tolist()))
            remap = {c: i for i, c in enumerate(train_classes)}
            yt_m = np.array([remap[v] for v in yt.tolist()], dtype=np.int32)
            model.fit(Xt, yt_m)
            raw_pred = model.predict(Xs)
            pred = np.array([train_classes[p] for p in raw_pred.tolist()], dtype=np.int64)
        else:
            model.fit(Xt, yt)
            pred = model.predict(Xs)

        pr, rc, f1, sup = precision_recall_fscore_support(
            ys, pred, labels=np.arange(n_classes), zero_division=0)
        has_support = sup > 0
        results[name] = {
            "macro_f1": float(np.mean(f1[has_support])) if has_support.any() else 0.0,
            "macro_recall": float(np.mean(rc[has_support])) if has_support.any() else 0.0,
        }
    return results


def exp1_near_duplicate_sensitivity(df: pd.DataFrame) -> None:
    """Exp 1: Compare baseline vs near-dup-filtered results."""
    print("[exp1] Near-duplicate sensitivity ...")
    out = OUT / "exp1"
    out.mkdir(parents=True, exist_ok=True)

    feat = feature_cols(df)
    split_col = "split_day"

    le = LabelEncoder()
    le.fit(df.loc[df[split_col] == "train", "attack_type"])
    known = set(le.classes_)
    n_classes = len(le.classes_)

    def encode(series):
        y = np.full(len(series), -1, dtype=np.int64)
        mask = series.isin(known)
        if mask.any():
            y[mask.to_numpy()] = le.transform(series[mask])
        return y

    def get_splits(data):
        Xt = data.loc[data[split_col] == "train", feat].copy()
        yt = encode(data.loc[data[split_col] == "train", "attack_type"])
        Xs = data.loc[data[split_col] == "test", feat].copy()
        ys = encode(data.loc[data[split_col] == "test", "attack_type"])
        return Xt, yt, Xs, ys

    # Baseline (full data)
    Xt, yt, Xs, ys = get_splits(df)
    baseline = _train_eval_multiclass(Xt, yt, Xs, ys, n_classes)

    # Filtered: remove exact duplicate feature rows across splits
    meta_cols = {"Label", "is_attack", "attack_type", "day", "source_file",
                 "split", "split_day", "split_strat", "row_hash"}
    feat_only = [c for c in df.columns if c not in meta_cols]
    df_filtered = df.drop_duplicates(subset=feat_only, keep="first")
    print(f"[exp1] Filtered: {len(df)} → {len(df_filtered)} rows "
          f"(dropped {len(df) - len(df_filtered)} exact feature duplicates)")

    Xt_f, yt_f, Xs_f, ys_f = get_splits(df_filtered)
    filtered = _train_eval_multiclass(Xt_f, yt_f, Xs_f, ys_f, n_classes)

    # Build table
    rows = []
    for model in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        b, f = baseline[model], filtered[model]
        rows.append({
            "Model": model,
            "Baseline Macro-F1": round(b["macro_f1"], 6),
            "Filtered Macro-F1": round(f["macro_f1"], 6),
            "Delta Macro-F1": round(f["macro_f1"] - b["macro_f1"], 6),
            "Baseline Macro-Recall": round(b["macro_recall"], 6),
            "Filtered Macro-Recall": round(f["macro_recall"], 6),
            "Delta Macro-Recall": round(f["macro_recall"] - b["macro_recall"], 6),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "table_V1_near_duplicate_sensitivity.csv", index=False)
    table.to_markdown(out / "table_V1_near_duplicate_sensitivity.md", index=False)

    # Grouped bar chart: baseline vs filtered F1
    models = table["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, table["Baseline Macro-F1"], w, label="Baseline", color="#4C72B0")
    ax.bar(x + w / 2, table["Filtered Macro-F1"], w, label="Near-dup filtered", color="#DD8452")
    ax.set_ylabel("Macro F1")
    ax.set_title("V1: Near-Duplicate Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.82, 0.90)
    ax.legend()
    for i, (b, f) in enumerate(zip(table["Baseline Macro-F1"], table["Filtered Macro-F1"])):
        delta = f - b
        ax.text(i, max(b, f) + 0.002, f"Δ={delta:+.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figure_V1_near_duplicate_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[exp1] wrote {out}")


def exp2_split_policy_sensitivity(df: pd.DataFrame) -> None:
    """Exp 2: Policy A (test=Fri) vs Policy B (test=Thu)."""
    print("[exp2] Split policy sensitivity ...")
    out = OUT / "exp2"
    out.mkdir(parents=True, exist_ok=True)

    feat = feature_cols(df)

    policies = {
        "A": {"train": ["Monday", "Tuesday", "Wednesday"], "val": "Thursday", "test": "Friday"},
        "B": {"train": ["Monday", "Tuesday"], "val": "Wednesday", "test": "Thursday"},
    }

    results = {}
    for policy_name, policy in policies.items():
        train_mask = df["day"].isin(policy["train"])
        test_mask = df["day"] == policy["test"]

        le = LabelEncoder()
        le.fit(df.loc[train_mask, "attack_type"])
        known = set(le.classes_)
        n_classes = len(le.classes_)

        def encode(series):
            y = np.full(len(series), -1, dtype=np.int64)
            mask = series.isin(known)
            if mask.any():
                y[mask.to_numpy()] = le.transform(series[mask])
            return y

        Xt = df.loc[train_mask, feat].copy()
        yt = encode(df.loc[train_mask, "attack_type"])
        Xs = df.loc[test_mask, feat].copy()
        ys = encode(df.loc[test_mask, "attack_type"])

        res = _train_eval_multiclass(Xt, yt, Xs, ys, n_classes)
        results[policy_name] = {"res": res, "test_day": policy["test"]}

    rows = []
    for model in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        a = results["A"]["res"][model]
        b = results["B"]["res"][model]
        rows.append({
            "Model": model,
            "Policy A test day": results["A"]["test_day"],
            "Policy A Macro-F1": round(a["macro_f1"], 6),
            "Policy B test day": results["B"]["test_day"],
            "Policy B Macro-F1": round(b["macro_f1"], 6),
            "Delta Macro-F1": round(b["macro_f1"] - a["macro_f1"], 6),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "table_V2_split_policy_sensitivity.csv", index=False)
    table.to_markdown(out / "table_V2_split_policy_sensitivity.md", index=False)

    # Grouped bar chart: policy A vs policy B
    models = table["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, table["Policy A Macro-F1"], w,
           label=f"Policy A (test={results['A']['test_day']})", color="#4C72B0")
    ax.bar(x + w / 2, table["Policy B Macro-F1"], w,
           label=f"Policy B (test={results['B']['test_day']})", color="#DD8452")
    ax.set_ylabel("Macro F1")
    ax.set_title("V2: Split Policy Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.02)
    ax.legend()
    for i, (a_f1, b_f1) in enumerate(zip(table["Policy A Macro-F1"], table["Policy B Macro-F1"])):
        delta = b_f1 - a_f1
        ax.text(i, max(a_f1, b_f1) + 0.005, f"Δ={delta:+.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figure_V2_split_policy_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"[exp2] wrote {out}")


def exp3_binary_operating_points(df: pd.DataFrame) -> None:
    """Exp 3: Binary operating points + recall-vs-FPR figure."""
    print("[exp3] Binary operating points ...")
    out = OUT / "exp3"
    out.mkdir(parents=True, exist_ok=True)

    feat = feature_cols(df)
    split_col = "split_day"

    def part(s):
        sub = df.loc[df[split_col] == s]
        X = sub[feat].copy()
        y = (sub["attack_type"].astype(str) != "Benign").astype(int).values
        return X, y

    Xt, yt = part("train")
    Xv, yv = part("val")
    Xs, ys = part("test")

    import xgboost as xgb

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=2000, tol=1e-3,
                                       class_weight="balanced", random_state=RNG)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=24, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RNG, n_jobs=-1, verbosity=0,
            scale_pos_weight=max((yt == 0).sum(), 1) / max(yt.sum(), 1),
        ),
    }

    # Try adding LightGBM if available
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            num_leaves=63, min_child_samples=20, is_unbalance=True,
            random_state=RNG, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    rows = []
    roc_data = {}

    for name, model in models.items():
        print(f"  [exp3] training {name} ...")
        model.fit(Xt, yt)
        proba_val = model.predict_proba(Xv)[:, 1]
        proba_test = model.predict_proba(Xs)[:, 1]

        # Best-F1 threshold from val (fine-grained)
        from sklearn.metrics import precision_recall_curve as prc
        prec_v, rec_v, thr_v = prc(yv, proba_val)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_v = np.where(
                (prec_v[:-1] + rec_v[:-1]) > 0,
                2 * prec_v[:-1] * rec_v[:-1] / (prec_v[:-1] + rec_v[:-1]),
                0.0,
            )
        best_idx = int(np.argmax(f1_v))
        best_thr = float(thr_v[best_idx])

        # FPR-constrained threshold (FPR <= 1% on val)
        fpr_v, tpr_v, thr_roc_v = roc_curve(yv, proba_val)
        fpr_ok = fpr_v <= 0.01
        if fpr_ok.any():
            # Best TPR at FPR <= 1%
            idx = np.where(fpr_ok)[0][-1]  # highest TPR within FPR constraint
            fpr_thr = float(thr_roc_v[min(idx, len(thr_roc_v) - 1)])
        else:
            fpr_thr = best_thr

        # Evaluate both operating points on test
        for op_name, thr in [("Best-F1", best_thr), ("FPR-constrained", fpr_thr)]:
            pred = (proba_test >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                ys, pred, average="binary", zero_division=0)
            n_neg = (ys == 0).sum()
            fp = ((pred == 1) & (ys == 0)).sum()
            fpr = float(fp / n_neg) if n_neg > 0 else 0.0
            rows.append({
                "Model": name,
                "Operating Point": op_name,
                "Threshold": round(thr, 4),
                "Test Precision": round(float(p), 4),
                "Test Recall": round(float(r), 4),
                "Test F1": round(float(f1), 4),
                "Test FPR": round(fpr, 4),
            })

        # ROC curve data for figure
        fpr_t, tpr_t, _ = roc_curve(ys, proba_test)
        roc_data[name] = (fpr_t, tpr_t)

    table = pd.DataFrame(rows)
    table.to_csv(out / "table_V3_binary_operating_points.csv", index=False)
    table.to_markdown(out / "table_V3_binary_operating_points.md", index=False)

    # Plot recall vs FPR
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"logreg": "#e74c3c", "random_forest": "#2ecc71",
              "xgboost": "#3498db", "lightgbm": "#9b59b6"}
    for name, (fpr_arr, tpr_arr) in roc_data.items():
        ax.plot(fpr_arr, tpr_arr, label=name, color=colors.get(name, "gray"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("Recall (True Positive Rate)", fontsize=12)
    ax.set_title("Binary Detection: Recall vs FPR (Day Split, Test = Friday)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.01, 0.15)  # Zoom into low-FPR region
    ax.set_ylim(0.5, 1.01)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "figure_V3_binary_recall_vs_fpr.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "figure_V3_binary_recall_vs_fpr.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[exp3] wrote {out}")


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data")

    df = pd.read_parquet(DATA_FILE)
    print(f"[validation] loaded {len(df)} rows")

    exp1_near_duplicate_sensitivity(df)
    exp2_split_policy_sensitivity(df)
    exp3_binary_operating_points(df)

    print("[validation] all experiments done.")


if __name__ == "__main__":
    main()
