from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_FILE, MODELS_DIR, REPORTS_DIR, SPLIT_COL_DAY, feature_cols
from src.config import RNG as SEED

sys.stdout.reconfigure(line_buffering=True)


RNG = np.random.default_rng(SEED)
N_ATTACK_SAMPLES = 500
N_TRAIN_AUG = 2000
T_STEPS = 15
K_CANDIDATES = 6
EPSILONS_SIGMA = [0.25, 0.5, 1.0, 2.0]

PERTURBABLE = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Min", "Bwd IAT Mean",
    "Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Mean",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "Total Backward Packets", "Subflow Bwd Packets",
    "Bwd Packets Length Total",
    "Avg Packet Size", "Avg Bwd Segment Size",
]


def load_data_and_model():
    df = pd.read_parquet(DATA_FILE)
    feats = feature_cols(df)
    train = df[df[SPLIT_COL_DAY] == "train"]
    test = df[df[SPLIT_COL_DAY] == "test"]
    print(f"train rows: {len(train):,}   test rows: {len(test):,}   features: {len(feats)}")

    mdl_dir = Path(MODELS_DIR) / "binary_day"
    rf: RandomForestClassifier = joblib.load(mdl_dir / "random_forest.joblib")
    xgb = joblib.load(mdl_dir / "xgboost.joblib")

    with open(REPORTS_DIR / "metrics_day_binary" / "binary_metrics.json") as f:
        metrics = json.load(f)
    theta_rf = metrics["random_forest"]["best_threshold_from_val"]
    theta_xgb = metrics["xgboost"]["best_threshold_from_val"]
    print(f"RF threshold (from val):  {theta_rf:.6f}")
    print(f"XGB threshold (from val): {theta_xgb:.6f}")
    return df, feats, train, test, rf, xgb, theta_rf, theta_xgb


def compute_benign_scale(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    benign = df[(df[SPLIT_COL_DAY] == "train") & (df["is_attack"] == 0)]
    scale = benign[feats].std(ddof=0).replace(0, 1.0)
    return scale


def greedy_evade(
    x: np.ndarray,
    model,
    theta: float,
    feat_idx: list[int],
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    t_steps: int = T_STEPS,
    k_candidates: int = K_CANDIDATES,
) -> tuple[np.ndarray, bool, int, np.ndarray]:
    x_adv = x.copy()
    all_mags = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
    half = k_candidates // 2
    mags = np.array(all_mags[4 - half:4] + all_mags[4:4 + half])
    for step in range(t_steps):
        p = model.predict_proba(x_adv.reshape(1, -1))[0, 1]
        if p < theta:
            return x_adv, True, step, np.abs(x_adv - x)
        n_cands = len(feat_idx) * len(mags)
        candidates = np.broadcast_to(x_adv, (n_cands, len(x))).copy()
        flat = 0
        for j in feat_idx:
            for m in mags:
                candidates[flat, j] = np.clip(x_adv[j] + m * scale[j], lower[j], upper[j])
                flat += 1
        probs = model.predict_proba(candidates)[:, 1]
        best = int(np.argmin(probs))
        j_best = feat_idx[best // len(mags)]
        m_best = mags[best % len(mags)]
        new_val = np.clip(x_adv[j_best] + m_best * scale[j_best], lower[j_best], upper[j_best])
        if abs(new_val - x_adv[j_best]) < 1e-12:
            break
        x_adv[j_best] = new_val
    flipped = model.predict_proba(x_adv.reshape(1, -1))[0, 1] < theta
    return x_adv, flipped, t_steps, np.abs(x_adv - x)


def evaluate_adversarial(
    X_clean: np.ndarray,
    model,
    theta: float,
    feat_idx: list[int],
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    label: str,
) -> dict:
    print(f"\n[{label}] running greedy evasion on {len(X_clean)} attack flows...")
    t0 = time.time()
    adv = np.zeros_like(X_clean)
    flipped = np.zeros(len(X_clean), dtype=bool)
    n_steps = np.zeros(len(X_clean), dtype=int)
    perf_delta = np.zeros_like(X_clean)
    for i in range(len(X_clean)):
        x = X_clean[i]
        x_a, ok, s, d = greedy_evade(x, model, theta, feat_idx, scale, lower, upper)
        adv[i] = x_a
        flipped[i] = ok
        n_steps[i] = s
        perf_delta[i] = d
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(X_clean)}  evasion rate so far: {flipped[:i+1].mean():.3f}"
                  f"  elapsed: {time.time()-t0:.1f}s", flush=True)
    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s. evasion rate: {flipped.mean():.4f}")
    perf_delta_sigma = perf_delta / scale[np.newaxis, :]
    linf_sigma = perf_delta_sigma.max(axis=1)
    return {
        "adv": adv,
        "flipped": flipped,
        "n_steps": n_steps,
        "perf_delta": perf_delta,
        "linf_sigma": linf_sigma,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    out_dir = REPORTS_DIR / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, feats, train, test, rf, xgb, theta_rf, theta_xgb = load_data_and_model()
    scale_ser = compute_benign_scale(df, feats)
    scale = scale_ser.values.astype(float)

    lower = np.percentile(train[feats].values.astype(float), 0.5, axis=0)
    lower = np.maximum(lower, 0.0)
    upper = np.percentile(train[feats].values.astype(float), 99.5, axis=0)

    feat_idx = [feats.index(f) for f in PERTURBABLE if f in feats]
    print(f"perturbable features: {len(feat_idx)} / {len(feats)}")

    X_test = test[feats].values.astype(float)
    y_test = test["is_attack"].values.astype(int)
    p_test = rf.predict_proba(X_test)[:, 1]
    correct_attack = (y_test == 1) & (p_test >= theta_rf)
    print(f"correctly classified attack flows in test: {correct_attack.sum():,}")

    idx_all = np.where(correct_attack)[0]
    idx = RNG.choice(idx_all, size=min(N_ATTACK_SAMPLES, len(idx_all)), replace=False)
    X_adv_source = X_test[idx]

    r_clean = evaluate_adversarial(X_adv_source, rf, theta_rf, feat_idx, scale, lower, upper, "RF-clean")

    p_xgb_on_adv = xgb.predict_proba(r_clean["adv"])[:, 1]
    p_xgb_on_clean = xgb.predict_proba(X_adv_source)[:, 1]
    xgb_evaded_adv = (p_xgb_on_adv < theta_xgb).mean()
    xgb_evaded_clean = (p_xgb_on_clean < theta_xgb).mean()
    print(f"XGBoost evasion on clean samples:   {xgb_evaded_clean:.4f}")
    print(f"XGBoost evasion on adv. samples:    {xgb_evaded_adv:.4f}")

    print("\n=== DEFENSE: adversarial training ===")
    X_tr = train[feats].values.astype(float)
    y_tr = train["is_attack"].values.astype(int)
    tr_attack_idx = np.where(y_tr == 1)[0]
    aug_idx = RNG.choice(tr_attack_idx, size=min(N_TRAIN_AUG, len(tr_attack_idx)), replace=False)
    print(f"generating {len(aug_idx)} adversarial training samples (this takes a while)...")
    aug_src = X_tr[aug_idx]
    r_aug = evaluate_adversarial(aug_src, rf, theta_rf, feat_idx, scale, lower, upper, "RF-aug-gen")

    X_aug = np.vstack([X_tr, r_aug["adv"]])
    y_aug = np.concatenate([y_tr, np.ones(len(r_aug["adv"]), dtype=int)])

    print(f"retraining RF on augmented set: {X_aug.shape[0]:,} rows...")
    rf_def = RandomForestClassifier(
        n_estimators=300, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED,
    )
    t0 = time.time()
    rf_def.fit(X_aug, y_aug)
    print(f"retrain took {time.time()-t0:.1f}s")

    val = df[df[SPLIT_COL_DAY] == "val"]
    p_val = rf_def.predict_proba(val[feats].values.astype(float))[:, 1]
    y_val = val["is_attack"].values.astype(int)
    from sklearn.metrics import roc_curve
    fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val)
    j = tpr_v - fpr_v
    theta_def = float(thr_v[int(np.argmax(j))])
    print(f"defended RF threshold (val Youden's J): {theta_def:.6f}")

    p_def_clean_full = rf_def.predict_proba(X_test)[:, 1]
    pred_def_clean_full = (p_def_clean_full >= theta_def).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    f1_def_clean = f1_score(y_test, pred_def_clean_full)
    auc_def_clean = roc_auc_score(y_test, p_def_clean_full)
    prec_def_clean = precision_score(y_test, pred_def_clean_full)
    rec_def_clean = recall_score(y_test, pred_def_clean_full)
    print(f"defended RF clean:  F1={f1_def_clean:.4f}  ROC-AUC={auc_def_clean:.4f}  "
          f"P={prec_def_clean:.4f}  R={rec_def_clean:.4f}")

    r_def = evaluate_adversarial(X_adv_source, rf_def, theta_def, feat_idx, scale, lower, upper, "RF-defended")

    pred_clean = (p_test >= theta_rf).astype(int)
    f1_clean = f1_score(y_test, pred_clean)
    auc_clean = roc_auc_score(y_test, p_test)
    prec_clean = precision_score(y_test, pred_clean)
    rec_clean = recall_score(y_test, pred_clean)

    curve_clean = {}
    curve_def = {}
    for eps in EPSILONS_SIGMA:
        rob_clean = ((~r_clean["flipped"]) | (r_clean["linf_sigma"] > eps)).mean()
        rob_def = ((~r_def["flipped"]) | (r_def["linf_sigma"] > eps)).mean()
        curve_clean[f"{eps}σ"] = float(rob_clean)
        curve_def[f"{eps}σ"] = float(rob_def)

    used = r_clean["perf_delta"][r_clean["flipped"]]
    feat_usage = (used > 0).mean(axis=0)
    top_used = np.argsort(feat_usage)[::-1][:10]
    top_feats = [(feats[i], float(feat_usage[i])) for i in top_used]

    results = {
        "threat_model": {
            "knowledge": "score-query black-box (probability output only, no gradients or weights)",
            "goal": "evasion (flip a true attack flow to be predicted as benign)",
            "attack_algorithm": "greedy coordinate-ascent over sigma-scaled perturbations",
            "perturbable_features_count": len(feat_idx),
            "held_fixed_count": len(feats) - len(feat_idx),
            "budget_units": "L-inf in units of benign-feature sigma",
            "max_iterations_per_flow": T_STEPS,
            "candidates_per_feature": K_CANDIDATES,
        },
        "defense_method": "naive single-shot adversarial training (Goodfellow 2015 style, not Madry 2018 iterative PGD-AT)",
        "victim": "random_forest (binary day split)",
        "theta_rf": float(theta_rf),
        "theta_xgb": float(theta_xgb),
        "theta_defended": float(theta_def),
        "n_attack_samples": int(len(X_adv_source)),
        "clean_undefended": {
            "f1": float(f1_clean),
            "precision": float(prec_clean),
            "recall": float(rec_clean),
            "roc_auc": float(auc_clean),
        },
        "clean_defended": {
            "f1": float(f1_def_clean),
            "precision": float(prec_def_clean),
            "recall": float(rec_def_clean),
            "roc_auc": float(auc_def_clean),
        },
        "evasion_rate_vs_rf_undefended": float(r_clean["flipped"].mean()),
        "evasion_rate_vs_rf_defended": float(r_def["flipped"].mean()),
        "median_linf_to_flip_undefended_sigma": float(np.median(r_clean["linf_sigma"][r_clean["flipped"]])) if r_clean["flipped"].any() else None,
        "median_linf_to_flip_defended_sigma": float(np.median(r_def["linf_sigma"][r_def["flipped"]])) if r_def["flipped"].any() else None,
        "transferability": {
            "xgb_evasion_on_clean": float(xgb_evaded_clean),
            "xgb_evasion_on_rf_adversarial": float(xgb_evaded_adv),
        },
        "robust_accuracy_curve_undefended": curve_clean,
        "robust_accuracy_curve_defended": curve_def,
        "top_features_exploited": top_feats,
    }

    with open(out_dir / "adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir/'adversarial_results.json'}")

    md = []
    md.append("# Adversarial Robustness on Binary Day Split (Random Forest)\n")
    md.append("## Threat model\n")
    md.append("- Adversary knowledge: **score-query black-box** (probability output only, no gradients or weights).")
    md.append("- Adversary goal: evasion, flip a true attack flow to be predicted as benign.")
    md.append(f"- Perturbable features: {len(feat_idx)} of {len(feats)} (timing, packet length, packet count, flow duration).")
    md.append(f"- Features held fixed: TCP flags, header lengths, init window, protocol ID ({len(feats)-len(feat_idx)} features).")
    md.append("- Budget: L-infinity perturbation expressed in units of benign-feature sigma.")
    md.append(f"- Attack algorithm: greedy coordinate-ascent, up to {T_STEPS} iterations per flow, {K_CANDIDATES} symmetric candidate magnitudes per feature.")
    md.append("- Defence evaluated: naive single-shot adversarial training (Goodfellow 2015 style). The defended RF uses identical hyperparameters to the baseline RF, so any clean-F1 change is attributable to data augmentation, not to capacity differences.\n")
    md.append("## Clean-vs-adversarial accuracy\n")
    md.append("| Setting | F1 | Precision | Recall | ROC-AUC |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| RF clean (undefended) | {f1_clean:.4f} | {prec_clean:.4f} | {rec_clean:.4f} | {auc_clean:.4f} |")
    md.append(f"| RF clean (adv-trained) | {f1_def_clean:.4f} | {prec_def_clean:.4f} | {rec_def_clean:.4f} | {auc_def_clean:.4f} |\n")

    md.append("## Evasion rate under greedy attack\n")
    md.append(f"Sampled {len(X_adv_source):,} correctly-classified attack flows.\n")
    md.append("| Model | Evasion rate | Median L∞ to flip (σ) |")
    md.append("|---|---:|---:|")
    u_med = results["median_linf_to_flip_undefended_sigma"]
    d_med = results["median_linf_to_flip_defended_sigma"]
    md.append(f"| Undefended RF | {r_clean['flipped'].mean():.4f} | {u_med:.3f} |" if u_med else
              f"| Undefended RF | {r_clean['flipped'].mean():.4f} | n/a |")
    md.append(f"| Adv-trained RF | {r_def['flipped'].mean():.4f} | {d_med:.3f} |" if d_med else
              f"| Adv-trained RF | {r_def['flipped'].mean():.4f} | n/a |")
    md.append("")

    md.append("## Robust accuracy vs perturbation budget\n")
    md.append("Robust accuracy = fraction of attack flows whose prediction survives within the given L∞ budget.\n")
    hdr = "| Model | " + " | ".join([f"ε = {e}σ" for e in EPSILONS_SIGMA]) + " |"
    sep = "|---|" + "|".join([":---:"] * len(EPSILONS_SIGMA)) + "|"
    row_c = "| Undefended RF | " + " | ".join([f"{curve_clean[f'{e}σ']:.3f}" for e in EPSILONS_SIGMA]) + " |"
    row_d = "| Adv-trained RF | " + " | ".join([f"{curve_def[f'{e}σ']:.3f}" for e in EPSILONS_SIGMA]) + " |"
    md += [hdr, sep, row_c, row_d, ""]

    md.append("## Transferability (attacks crafted vs RF, scored by XGBoost)\n")
    md.append("| Input to XGBoost | Evasion rate |")
    md.append("|---|---:|")
    md.append(f"| Clean attack flows | {xgb_evaded_clean:.4f} |")
    md.append(f"| RF-adversarial flows | {xgb_evaded_adv:.4f} |\n")

    md.append("## Top features exploited in successful attacks\n")
    md.append("Fraction of successful evasions in which each feature was moved.\n")
    md.append("| Feature | % of successful attacks moving this feature |")
    md.append("|---|---:|")
    for f_name, frac in top_feats:
        md.append(f"| {f_name} | {frac*100:.1f} |")
    md.append("")

    md.append("## Interpretation\n")
    md.append("- If evasion rate is high at small ε, the detector is brittle under realistic, low-effort perturbations.")
    md.append("- Adversarial training shifts the robust-accuracy curve upward at the cost of some clean-test performance.")
    md.append("- High XGBoost transferability means adversarial examples generalise across the tree-ensemble family, so a defender cannot rely on model secrecy.")
    md.append("- The features most often moved tell the defender which flow statistics need additional sanity checks (e.g. server-side enforcement of minimum packet timing).")

    with open(out_dir / "adversarial_results.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {out_dir/'adversarial_results.md'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    xs = EPSILONS_SIGMA
    ax.plot(xs, [curve_clean[f"{e}σ"] for e in xs], marker="o", label="Undefended RF", linewidth=2)
    ax.plot(xs, [curve_def[f"{e}σ"] for e in xs], marker="s", label="Adv-trained RF", linewidth=2)
    ax.set_xlabel("Attacker budget ε (σ of benign feature)")
    ax.set_ylabel("Robust accuracy on attack flows")
    ax.set_title("Robustness curve")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    names = [f for f, _ in top_feats][::-1]
    vals = [v * 100 for _, v in top_feats][::-1]
    ax.barh(range(len(names)), vals, color="#c0392b", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("% of successful evasions moving this feature")
    ax.set_title("Top features exploited by the attack")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_fig = fig_dir / "adversarial_robustness.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_fig}")

    print("\n=== SUMMARY ===")
    print(f"Evasion rate vs undefended RF: {r_clean['flipped'].mean():.4f}")
    print(f"Evasion rate vs adv-trained RF: {r_def['flipped'].mean():.4f}")
    print(f"Clean F1 (undef -> adv-trained): {f1_clean:.4f} -> {f1_def_clean:.4f}")
    print(f"XGB evasion on RF-adv samples: {xgb_evaded_adv:.4f}")


if __name__ == "__main__":
    main()
