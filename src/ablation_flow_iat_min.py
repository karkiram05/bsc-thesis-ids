from __future__ import annotations
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve

from src.config import (
    DATA_FILE, REPORTS_DIR, SPLIT_COL_DAY, RNG, feature_cols
)

from src.adversarial_eval import (
    evaluate_adversarial, PERTURBABLE, N_ATTACK_SAMPLES,
)


DROP_FEATURE = "Flow IAT Min"
OUT_MD = REPORTS_DIR / "adversarial" / "ablation_flow_iat_min.md"


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run `make data` first.")

    print(f"[ablation] loading {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    full_feats = feature_cols(df)
    if DROP_FEATURE not in full_feats:
        raise SystemExit(f"{DROP_FEATURE} not in feature set; nothing to drop.")
    feats = [c for c in full_feats if c != DROP_FEATURE]
    print(f"[ablation] features kept: {len(feats)} (dropped '{DROP_FEATURE}')")

    train = df[df[SPLIT_COL_DAY] == "train"]
    val = df[df[SPLIT_COL_DAY] == "val"]
    test = df[df[SPLIT_COL_DAY] == "test"]

    X_tr = train[feats].values.astype(float)
    y_tr = train["is_attack"].values.astype(int)
    X_val = val[feats].values.astype(float)
    y_val = val["is_attack"].values.astype(int)
    X_te = test[feats].values.astype(float)
    y_te = test["is_attack"].values.astype(int)

    print(f"[ablation] train={len(y_tr):,}  val={len(y_val):,}  test={len(y_te):,}")

    print("[ablation] retraining RF without Flow IAT Min ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=24, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RNG, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    print(f"[ablation] retrain took {time.time() - t0:.1f}s")

    p_val = rf.predict_proba(X_val)[:, 1]
    fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val)
    j = tpr_v - fpr_v
    theta = float(thr_v[int(np.argmax(j))])
    print(f"[ablation] Youden-J threshold: {theta:.6f}")

    p_te = rf.predict_proba(X_te)[:, 1]
    pred_te = (p_te >= theta).astype(int)
    f1_clean = f1_score(y_te, pred_te)
    auc_clean = roc_auc_score(y_te, p_te)
    prec_clean = precision_score(y_te, pred_te)
    rec_clean = recall_score(y_te, pred_te)
    print(f"[ablation] clean F1={f1_clean:.4f}  AUC={auc_clean:.4f}  "
          f"P={prec_clean:.4f}  R={rec_clean:.4f}")

    perturbable = [c for c in PERTURBABLE if c in feats]
    feat_idx = np.array([feats.index(c) for c in perturbable])
    print(f"[ablation] perturbable features: {len(perturbable)} of {len(feats)}")

    benign_mask = y_tr == 0
    scale = X_tr[benign_mask].std(axis=0)
    scale[scale == 0] = 1.0
    lower = np.percentile(X_tr, 0.5, axis=0)
    upper = np.percentile(X_tr, 99.5, axis=0)

    attack_mask = (y_te == 1) & (pred_te == 1)
    attack_idx = np.where(attack_mask)[0]
    rng = np.random.default_rng(RNG)
    sample = rng.choice(attack_idx, size=min(N_ATTACK_SAMPLES, len(attack_idx)), replace=False)
    X_attack = X_te[sample]
    print(f"[ablation] running greedy attack on {len(X_attack)} attack flows ...")
    r = evaluate_adversarial(X_attack, rf, theta, feat_idx, scale, lower, upper, "RF-ablated")

    evasion_rate = float(r["flipped"].mean())
    median_linf = float(np.median(r["linf_sigma"][r["flipped"]])) if r["flipped"].any() else None

    n_flipped = int(r["flipped"].sum())
    if n_flipped > 0:
        deltas = np.abs(r["adv"][r["flipped"]] - X_attack[r["flipped"]])
        moved = (deltas > 1e-9).sum(axis=0) / n_flipped
        top = sorted(enumerate(moved), key=lambda x: -x[1])[:10]
        top_feats = [(feats[i], float(p * 100)) for i, p in top if p > 0]
    else:
        top_feats = []

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Flow IAT Min Ablation\n\n")
    lines.append("Single-feature ablation: drop `Flow IAT Min` from training, "
                 "retrain the binary day-split random forest, retune the threshold, "
                 "and re-run the score-query black-box greedy attack.\n\n")
    lines.append("## Setup\n\n")
    lines.append(f"- Features after drop: {len(feats)} (dropped `Flow IAT Min`)\n")
    lines.append(f"- Perturbable features for attack: {len(perturbable)} of {len(feats)}\n")
    lines.append(f"- Threshold (Youden's J on val): {theta:.6f}\n")
    lines.append("- random_state: 42\n\n")
    lines.append("## Clean performance (binary day-split test)\n\n")
    lines.append("| Metric | Original RF | Ablated RF |\n")
    lines.append("|---|---:|---:|\n")
    lines.append(f"| F1 (Youden) | 0.8586 | {f1_clean:.4f} |\n")
    lines.append(f"| ROC-AUC | 0.9396 | {auc_clean:.4f} |\n")
    lines.append(f"| Precision | 0.8509 | {prec_clean:.4f} |\n")
    lines.append(f"| Recall | 0.8665 | {rec_clean:.4f} |\n\n")
    lines.append("## Adversarial robustness (greedy black-box attack)\n\n")
    lines.append("| Quantity | Original RF | Ablated RF |\n")
    lines.append("|---|---:|---:|\n")
    lines.append(f"| Unbounded evasion rate | 0.2440 | {evasion_rate:.4f} |\n")
    median_str = f"{median_linf:.3f}" if median_linf is not None else "n/a"
    lines.append(f"| Median L-inf to flip (sigma) | 0.250 | {median_str} |\n\n")
    lines.append("## Most-moved features in successful evasions\n\n")
    if top_feats:
        lines.append("| Feature | % of successful evasions moving this feature |\n")
        lines.append("|---|---:|\n")
        for name, pct in top_feats:
            lines.append(f"| {name} | {pct:.1f} |\n")
    else:
        lines.append("(No successful evasions in the sample.)\n")
    lines.append("\n## Interpretation\n\n")
    if evasion_rate < 0.18:
        lines.append("Evasion rate dropped substantially. The original detector "
                     "was relying on Flow IAT Min as a brittle shortcut, and the "
                     "ablated model is more robust to score-query black-box attacks.\n")
    elif evasion_rate > 0.28:
        lines.append("Evasion rate rose. Flow IAT Min was a robust signal; "
                     "removing it shifted the attack onto less stable features.\n")
    else:
        lines.append("Evasion rate is roughly unchanged. The attack simply "
                     "re-concentrated on the next most discriminative timing feature; "
                     "this is consistent with a genuine timing-based attack surface "
                     "rather than a single-feature shortcut.\n")
    OUT_MD.write_text("".join(lines))
    print(f"[ablation] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
