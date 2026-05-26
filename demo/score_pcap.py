from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models/binary_day/random_forest.joblib"
FEATS_FILE = ROOT / "models/binary_day/feature_names.json"
THETA = 0.009


COL_MAP = {
    "Protocol": "protocol",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Fwd Packets Length Total": "totlen_fwd_pkts",
    "Bwd Packets Length Total": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Packet Length Min": "pkt_len_min",
    "Packet Length Max": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Avg Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init Fwd Win Bytes": "init_fwd_win_byts",
    "Init Bwd Win Bytes": "init_bwd_win_byts",
    "Fwd Act Data Packets": "fwd_act_data_pkts",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


def main(csv_path: str) -> None:
    if not MODEL_FILE.exists():
        raise SystemExit(f"Missing {MODEL_FILE}. Run `make full` first.")
    model = joblib.load(MODEL_FILE)
    feats = json.loads(FEATS_FILE.read_text())

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    X_cols = {}
    missing = []
    for feat in feats:
        src = COL_MAP.get(feat)
        if src and src in df.columns:
            X_cols[feat] = pd.to_numeric(df[src], errors="coerce").fillna(0)
        else:
            missing.append(feat)
            X_cols[feat] = 0.0
    X_df = pd.DataFrame(X_cols)

    if missing:
        print(f"WARNING: {len(missing)} of {len(feats)} features unmappable, "
              f"filled with 0:")
        for c in missing[:5]:
            print(f"  - {c} (no mapping or column missing)")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    X = X_df.values
    p = model.predict_proba(X)[:, 1]
    alerts = int((p >= THETA).sum())
    n = len(p)

    print(f"\nFlows scored : {n}")
    print(f"Alerts raised: {alerts}  (threshold p >= {THETA})")
    print(f"Attack rate  : {alerts / max(n, 1) * 100:.1f}%")

    print("\nTop 10 flows by p(attack):")
    out = pd.DataFrame({"p_attack": p.round(4)})
    for col in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol"):
        if col in df.columns:
            out[col] = df[col].values
    print(out.sort_values("p_attack", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python score_pcap.py <flows.csv>")
    main(sys.argv[1])
