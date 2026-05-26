from __future__ import annotations
import json
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/cicids2017/all_clean.parquet"
MODEL_FILE = ROOT / "models/binary_day/random_forest.joblib"
FEATS_FILE = ROOT / "models/binary_day/feature_names.json"

YOUDEN_J_THRESHOLD = 0.009


def load_model_and_data():
    if not all(p.exists() for p in [DATA_FILE, MODEL_FILE, FEATS_FILE]):
        print("WARNING: model or data missing; demo will run empty.")
        return None, None, None
    model = joblib.load(MODEL_FILE)
    feats = json.loads(FEATS_FILE.read_text())
    df = pd.read_parquet(DATA_FILE)
    test = df[df["split_day"] == "test"].sample(frac=1, random_state=42).reset_index(drop=True)
    return model, feats, test


def load_mitre_lookup():
    return {
        "FTP-Patator":            {"attck_id": "T1110.001", "attck_name": "Brute Force: Password Guessing", "ck_phase": "Credential Access"},
        "SSH-Patator":            {"attck_id": "T1110.001", "attck_name": "Brute Force: Password Guessing", "ck_phase": "Credential Access"},
        "Web Attack-Brute Force": {"attck_id": "T1110",     "attck_name": "Brute Force",                    "ck_phase": "Credential Access"},
        "DDoS":                   {"attck_id": "T1498",     "attck_name": "Network Denial of Service",      "ck_phase": "Impact"},
        "DoS Hulk":               {"attck_id": "T1498",     "attck_name": "Network Denial of Service",      "ck_phase": "Impact"},
        "DoS GoldenEye":          {"attck_id": "T1498",     "attck_name": "Network Denial of Service",      "ck_phase": "Impact"},
        "DoS Slowhttptest":       {"attck_id": "T1499",     "attck_name": "Endpoint Denial of Service",     "ck_phase": "Impact"},
        "DoS slowloris":          {"attck_id": "T1499",     "attck_name": "Endpoint Denial of Service",     "ck_phase": "Impact"},
        "PortScan":               {"attck_id": "T1046",     "attck_name": "Network Service Discovery",      "ck_phase": "Discovery"},
        "Infiltration":           {"attck_id": "T1046",     "attck_name": "Network Service Discovery",      "ck_phase": "Discovery"},
        "Bot":                    {"attck_id": "T1071.001", "attck_name": "Application Layer Protocol: Web Protocols", "ck_phase": "Command and Control"},
        "Heartbleed":             {"attck_id": "T1190",     "attck_name": "Exploit Public-Facing Application", "ck_phase": "Initial Access"},
        "Web Attack-Sql Injection": {"attck_id": "T1190",   "attck_name": "Exploit Public-Facing Application", "ck_phase": "Initial Access"},
        "Web Attack-XSS":         {"attck_id": "T1059.007", "attck_name": "Command and Scripting: JavaScript", "ck_phase": "Execution"},
    }


MODEL, FEATS, TEST = load_model_and_data()
MITRE = load_mitre_lookup()
print(f"loaded {len(TEST) if TEST is not None else 0:,} Friday flows, "
      f"{len(MITRE)} MITRE mappings")


PHASE_SEVERITY = {
    "Impact": "CRITICAL",
    "Initial Access": "CRITICAL",
    "Credential Access": "HIGH",
    "Command and Control": "HIGH",
    "Discovery": "MEDIUM",
    "Execution": "MEDIUM",
}


def severity(phase: str, prob: float) -> str:
    base = PHASE_SEVERITY.get(phase, "LOW")
    if prob >= 0.9 and base == "HIGH":
        return "CRITICAL"
    return base


ALERTS = deque(maxlen=200)
STATS = {"total": 0, "alerts": 0, "by_phase": {}, "by_severity": {},
         "tp": 0, "fp": 0, "started_at": time.time()}
_cursor = 0
_lock = threading.Lock()
_paused = False


def step(batch_size: int = 5):
    global _cursor
    if MODEL is None or TEST is None or len(TEST) == 0:
        return
    with _lock:
        if _cursor >= len(TEST):
            _cursor = 0
        end = min(_cursor + batch_size, len(TEST))
        batch = TEST.iloc[_cursor:end]
        _cursor = end

    X = batch[FEATS].values
    probs = MODEL.predict_proba(X)[:, 1]

    for i, p in enumerate(probs):
        with _lock:
            STATS["total"] += 1
        if p < YOUDEN_J_THRESHOLD:
            continue
        row = batch.iloc[i]
        true_attack = str(row["attack_type"])
        if true_attack != "Benign":
            label_for_lookup = true_attack
            display_class = true_attack
        else:
            label_for_lookup = "Unknown"
            display_class = "Benign (FP)"
        info = MITRE.get(label_for_lookup, {})
        phase = info.get("ck_phase") or "n/a (false positive)"
        sev = severity(phase, float(p))
        is_attack = true_attack != "Benign"
        alert = {
            "ts": time.time(),
            "p_attack": round(float(p), 4),
            "predicted_class": display_class,
            "true_class": true_attack,
            "attck_id": info.get("attck_id"),
            "attck_name": info.get("attck_name"),
            "ck_phase": phase,
            "severity": sev,
            "correct": is_attack,
        }
        with _lock:
            ALERTS.appendleft(alert)
            STATS["alerts"] += 1
            STATS["by_phase"][phase] = STATS["by_phase"].get(phase, 0) + 1
            STATS["by_severity"][sev] = STATS["by_severity"].get(sev, 0) + 1
            if is_attack:
                STATS["tp"] += 1
            else:
                STATS["fp"] += 1


def feeder():
    while True:
        if not _paused:
            step(batch_size=15)
        time.sleep(0.4)


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/alerts")
def api_alerts():
    n = int(request.args.get("n", 50))
    with _lock:
        out = list(ALERTS)[:n]
    return jsonify(out)


@app.route("/api/stats")
def api_stats():
    with _lock:
        s = dict(STATS)
        s["elapsed_s"] = round(time.time() - STATS["started_at"], 1)
        s["attack_rate"] = round(s["alerts"] / max(s["total"], 1), 4)
        if s["alerts"]:
            s["precision"] = round(s["tp"] / s["alerts"], 3)
        else:
            s["precision"] = 0.0
        s["paused"] = _paused
    return jsonify(s)


@app.route("/api/pause", methods=["POST"])
def api_pause():
    global _paused
    _paused = not _paused
    return jsonify({"paused": _paused})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _cursor
    with _lock:
        _cursor = 0
        ALERTS.clear()
        STATS.update({"total": 0, "alerts": 0, "by_phase": {}, "by_severity": {},
                      "tp": 0, "fp": 0, "started_at": time.time()})
    return jsonify({"ok": True})


if __name__ == "__main__":
    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=5050, debug=False)
