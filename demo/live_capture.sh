#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-20}"
IFACE="${2:-en0}"
WORK_DIR="${TMPDIR:-/tmp}"
PCAP="${WORK_DIR}/live_scan.pcap"
CSV="${WORK_DIR}/live_flows.csv"

TSHARK="/Applications/Wireshark.app/Contents/MacOS/tshark"
if [[ ! -x "$TSHARK" ]]; then
    if command -v tshark > /dev/null 2>&1; then
        TSHARK="$(command -v tshark)"
    else
        echo "tshark not found. Install Wireshark: brew install --cask wireshark"
        exit 1
    fi
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
CIC="${ROOT}/.venv/bin/cicflowmeter"
if [[ ! -x "$PY" || ! -x "$CIC" ]]; then
    echo "Activate venv and run: pip install git+https://github.com/hieulw/cicflowmeter.git"
    exit 1
fi

echo "[1/4] capturing ${DURATION}s on ${IFACE} -> ${PCAP}"
rm -f "$PCAP"
"$TSHARK" -i "$IFACE" -a duration:"$DURATION" -w "$PCAP" > /tmp/tshark.log 2>&1 &
TSHARK_PID=$!

if [[ "${DEFENCE_SCAN:-0}" == "1" ]]; then
    GW="$(route -n get default 2>/dev/null | awk '/gateway/{print $2}')"
    if [[ -n "$GW" ]] && command -v nmap > /dev/null 2>&1; then
        sleep 2
        echo "[1b ] nmap -sT -p 1-500 -T4 $GW (during capture)"
        nmap -sT -p 1-500 --max-retries 0 -T4 "$GW" > /tmp/nmap.log 2>&1 &
    fi
fi

wait "$TSHARK_PID"
echo "[2/4] captured: $(tail -1 /tmp/tshark.log)"

echo "[3/4] cicflowmeter $PCAP -> $CSV"
"$CIC" -f "$PCAP" -c "$CSV" 2>&1 | tail -2
echo "      $(wc -l < "$CSV") lines (header + flows)"

echo "[4/4] scoring with binary day-split RF (theta=0.009):"
"$PY" "$(dirname "$0")/score_pcap.py" "$CSV"
