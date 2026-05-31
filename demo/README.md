# SOC Live Demo

A Flask web dashboard that scores the Friday test flows through the deployed binary day-split Random Forest and emits MITRE-tagged alerts in real time. Bundled with a Wireshark live-capture bridge that runs the same model against real packets captured from the host network interface.

## What the dashboard shows

- Rolling alerts table, newest at the top, columns: time, p(attack), predicted class, true class, ATT&CK ID, kill-chain phase, severity
- Green rows: true positives (the model correctly flagged an attack)
- Red rows: false positives (benign flows above the 0.009 threshold)
- Right panel: total flows scored, alerts raised, attack rate, precision, uptime, breakdown by kill-chain phase, breakdown by severity
- Pause and Reset controls

Alerts are emitted when the model's predicted probability exceeds the Youden's J threshold of 0.009. MITRE technique IDs and kill-chain phases come from a built-in table covering all 14 CICIDS2017 attack classes (Brute Force, DoS, DDoS, PortScan, Bot, Heartbleed, Web Attacks). Severity is derived from the kill-chain phase: Impact and Initial Access are CRITICAL, Credential Access and Command and Control are HIGH, Discovery and Execution are MEDIUM, false positives are LOW.

## Quick start

```bash
cd path/to/bsc-thesis-ids
source .venv/bin/activate
cd demo
python app.py
```

Open `http://127.0.0.1:5050` in a browser. The feed runs at roughly 15 flows per tick, one tick per 0.4 seconds, so alerts populate within 5 seconds.

## Live packet capture (optional)

A one-shot wrapper captures real packets from the laptop network interface, converts them to the 73 CICIDS2017 flow features, and scores them with the same Random Forest model. Useful for demonstrating that the model trained on 2017 data still alerts on traffic from today.

```bash
DEFENCE_SCAN=1 ./live_capture.sh         # 20 s capture, plus nmap against the default gateway
./live_capture.sh 30 en0                 # custom duration and interface
```

Setup steps for tshark, nmap, and the hieulw CICFlowMeter fork are documented in `WIRESHARK.md`.

## Why this matters for the defence

Three things examiners can take away:

1. The model produces concrete, structured alerts: probability, predicted class, ATT&CK technique, phase, severity. This is what a SOC pipeline emits.
2. The MITRE mapping turns numbers into playbook tags. T1110 routes to credential-stuffing playbooks, T1498 to DDoS playbooks, T1046 to scan-detection playbooks.
3. False positives are visible in red alongside true positives in green, so the analyst sees the precision trade-off in real time. The 86 to 87 percent precision figure in the right-hand panel matches the day-split test set numbers in the thesis.

If demonstrating the adversarial finding, pause the feed at any attack alert, point to the `p(attack)` value, and explain that adding a 0.16 millisecond inter-packet delay would push Flow IAT Min across the boundary and drop the score below 0.009. That is the 20.8 percent bounded evasion rate from Chapter 5.

## Files

```
demo/
├── app.py             # Flask server, MITRE lookup, scoring loop
├── score_pcap.py      # Score a CICFlowMeter CSV with the deployed RF
├── live_capture.sh    # Wrapper: tshark + cicflowmeter + score_pcap.py
├── templates/
│   └── dashboard.html # Browser UI
├── README.md          # This file
└── WIRESHARK.md       # Live capture playbook (Wireshark, nmap, cicflowmeter)
```
