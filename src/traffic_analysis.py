"""Blue team traffic analysis: per-attack flow stats, detection rules, MITRE mapping."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.config import DATA_FILE, REPORTS_DIR, RNG

OUT_DIR = REPORTS_DIR / "traffic_analysis"
FIG_DIR = OUT_DIR / "figures"

# MITRE ATT&CK mapping per attack type
MITRE = {
    "Benign": {
        "tactic": None, "technique_id": None, "technique": None,
        "kill_chain": None,
        "soc_action": "No action required.",
        "detection_note": "Baseline normal traffic.",
    },
    "FTP-Patator": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "kill_chain": "Credential Access",
        "soc_action": (
            "1. Identify source IP and block at perimeter firewall.\n"
            "2. Check FTP server logs for successful logins from same IP.\n"
            "3. Reset credentials on targeted FTP accounts.\n"
            "4. Alert on: >20 FTP connections/min from single source."
        ),
        "detection_note": (
            "FTP brute force generates many short flows to port 21. "
            "Key signals: high Flow Count, low Fwd Packet Length Mean (~5 bytes per attempt), "
            "very short Flow Duration, high Bwd Packets/Flow ratio on failures."
        ),
    },
    "SSH-Patator": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "kill_chain": "Credential Access",
        "soc_action": (
            "1. Block source IP at firewall after threshold (e.g. 10 failed attempts).\n"
            "2. Enable SSH key-only authentication — disable password auth.\n"
            "3. Check for successful SSH sessions from same source.\n"
            "4. Alert on: >10 SSH connections/min from single source IP."
        ),
        "detection_note": (
            "SSH brute force looks similar to FTP but on port 22. "
            "Many short TCP flows, consistently small packet sizes in both directions, "
            "high IAT (inter-arrival time) variance as attacker loops through passwords."
        ),
    },
    "DoS GoldenEye": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique": "Network Denial of Service",
        "kill_chain": "Impact",
        "soc_action": (
            "1. Rate-limit HTTP Keep-Alive connections at load balancer.\n"
            "2. Deploy connection timeout rules (close idle connections after 30s).\n"
            "3. Identify source IPs and null-route at upstream provider.\n"
            "4. Alert on: single IP holding >50 concurrent HTTP connections."
        ),
        "detection_note": (
            "GoldenEye exploits HTTP Keep-Alive to hold connections open. "
            "Flow Duration is very long, Fwd Packet Length is small but persistent. "
            "Low packets-per-second but constant connection count growth."
        ),
    },
    "DoS Hulk": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique": "Network Denial of Service",
        "kill_chain": "Impact",
        "soc_action": (
            "1. Enable HTTP rate limiting at WAF — block >100 requests/sec/IP.\n"
            "2. Activate DDoS scrubbing if available (cloud provider or CDN).\n"
            "3. Check server CPU and memory — Hulk exhausts application threads.\n"
            "4. Alert on: >500 unique HTTP GET requests/sec to same endpoint."
        ),
        "detection_note": (
            "Hulk generates randomised HTTP GET floods. Very high Fwd Packet count, "
            "high total bytes, very short Flow Duration per connection. "
            "Unique feature: randomised User-Agent strings (not visible in flow data "
            "but implied by the volume pattern)."
        ),
    },
    "DoS Slowhttptest": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service: Application Exhaustion",
        "kill_chain": "Impact",
        "soc_action": (
            "1. Set minimum HTTP request rate — drop connections sending <1 byte/10s.\n"
            "2. Limit max concurrent connections per IP at web server level.\n"
            "3. Deploy mod_reqtimeout (Apache) or equivalent.\n"
            "4. Alert on: connections with Flow Duration >60s and <100 bytes transferred."
        ),
        "detection_note": (
            "Slowhttptest sends partial HTTP requests very slowly to exhaust the "
            "server connection pool. Extremely long Flow Duration, very low byte count, "
            "near-zero packet rate. Distinct from normal slow clients by volume."
        ),
    },
    "DoS slowloris": {
        "tactic": "Impact",
        "technique_id": "T1499",
        "technique": "Endpoint Denial of Service: Application Exhaustion",
        "kill_chain": "Impact",
        "soc_action": (
            "1. Same as SlowHTTPTest — deploy connection timeout rules.\n"
            "2. Use nginx instead of Apache where possible (more resistant).\n"
            "3. Alert on: connections open >30s with <50 bytes sent."
        ),
        "detection_note": (
            "Slowloris holds connections open with partial HTTP headers. "
            "Key flow feature: very long Duration, tiny Fwd Packet Length, "
            "near-zero Bwd Packet count (server waiting, never responding fully)."
        ),
    },
    "Heartbleed": {
        "tactic": "Initial Access",
        "technique_id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "kill_chain": "Initial Access",
        "soc_action": (
            "1. IMMEDIATE: Patch OpenSSL to >=1.0.1g on all affected servers.\n"
            "2. Revoke and reissue ALL TLS certificates on affected hosts.\n"
            "3. Force password resets for all users — session tokens may be leaked.\n"
            "4. Check server memory dumps for leaked private key material.\n"
            "5. Alert on: malformed TLS heartbeat requests (unusual payload length)."
        ),
        "detection_note": (
            "Heartbleed sends malformed TLS heartbeat extension requests. "
            "Flow is very short, small Fwd Packet (the exploit request), "
            "large Bwd Packet (server leaking up to 64KB of memory). "
            "Asymmetric packet size ratio is the key signature."
        ),
    },
    "Web Attack-Brute Force": {
        "tactic": "Credential Access",
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "kill_chain": "Credential Access",
        "soc_action": (
            "1. Enable account lockout after 5 failed login attempts.\n"
            "2. Deploy CAPTCHA on login forms.\n"
            "3. Block source IP at WAF after threshold.\n"
            "4. Alert on: >10 POST requests to /login endpoint from same IP in 60s."
        ),
        "detection_note": (
            "Web login brute force generates repeated HTTP POST flows to the same endpoint. "
            "Moderate Flow Duration, consistent Fwd Packet Length (form submission size), "
            "small Bwd Packet on failures (login error page)."
        ),
    },
    "Web Attack-Sql Injection": {
        "tactic": "Initial Access",
        "technique_id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "kill_chain": "Initial Access",
        "soc_action": (
            "1. Review WAF logs for SQL metacharacter patterns (', --, UNION SELECT).\n"
            "2. Check database query logs for unusual SELECT statements.\n"
            "3. Identify which endpoint was targeted and audit its input validation.\n"
            "4. Alert on: HTTP requests containing SQL keywords in URL or POST body."
        ),
        "detection_note": (
            "SQLi flows look like normal HTTP but with slightly larger Fwd Packet "
            "(injected payload in the request). Response size (Bwd) varies by success. "
            "Flow-level detection is hard — WAF or application-layer IDS needed for precision."
        ),
    },
    "Web Attack-XSS": {
        "tactic": "Execution",
        "technique_id": "T1059.007",
        "technique": "Command and Scripting Interpreter: JavaScript",
        "kill_chain": "Execution",
        "soc_action": (
            "1. Implement Content-Security-Policy headers on all web pages.\n"
            "2. Audit application for missing output encoding.\n"
            "3. Check for injected <script> tags in stored content (database).\n"
            "4. Alert on: HTTP requests containing <script> or javascript: in parameters."
        ),
        "detection_note": (
            "XSS is one of the hardest to detect at flow level — the payload is "
            "embedded in HTTP content. Flow statistics are nearly identical to benign. "
            "ML detection relies on subtle Fwd/Bwd packet length asymmetry."
        ),
    },
    "Infiltration": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique": "Network Service Discovery",
        "kill_chain": "Discovery",
        "soc_action": (
            "1. Identify internal host performing the scan and isolate it.\n"
            "2. Check for malware on the scanning host — infiltration implies compromise.\n"
            "3. Review firewall logs for lateral movement attempts.\n"
            "4. Alert on: single internal host connecting to >10 distinct ports in 60s."
        ),
        "detection_note": (
            "Infiltration traffic mimics legitimate internal scanning but with "
            "characteristic port sweep patterns. Many short flows to different ports, "
            "low byte count, high destination port variance."
        ),
    },
    "Bot": {
        "tactic": "Command and Control",
        "technique_id": "T1071.001",
        "technique": "Application Layer Protocol: Web Protocols",
        "kill_chain": "Command and Control",
        "soc_action": (
            "1. Identify C2 domain via DNS logs — block domain and IP at firewall.\n"
            "2. Isolate the infected host from the network immediately.\n"
            "3. Perform forensic analysis — check scheduled tasks, registry run keys.\n"
            "4. Alert on: periodic outbound connections to same external IP at fixed intervals."
        ),
        "detection_note": (
            "Botnet C2 beaconing is characterised by periodic, regular flows. "
            "Key flow feature: consistent Flow Duration, regular Inter-Arrival Time, "
            "small but steady packet exchange (heartbeat pattern). "
            "Beaconing interval is often fixed (e.g. every 60s)."
        ),
    },
    "DDoS": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique": "Network Denial of Service",
        "kill_chain": "Impact",
        "soc_action": (
            "1. Activate upstream DDoS mitigation (contact ISP or CDN provider).\n"
            "2. Enable blackhole routing for targeted IP if service can be moved.\n"
            "3. Identify attack vector (UDP flood, HTTP flood, amplification).\n"
            "4. Alert on: inbound traffic >10x baseline volume from distributed sources."
        ),
        "detection_note": (
            "DDoS generates massive traffic volume from many sources. "
            "Very high Fwd Packet count, high total bytes, short Flow Duration per source. "
            "Aggregate flow count to destination IP is the primary indicator."
        ),
    },
    "PortScan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique": "Network Service Discovery",
        "kill_chain": "Discovery",
        "soc_action": (
            "1. Block source IP at perimeter — port scanning precedes attacks.\n"
            "2. Log the scan for threat intelligence — record source, timing, ports.\n"
            "3. Review which services were discovered — harden exposed ports.\n"
            "4. Alert on: >100 distinct destination ports from single source in 60s."
        ),
        "detection_note": (
            "Port scanning produces many very short TCP flows (SYN only or SYN-RST). "
            "Key features: very low Flow Duration, low Fwd Packet Length, "
            "high destination port variance, near-zero Bwd Packet count on closed ports."
        ),
    },
}

# Key flow features for analysis
KEY_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd Packet Length Max",
    "Bwd Packet Length Max",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# Tactic colours for plots
TACTIC_COLORS = {
    "Reconnaissance":      "#4e9af1",
    "Discovery":           "#52b788",
    "Credential Access":   "#e76f51",
    "Initial Access":      "#e63946",
    "Execution":           "#ae2012",
    "Impact":              "#6d0022",
    "Command and Control": "#2d6a4f",
    None:                  "#adb5bd",
}


# Helpers

def _load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}. Run: python -m src.prepare_data first.")
    print(f"[traffic_analysis] loading {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)
    print(f"[traffic_analysis] {len(df):,} rows, {len(df.columns)} columns")
    return df


def _available_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Filter to features present in df."""
    return [f for f in wanted if f in df.columns]


def _get_mitre(attack: str) -> dict:
    """Look up MITRE entry with fuzzy fallback."""
    if attack in MITRE:
        return MITRE[attack]
    # fuzzy fallback
    al = attack.lower()
    for k, v in MITRE.items():
        if k.lower() == al:
            return v
    return {"tactic": "Unknown", "technique_id": "—", "technique": "—",
            "kill_chain": "—", "soc_action": "Review manually.", "detection_note": "—"}


# Section 1: Flow statistics per attack type

def section_flow_statistics(df: pd.DataFrame, feat: list[str]) -> str:
    """Per-attack flow-level stats."""
    lines = [
        "# 1. Flow-Level Traffic Statistics per Attack Type\n\n",
        "Per-attack flow statistics — what each attack looks like at the network level.\n\n",
    ]

    attacks = sorted(df["attack_type"].unique())
    stat_cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                 "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                 "Packet Length Mean", "Flow IAT Mean"]
    stat_cols = _available_features(df, stat_cols)

    for attack in attacks:
        sub = df[df["attack_type"] == attack]
        m = _get_mitre(attack)
        tactic = m.get("tactic") or "—"
        tid = m.get("technique_id") or "—"
        technique = m.get("technique") or "—"

        lines.append(f"## {attack}\n\n")
        lines.append(f"- **ATT&CK tactic**: {tactic}\n")
        lines.append(f"- **Technique**: {tid} — {technique}\n")
        lines.append(f"- **Flow count**: {len(sub):,} ({len(sub)/len(df)*100:.2f}% of dataset)\n\n")

        if stat_cols:
            lines.append("| Feature | Mean | Median | Std | Min | Max |\n")
            lines.append("|---------|------|--------|-----|-----|-----|\n")
            for col in stat_cols:
                if col not in sub.columns:
                    continue
                s = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) == 0:
                    continue
                lines.append(
                    f"| {col} | {s.mean():.2f} | {s.median():.2f} | "
                    f"{s.std():.2f} | {s.min():.2f} | {s.max():.2f} |\n"
                )
        lines.append("\n")

        # Security interpretation
        note = m.get("detection_note", "")
        if note:
            lines.append(f"**Blue team interpretation**: {note}\n\n")

        lines.append("---\n\n")

    return "".join(lines)


# Section 2: Detection signatures (Sigma-style logic)

def section_detection_signatures(df: pd.DataFrame, feat: list[str]) -> str:
    """Sigma-style detection rules derived from flow statistics."""
    lines = [
        "# 2. Detection Signatures — Flow-Based Detection Logic\n\n",
        "Detection rules derived from flow statistics (Sigma-style). "
        "Real deployment needs network-specific threshold tuning.\n\n",
    ]

    SIGMA_RULES = {
        "FTP-Patator": {
            "title": "FTP Brute Force via High Connection Rate",
            "description": "Detects FTP password guessing based on abnormally high flow count with short duration.",
            "condition": [
                "destination_port = 21",
                "Flow Duration < 1000ms (very short authentication attempt)",
                "Total Fwd Packets < 5 (minimal handshake only)",
                "Alert if: same source IP generates >20 such flows per minute",
            ],
            "false_positives": ["Legitimate FTP backup jobs", "Batch FTP transfers"],
            "severity": "HIGH",
        },
        "SSH-Patator": {
            "title": "SSH Brute Force via Repetitive Short Flows",
            "description": "Detects SSH password guessing by volume of failed short-duration SSH flows.",
            "condition": [
                "destination_port = 22",
                "Flow Duration < 5000ms",
                "Fwd Packet Length Mean < 200 bytes",
                "Alert if: >10 SSH flows from same source in 60 seconds",
            ],
            "false_positives": ["Automated SSH health checks", "Misconfigured SSH clients"],
            "severity": "HIGH",
        },
        "DoS GoldenEye": {
            "title": "HTTP Keep-Alive Connection Exhaustion (GoldenEye)",
            "description": "Detects GoldenEye DoS by identifying long-lived HTTP connections with low throughput.",
            "condition": [
                "destination_port = 80 or 443",
                "Flow Duration > 30000ms (long-lived)",
                "Fwd Packet Length Mean < 500 bytes (small requests)",
                "Bwd Packet count ≈ 0 (server not responding or waiting)",
                "Alert if: >50 such connections from same source simultaneously",
            ],
            "false_positives": ["Slow legitimate clients on poor connections"],
            "severity": "HIGH",
        },
        "DoS Hulk": {
            "title": "HTTP Flood (Hulk)",
            "description": "Detects Hulk HTTP flood by extremely high request rate to web server.",
            "condition": [
                "destination_port = 80 or 443",
                "Total Fwd Packets > 100 per flow",
                "Flow Duration < 10000ms (high-speed flood)",
                "Alert if: total inbound HTTP flows to single destination >500/second",
            ],
            "false_positives": ["Legitimate traffic spikes during flash sales", "CDN origin pulls"],
            "severity": "CRITICAL",
        },
        "DoS Slowhttptest": {
            "title": "Slow HTTP Body Attack (SlowHTTPTest)",
            "description": "Detects slow HTTP body attacks by very long flows with near-zero throughput.",
            "condition": [
                "destination_port = 80 or 443",
                "Flow Duration > 60000ms",
                "Total bytes transferred < 500 bytes in total",
                "Fwd Packet Length Std ≈ 0 (uniform tiny packets)",
                "Alert if: >20 such flows to same server simultaneously",
            ],
            "false_positives": ["Legitimate slow uploads on poor connections"],
            "severity": "MEDIUM",
        },
        "DoS slowloris": {
            "title": "Slowloris Connection Hold Attack",
            "description": "Detects Slowloris by connections that stay open with minimal data transfer.",
            "condition": [
                "destination_port = 80 or 443",
                "Flow Duration > 30000ms",
                "Fwd Packet count < 10 in entire duration",
                "Bwd Packet count = 0 (server holds connection open waiting)",
                "Alert if: single IP holds >30 such connections open",
            ],
            "false_positives": ["Very slow legitimate clients"],
            "severity": "MEDIUM",
        },
        "Heartbleed": {
            "title": "OpenSSL Heartbleed Exploitation Attempt",
            "description": "Detects Heartbleed by asymmetric TLS flow: small request, large response.",
            "condition": [
                "destination_port = 443 (HTTPS/TLS)",
                "Fwd Packet Length Mean < 100 bytes (small heartbeat request)",
                "Bwd Packet Length Max > 10000 bytes (server memory leak in response)",
                "Bwd/Fwd byte ratio > 100x (extreme asymmetry)",
                "Alert immediately — zero tolerance for this pattern",
            ],
            "false_positives": ["None expected — this pattern is highly specific"],
            "severity": "CRITICAL",
        },
        "Web Attack-Brute Force": {
            "title": "Web Login Brute Force",
            "description": "Detects web authentication brute force by repeated POST requests.",
            "condition": [
                "destination_port = 80 or 443",
                "HTTP method = POST (not visible in flow, but implied by packet size)",
                "Fwd Packet Length Mean 200-800 bytes (form submission)",
                "Bwd Packet Length Mean < 500 bytes (login failure response)",
                "Alert if: >10 identical-size POST flows to same URL from same IP in 60s",
            ],
            "false_positives": ["Password managers testing saved credentials"],
            "severity": "HIGH",
        },
        "Web Attack-Sql Injection": {
            "title": "SQL Injection Attempt",
            "description": "Detects SQLi by slightly enlarged HTTP requests containing injection payload.",
            "condition": [
                "destination_port = 80 or 443",
                "Fwd Packet Length slightly > normal GET (extra payload bytes)",
                "HTTP parameters contain: ', --, UNION, SELECT, DROP, INSERT",
                "Recommend WAF rule: block requests matching SQL metacharacter patterns",
            ],
            "false_positives": ["Developers testing forms", "Security scanners (Burp Suite)"],
            "severity": "HIGH",
        },
        "Web Attack-XSS": {
            "title": "Cross-Site Scripting (XSS) Attempt",
            "description": "Detects XSS by HTTP requests containing script injection payloads.",
            "condition": [
                "destination_port = 80 or 443",
                "HTTP parameters contain: <script>, javascript:, onerror=, onload=",
                "Flow-level detection has low precision — WAF required",
                "Recommend: Content-Security-Policy header + output encoding audit",
            ],
            "false_positives": ["Security researchers", "Input validation test suites"],
            "severity": "HIGH",
        },
        "Infiltration": {
            "title": "Internal Network Service Discovery (Post-Compromise Scan)",
            "description": "Detects internal host performing port sweep — indicator of compromise.",
            "condition": [
                "Source IP is internal (RFC1918 range)",
                "Many distinct destination ports contacted in short time",
                "Flow Duration < 500ms per port (no real service interaction)",
                "Alert if: internal host contacts >10 distinct ports on >3 hosts in 60s",
            ],
            "false_positives": ["IT asset discovery tools (Nmap scheduled scans)", "Vulnerability scanners"],
            "severity": "CRITICAL",
        },
        "Bot": {
            "title": "Botnet C2 Beaconing",
            "description": "Detects bot C2 by periodic, regular outbound connections (beaconing).",
            "condition": [
                "Outbound connection to external IP at regular intervals (e.g. every 60s)",
                "Flow Duration consistent across all beacons (±10%)",
                "Flow IAT Std very low (regular timing = automated, not human)",
                "Small payload size (C2 heartbeat, not bulk data transfer)",
                "Alert if: same (src_ip, dst_ip, dst_port) tuple seen >5 times with regular IAT",
            ],
            "false_positives": ["Antivirus telemetry", "Cloud sync clients", "NTP"],
            "severity": "CRITICAL",
        },
        "DDoS": {
            "title": "Distributed Denial of Service Attack",
            "description": "Detects DDoS by massive inbound traffic volume from distributed sources.",
            "condition": [
                "Total inbound flows to single destination IP > 10x baseline in 60s",
                "Many distinct source IPs (distributed = not brute force from one IP)",
                "Short Flow Duration per source (stateless flood)",
                "High Total Fwd Packets, low Bwd Packets (one-way flood)",
                "Alert immediately — escalate to ISP/CDN for scrubbing",
            ],
            "false_positives": ["Flash crowds (viral events)", "CDN traffic spikes"],
            "severity": "CRITICAL",
        },
        "PortScan": {
            "title": "External Port Scan / Reconnaissance",
            "description": "Detects external attacker probing network for open services.",
            "condition": [
                "Single external source IP contacts many destination ports",
                "Flow Duration < 100ms (SYN scan — no full handshake)",
                "Total Fwd Packets = 1 (single SYN packet only)",
                "Total Backward Packets = 0 (port closed) or 1 (port open)",
                "Alert if: >100 distinct ports contacted by same source in 60s",
            ],
            "false_positives": ["Legitimate security scanners", "Shodan/Censys internet-wide scans"],
            "severity": "MEDIUM",
        },
    }

    for attack, rule in SIGMA_RULES.items():
        sev_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(rule["severity"], "⚪")
        lines.append(f"## {attack} — {sev_color} {rule['severity']}\n\n")
        lines.append(f"**Rule title**: {rule['title']}\n\n")
        lines.append(f"**Description**: {rule['description']}\n\n")
        lines.append("**Detection conditions**:\n")
        for c in rule["condition"]:
            lines.append(f"- {c}\n")
        lines.append("\n**False positives**:\n")
        for fp in rule["false_positives"]:
            lines.append(f"- {fp}\n")
        lines.append("\n---\n\n")

    return "".join(lines)


# Section 3: MITRE ATT&CK tactic profile

def section_mitre_tactic_profile(df: pd.DataFrame) -> str:
    """ATT&CK tactic distribution and threat profile of the dataset."""
    lines = [
        "# 3. MITRE ATT&CK Threat Profile of CICIDS2017\n\n",
        "This section describes the adversary behaviour coverage of the dataset ",
        "from a blue team perspective. A SOC team uses this to understand what ",
        "threat categories they are defending against.\n\n",
    ]

    # Count flows per tactic
    tactic_counts: dict[str, int] = {}
    attack_to_tactic: dict[str, str] = {}
    for attack in df["attack_type"].unique():
        m = _get_mitre(attack)
        tactic = m.get("tactic") or "Benign/Normal"
        attack_to_tactic[attack] = tactic
        n = int((df["attack_type"] == attack).sum())
        tactic_counts[tactic] = tactic_counts.get(tactic, 0) + n

    total_attack = int((df["attack_type"] != "Benign").sum())
    total = len(df)

    lines.append("## ATT&CK Tactic Distribution\n\n")
    lines.append("| ATT&CK Tactic | Flow Count | % of Attacks | % of Dataset | Attack Types Covered |\n")
    lines.append("|---------------|-----------|--------------|--------------|---------------------|\n")

    for tactic, count in sorted(tactic_counts.items(), key=lambda x: -x[1]):
        if tactic == "Benign/Normal":
            continue
        pct_attack = count / total_attack * 100 if total_attack > 0 else 0
        pct_total = count / total * 100
        attacks_in = [a for a, t in attack_to_tactic.items() if t == tactic and a != "Benign"]
        lines.append(
            f"| {tactic} | {count:,} | {pct_attack:.1f}% | {pct_total:.1f}% | "
            f"{', '.join(sorted(attacks_in))} |\n"
        )
    lines.append("\n")

    lines.append("## Threat Coverage Analysis\n\n")
    lines.append(
        "CICIDS2017 covers the following phases of the adversary kill chain:\n\n"
    )
    lines.append(
        "- **Reconnaissance**: PortScan, Infiltration — attackers mapping the network before striking.\n"
        "- **Credential Access**: FTP-Patator, SSH-Patator, Web Attack-Brute Force — "
        "gaining initial foothold via stolen credentials.\n"
        "- **Initial Access / Exploitation**: Heartbleed, Web Attack-SQLi, Web Attack-XSS — "
        "exploiting vulnerabilities in public-facing services.\n"
        "- **Command and Control**: Bot — maintaining persistent access and receiving instructions.\n"
        "- **Impact**: DoS variants, DDoS — disrupting service availability.\n\n"
    )
    lines.append(
        "**What is NOT covered**: Lateral movement, Privilege Escalation, Persistence, "
        "Data Exfiltration (as labelled attack classes). This is a known limitation of CICIDS2017 "
        "for real-world SOC use — the dataset emphasises perimeter attacks.\n\n"
    )

    lines.append("## Dataset Imbalance from a Security Perspective\n\n")
    lines.append(
        "The dataset is heavily dominated by Impact-category attacks (DoS/DDoS) "
        "because these generate enormous traffic volume. From a blue team perspective, "
        "this imbalance matters:\n\n"
        "- DoS/DDoS is easy to detect by volume — even a simple threshold rule works.\n"
        "- The hard detection problems are the low-volume attacks: "
        "Heartbleed (11 flows), Infiltration (36 flows), Web Attack-SQLi (21 flows).\n"
        "- A good IDS must detect these rare but high-severity events. "
        "This is why macro F1 (which weights all classes equally) is the right metric — "
        "not accuracy, which would score 99%+ by just ignoring rare classes.\n\n"
    )

    return "".join(lines)


# Section 4: SOC triage playbook

def section_soc_triage_playbook(df: pd.DataFrame) -> str:
    """
    SOC alert triage guide — what to do when the ML model fires an alert.
    Written as a practical reference for a SOC analyst or L1/L2 engineer.
    """
    lines = [
        "# 4. SOC Alert Triage Playbook\n\n",
        "This playbook describes the response actions for each alert type the ML model produces. ",
        "It is written for a SOC analyst (L1/L2) who receives an alert and needs to triage it.\n\n",
        "> **How to use this**: When the IDS model fires an alert with a predicted attack type, ",
        "find the corresponding entry below. Follow the triage steps in order. ",
        "Escalate to L3/security engineer if the alert is confirmed.\n\n",
    ]

    SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    SEVERITY_MAP = {
        "Heartbleed": "CRITICAL",
        "Bot": "CRITICAL",
        "Infiltration": "CRITICAL",
        "DDoS": "CRITICAL",
        "DoS Hulk": "HIGH",
        "FTP-Patator": "HIGH",
        "SSH-Patator": "HIGH",
        "Web Attack-Brute Force": "HIGH",
        "Web Attack-Sql Injection": "HIGH",
        "Web Attack-XSS": "HIGH",
        "DoS GoldenEye": "HIGH",
        "DoS Slowhttptest": "MEDIUM",
        "DoS slowloris": "MEDIUM",
        "PortScan": "MEDIUM",
    }

    # Group by severity
    for sev in SEVERITY_ORDER:
        attacks_in_sev = [a for a, s in SEVERITY_MAP.items() if s == sev]
        if not attacks_in_sev:
            continue
        sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[sev]
        lines.append(f"## {sev_emoji} {sev} Severity Alerts\n\n")

        for attack in attacks_in_sev:
            m = _get_mitre(attack)
            n_flows = int((df["attack_type"] == attack).sum())
            lines.append(f"### Alert: {attack}\n\n")
            lines.append(f"- **ATT&CK**: {m.get('technique_id','—')} — {m.get('technique','—')}\n")
            lines.append(f"- **Tactic**: {m.get('tactic','—')}\n")
            lines.append(f"- **Flows in dataset**: {n_flows:,}\n\n")
            lines.append("**Triage steps**:\n\n")
            soc = m.get("soc_action", "Review manually.")
            for line in soc.strip().split("\n"):
                lines.append(f"{line.strip()}\n")
            lines.append("\n**Detection note** (what the network signature looks like):\n\n")
            lines.append(f"> {m.get('detection_note', '—')}\n\n")
            lines.append("---\n\n")

    lines.append("## Escalation Matrix\n\n")
    lines.append("| Alert Severity | Response Time | First Action | Escalate To |\n")
    lines.append("|---------------|--------------|--------------|-------------|\n")
    lines.append("| 🔴 CRITICAL | Immediately | Isolate host / block IP | L3 + management |\n")
    lines.append("| 🟠 HIGH | Within 15 min | Block source IP, investigate | L2/L3 analyst |\n")
    lines.append("| 🟡 MEDIUM | Within 1 hour | Log and investigate | L2 analyst |\n")
    lines.append("| 🟢 LOW | Within 4 hours | Review in next shift | L1 analyst |\n")
    lines.append("\n")

    return "".join(lines)


# Section 5: Feature separability (what makes each attack detectable)

def section_feature_separability(df: pd.DataFrame, feat: list[str]) -> str:
    """
    For each attack type, identify the top flow features that most distinguish
    it from benign traffic. Uses normalised mean difference (Cohen's d style).
    This tells a detection engineer which features to focus on.
    """
    lines = [
        "# 5. Feature Separability — What Makes Each Attack Detectable\n\n",
        "This section identifies the flow features that most strongly distinguish ",
        "each attack from benign traffic. A detection engineer uses this to understand ",
        "why the ML model works — and to write manual detection rules.\n\n",
        "**Method**: For each feature, we compute the absolute normalised difference ",
        "between the attack mean and benign mean, scaled by the benign standard deviation. ",
        "Higher values = stronger signal for detection.\n\n",
    ]

    benign = df[df["attack_type"] == "Benign"]
    attacks = [a for a in sorted(df["attack_type"].unique()) if a != "Benign"]

    for attack in attacks:
        sub = df[df["attack_type"] == attack]
        m = _get_mitre(attack)
        lines.append(f"## {attack} (ATT&CK: {m.get('technique_id','—')})\n\n")

        scores = []
        for f in feat:
            if f not in df.columns:
                continue
            try:
                b = benign[f].replace([np.inf, -np.inf], np.nan).dropna()
                a = sub[f].replace([np.inf, -np.inf], np.nan).dropna()
                if len(b) < 10 or len(a) < 3:
                    continue
                b_std = b.std()
                if b_std < 1e-9:
                    continue
                diff = abs(a.mean() - b.mean()) / b_std
                direction = "↑ higher" if a.mean() > b.mean() else "↓ lower"
                scores.append((f, diff, direction, float(a.mean()), float(b.mean())))
            except Exception:
                continue

        scores.sort(key=lambda x: -x[1])
        top = scores[:8]

        if top:
            lines.append("| Feature | Signal Strength | Direction vs Benign | Attack Mean | Benign Mean |\n")
            lines.append("|---------|----------------|---------------------|-------------|-------------|\n")
            for fname, score, direction, amean, bmean in top:
                bar = "█" * min(int(score / 2), 10) if score < 20 else "█" * 10 + "+"
                lines.append(f"| {fname} | {score:.1f} {bar} | {direction} | {amean:.2f} | {bmean:.2f} |\n")
        else:
            lines.append("Insufficient data for separability analysis.\n")

        lines.append("\n")

    return "".join(lines)


# Figures

def plot_tactic_distribution(df: pd.DataFrame) -> None:
    """Bar chart: flow count per ATT&CK tactic."""
    tactic_counts: dict[str, int] = {}
    for attack in df["attack_type"].unique():
        m = _get_mitre(attack)
        tactic = m.get("tactic") or "Benign/Normal"
        tactic_counts[tactic] = tactic_counts.get(tactic, 0) + int((df["attack_type"] == attack).sum())

    # Remove benign for attack-only view
    tactic_counts.pop("Benign/Normal", None)
    tactics = sorted(tactic_counts, key=lambda x: -tactic_counts[x])
    counts = [tactic_counts[t] for t in tactics]
    colors = [TACTIC_COLORS.get(t, "#aaa") for t in tactics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(tactics, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Flows", fontsize=11)
    ax.set_title("MITRE ATT&CK Tactic Distribution in CICIDS2017\n(Blue Team Threat Profile)", fontsize=12)
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=8)
    ax.set_xlim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tactic_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[traffic_analysis] saved tactic_distribution.png")


def plot_attack_volume(df: pd.DataFrame) -> None:
    """Horizontal bar chart showing class imbalance."""
    counts = df[df["attack_type"] != "Benign"]["attack_type"].value_counts()
    attacks = counts.index.tolist()
    vals = counts.values.tolist()
    colors = []
    for a in attacks:
        m = _get_mitre(a)
        colors.append(TACTIC_COLORS.get(m.get("tactic"), "#aaa"))

    fig, ax = plt.subplots(figsize=(10, max(5, len(attacks) * 0.45)))
    bars = ax.barh(attacks, vals, color=colors, edgecolor="white")
    ax.set_xlabel("Flow Count (log scale)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("Attack Class Volume in CICIDS2017\n(Note: log scale — huge imbalance)", fontsize=12)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=8)

    # Legend by tactic
    seen = {}
    for a, c in zip(attacks, colors):
        t = _get_mitre(a).get("tactic", "Unknown")
        if t not in seen:
            seen[t] = c
    patches = [mpatches.Patch(color=c, label=t) for t, c in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, title="ATT&CK Tactic")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attack_volume.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[traffic_analysis] saved attack_volume.png")


def plot_feature_heatmap(df: pd.DataFrame, feat: list[str]) -> None:
    """Heatmap of normalised mean feature values per attack type."""
    use_feat = feat[:15]  # top 15 for readability
    attacks = sorted(df["attack_type"].unique())

    matrix = []
    for attack in attacks:
        sub = df[df["attack_type"] == attack]
        row = []
        for f in use_feat:
            if f not in df.columns:
                row.append(0.0)
                continue
            vals = sub[f].replace([np.inf, -np.inf], np.nan).dropna()
            row.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        matrix.append(row)

    mat = np.array(matrix, dtype=np.float64)
    # Normalise each column (feature) to 0-1
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    mat_norm = (mat - col_min) / col_range

    fig, ax = plt.subplots(figsize=(max(10, len(use_feat) * 0.7), max(6, len(attacks) * 0.4)))
    im = ax.imshow(mat_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(use_feat)))
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_xticklabels(use_feat, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(attacks, fontsize=8)
    ax.set_title("Normalised Mean Flow Features per Attack Type\n(darker = higher relative value)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Normalised value (0=min, 1=max)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[traffic_analysis] saved feature_heatmap.png")


def plot_flow_duration(df: pd.DataFrame) -> None:
    """Box plot of Flow Duration per attack type (log scale)."""
    if "Flow Duration" not in df.columns:
        print("[traffic_analysis] Flow Duration not found, skipping box plot")
        return

    attacks = sorted(df["attack_type"].unique())
    data = []
    labels = []
    for attack in attacks:
        sub = df[df["attack_type"] == attack]["Flow Duration"]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[sub > 0]
        if len(sub) > 10:
            data.append(sub.sample(min(5000, len(sub)), random_state=RNG).values)
            labels.append(attack)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.45)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False)
    colors_list = []
    for label in labels:
        m = _get_mitre(label)
        colors_list.append(TACTIC_COLORS.get(m.get("tactic"), "#aaa"))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Flow Duration (microseconds, log scale)", fontsize=10)
    ax.set_title("Flow Duration Distribution per Attack Type\n(colour = ATT&CK tactic)", fontsize=11)
    seen = {}
    for label, color in zip(labels, colors_list):
        t = _get_mitre(label).get("tactic", "Unknown")
        if t not in seen:
            seen[t] = color
    patches = [mpatches.Patch(color=c, label=t, alpha=0.7) for t, c in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, title="ATT&CK Tactic")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "flow_duration_box.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[traffic_analysis] saved flow_duration_box.png")


# Main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_data()
    feat = _available_features(df, KEY_FEATURES)
    print(f"[traffic_analysis] using {len(feat)} key features for analysis")

    print("[traffic_analysis] generating figures ...")
    plot_tactic_distribution(df)
    plot_attack_volume(df)
    plot_feature_heatmap(df, feat)
    plot_flow_duration(df)

    print("[traffic_analysis] writing flow statistics ...")
    sec1 = section_flow_statistics(df, feat)
    (OUT_DIR / "flow_statistics.md").write_text(sec1, encoding="utf-8")
    print(f"[traffic_analysis] saved flow_statistics.md")

    print("[traffic_analysis] writing detection signatures ...")
    sec2 = section_detection_signatures(df, feat)
    (OUT_DIR / "detection_signatures.md").write_text(sec2, encoding="utf-8")
    print(f"[traffic_analysis] saved detection_signatures.md")

    print("[traffic_analysis] writing MITRE tactic profile ...")
    sec3 = section_mitre_tactic_profile(df)
    (OUT_DIR / "mitre_tactic_profile.md").write_text(sec3, encoding="utf-8")
    print(f"[traffic_analysis] saved mitre_tactic_profile.md")

    print("[traffic_analysis] writing SOC triage playbook ...")
    sec4 = section_soc_triage_playbook(df)
    (OUT_DIR / "soc_triage_playbook.md").write_text(sec4, encoding="utf-8")
    print(f"[traffic_analysis] saved soc_triage_playbook.md")

    print("[traffic_analysis] writing feature separability ...")
    sec5 = section_feature_separability(df, feat)
    (OUT_DIR / "feature_separability.md").write_text(sec5, encoding="utf-8")
    print(f"[traffic_analysis] saved feature_separability.md")

    print(f"\n[traffic_analysis] all outputs in {OUT_DIR}")
    print("Files written:")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
