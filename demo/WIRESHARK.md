# Wireshark + Thesis Pipeline: Defence Playbook

End-to-end live demo, verified working on macOS with Wireshark 4
and Python 3.12. Captures real packets off your laptop NIC, converts
them to the same 73 CICIDS2017 flow features the thesis trained on,
and scores them with the deployed binary day-split random forest.

## One-shot path

If you only want the live demo to work, do this:

```bash
brew install --cask wireshark
brew install nmap
source /Users/ramkarki/bsc-thesis-ids/.venv/bin/activate
pip uninstall -y cicflowmeter
pip install git+https://github.com/hieulw/cicflowmeter.git
cd /Users/ramkarki/bsc-thesis-ids/demo
DEFENCE_SCAN=1 ./live_capture.sh
```

Output looks like:

```
[1/4] capturing 20s on en0
[1b ] nmap -sT -p 1-500 -T4 192.168.8.1 (during capture)
[2/4] captured: 2852 packets captured
[3/4] cicflowmeter -> /tmp/live_flows.csv (522 lines)
[4/4] scoring with binary day-split RF (theta=0.009):
Flows scored : 521
Alerts raised: 21  (threshold p >= 0.009)
Attack rate  : 4.0%
```

That is the model alerting on your own nmap traffic from your own
laptop, in 2026, using a model trained on 2017 CICIDS data.

## Why this matters for the defence

Examiner question you should expect:

> "Your model was trained on a 2017 dataset. Does it actually work on
> live traffic?"

With this playbook you have an answer. Capture 20 seconds of your
own traffic, score with the deployed RF, show the result. The model
alerts on your port scan because port-scan flow signatures are
structurally stable, not because the year is 2017.

## Honest limitations up front

1. **Two-stage pipeline, not real-time.** PCAP capture and
   CICFlowMeter both batch. A production deployment would use Zeek or
   Suricata, which emit flow records directly without the PCAP round
   trip. The Future Work section of Chapter 8 already calls this out.
2. **TCP connect scan, not SYN scan.** Without sudo, nmap falls back
   to `-sT` (TCP connect). This is quieter than `-sS` because each
   connection completes the handshake. If you want louder alerts,
   `sudo nmap -sS` during the capture, but the demo works fine
   without sudo.
3. **Threshold tuned for the lab dataset.** The 0.009 Youden-J
   threshold maximises F1 on the CICIDS2017 Friday test set. On live
   home Wi-Fi traffic, you will see a handful of false positives on
   benign HTTPS flows. This is precisely the false-positive cost
   discussed in Sections 5.2 and 6.3, made concrete.

## One-time setup

### 1. Install Wireshark

```bash
brew install --cask wireshark
```

First launch will prompt for the BPF (Berkeley Packet Filter)
capture permissions. Approve them. This installs ChmodBPF, which adds
your user to the `access_bpf` group so non-root tshark works.

Verify:

```bash
groups | tr ' ' '\n' | grep access_bpf    # should print 'access_bpf'
/Applications/Wireshark.app/Contents/MacOS/tshark -D | head -5
```

The tshark binary lives inside the .app bundle. The wrapper script
finds it automatically; if you want it on your shell PATH:

```bash
echo 'export PATH="/Applications/Wireshark.app/Contents/MacOS:$PATH"' >> ~/.zshrc
```

### 2. Install CICFlowMeter (hieulw fork)

The mainline `cicflowmeter` package on PyPI (currently 0.4.2) has a
broken default for `--fields` (bool instead of str). The hieulw fork
is the one most CICIDS2017 follow-up papers use:

```bash
source /Users/ramkarki/bsc-thesis-ids/.venv/bin/activate
pip uninstall -y cicflowmeter
pip install git+https://github.com/hieulw/cicflowmeter.git
```

Verify:

```bash
cicflowmeter --help | head -5
```

### 3. Install nmap

```bash
brew install nmap
```

### 4. Confirm trained model is in place

```bash
ls -lh models/binary_day/random_forest.joblib
ls -lh models/binary_day/feature_names.json
```

Both must exist. If missing, run `make full` first.

## The wrapper script

`demo/live_capture.sh` is the single command that does it all:

```bash
./live_capture.sh                  # 20s capture on en0
./live_capture.sh 30 en0           # custom duration/interface
DEFENCE_SCAN=1 ./live_capture.sh   # also fire an nmap connect-scan
                                   # at the default gateway during
                                   # the capture window
```

Internally it:
1. Captures with the bundled tshark for N seconds (no sudo needed).
2. If `DEFENCE_SCAN=1`, kicks off `nmap -sT -p 1-500 -T4` against your
   default gateway during the capture window.
3. Runs `cicflowmeter` on the resulting pcap.
4. Calls `score_pcap.py` which applies the `Title Case <-> snake_case`
   column mapping and scores flows with the RF.

## Suggested 3-minute defence demo

1. **Dashboard reel** (30 s). `cd demo && python app.py`. Open
   http://127.0.0.1:5050. "This is the deployed random forest from
   Section 5.2. Each row is a Friday test flow scored above the 0.009
   threshold. Green is a correct alert, red is a false positive on a
   benign flow that crossed the threshold."

2. **MITRE enrichment** (30 s). "Each alert is tagged with the ATT&CK
   technique from Section 6.1. T1110 routes to brute-force playbooks,
   T1498 to DDoS playbooks. A real SOC consumes these IDs directly."

3. **Live capture punchline** (1 min). Switch to a terminal:
   ```bash
   DEFENCE_SCAN=1 ./live_capture.sh 15
   ```
   "tshark is capturing my Wi-Fi for 15 seconds. nmap is firing 500
   port-scan connections at my router during the same window.
   CICFlowMeter converts the pcap to the 73-feature representation,
   and the model scores it. 521 flows in, 21 alerts out, mostly
   scan flows to 192.168.8.1. The training data is from 2017. The
   scan is from today. The model still flags it."

4. **Adversarial caveat** (1 min). "But this same model has a 20.8
   percent bounded evasion rate in Section 5.5. If I added a 0.16
   millisecond delay between every fifth SYN packet, the alerts
   would disappear. The detector is useful but not unforgeable. The
   deployment recommendation in Section 5.6 is RF for current
   operations, paired with Zeek for flow primitives and a second
   model in an ensemble."

## Wireshark display filters worth knowing for the viva

These define the attack flows the thesis was trained on:

- SYN with no ACK (scan or connection start):
  `tcp.flags.syn == 1 and tcp.flags.ack == 0`
- HTTP floods (Hulk DoS):
  `http.request.method == "GET"`
- Slow HTTP (Slowloris):
  `tcp.port == 80 and tcp.analysis.keep_alive`
- DNS exfiltration candidates:
  `dns and frame.len > 100`
- Heartbleed (CVE-2014-0160):
  `ssl.record.content_type == 24`
- FTP failed logins (brute force):
  `ftp.response.code == 530`

## Troubleshooting

**`tshark: command not found`**. Wireshark ships tshark inside the
.app bundle, not on PATH. Either add it to PATH (see step 1) or use
the wrapper, which finds it automatically.

**`Capturing on 'en0'` then 0 packets**. ChmodBPF perms missing.
Reinstall Wireshark with the cask and approve the permission prompt.

**`cicflowmeter: 'bool' object has no attribute 'split'`**. You are
on the mainline PyPI version 0.4.2 which has the bug. Switch to the
hieulw fork (see step 2).

**`nmap: You requested a scan type which requires root privileges`**.
SYN scan (`-sS`) needs root. Use `-sT` (TCP connect) instead, or
`sudo nmap -sS`.

**Score reports `73 of 73 features unmappable`**. The CICFlowMeter
output uses different column names. `score_pcap.py` has the
Title Case to snake_case mapping built in for the hieulw fork. If
you used the Java original instead, columns are already Title Case
and the mapping is the identity, but `score_pcap.py` would need a
small tweak: change `COL_MAP[k] = v` to `COL_MAP[k] = k`.

**All flows score 0.001**. Capture is genuinely benign. Run `nmap`
during the capture, or extend duration.

**Dashboard does not refresh**. Check
`http://127.0.0.1:5050/api/stats` directly. If you see JSON, the
backend is fine and the issue is the browser tab.

## File map

```
demo/
  app.py             Flask SOC dashboard (replay)
  score_pcap.py      Score a CICFlowMeter CSV with the deployed RF
  live_capture.sh    One-shot wrapper: capture, convert, score
  WIRESHARK.md       This file
  README.md          Dashboard quickstart
  templates/
    dashboard.html   Dashboard UI (dark terminal aesthetic)
```
