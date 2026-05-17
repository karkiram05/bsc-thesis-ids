# Thesis Video Script — Flow-Based ML for Network Intrusion Detection

**Author:** Ram Karki | **Supervisor:** Gaurav Choudhary | **DTU Compute, 15 ECTS**

**Target length:** 20–25 minutes spoken (≈3,500 words). Adjust pace to taste.

**How to use this script:**
- The text inside `[SLIDE: …]` markers tells you what to put on screen.
- The text in **bold-italic** is the spoken narration.
- The text in plain italics is a director note or stage direction.
- Speak slowly. Pause briefly after every figure or table.

---

## Section 1 — Cold open (≈30 seconds)

[SLIDE 1: Title — "Flow-Based Machine Learning for Network Intrusion Detection" with the DTU logo and your name]

*Look at the camera. Smile.*

> *Hello. My name is Ram Karki, and this is my Bachelor of Science thesis at DTU Compute. It is about machine-learning systems that detect cyber-attacks on computer networks. In the next twenty minutes I will explain what the project does, why it matters, and what I actually found. The honest answer is more interesting than the typical benchmark paper suggests.*

[SLIDE 2: One large headline — "Published IDS scores: F1 ≈ 0.95. Honest deployment scores: F1 ≈ 0.44. This thesis explains the gap."]

> *That gap is what this thesis is about.*

---

## Section 2 — The problem (≈2 minutes)

[SLIDE 3: A simple cartoon — hospital network, packets flowing, one of them red. Caption: "Spot the attacker."]

> *Imagine the IT team of a mid-sized hospital. Their network moves hundreds of thousands of small data packets every minute: patient record lookups, internal email, video calls, software updates. A small number of those packets belong to attackers. Someone is probing for unpatched servers. Someone is guessing passwords against the staff portal. Someone is holding open thousands of slow connections to exhaust the booking system.*

> *The hospital's security team — called a Security Operations Centre, or SOC — has to find those few malicious flows in the flood of normal traffic. Humans cannot do this by hand. A modern enterprise generates millions of network flows per day, and a single analyst processes only fifty to two hundred alerts per shift. If the system raises too many false alarms, the queue overflows and real incidents get missed. That problem has a name in the industry: alert fatigue.*

[SLIDE 4: Three bullets — "Drift. Threshold transfer. Adversarial brittleness."]

> *So security teams deploy automated intrusion detection systems. Increasingly, those systems are built with machine learning. The promise is simple: train a model on labelled flows, deploy it at the network edge, and let it surface only the suspicious flows for a human to review.*

> *Public papers report machine-learning intrusion-detection scores above ninety-five percent macro F1. But field practitioners say these models fail in production. Three things keep breaking. First, traffic patterns shift over time, and a model trained on yesterday's data decays. Second, the threshold tuned in the lab does not transfer cleanly to live traffic. Third, a determined attacker can fool the model with small changes to the timing of their packets. My thesis measures the gap between the laboratory number and the deployment number, and identifies the dominant causes.*

---

## Section 3 — What is a flow, and why does it matter (≈1.5 minutes)

[SLIDE 5: Diagram — two computers connected by a stream of packets, with a box around them labelled "Flow"]

> *Before going further, I need to define one piece of jargon. A network flow is a single conversation between two computers. Five things have to match for packets to belong to the same flow: source IP, source port, destination IP, destination port, and the protocol used. Within that group, the tool I use, called CICFlowMeter, summarises the whole conversation into eighty-four statistics: how long the conversation lasted, how many packets in each direction, how big the packets were, the timing between consecutive packets, which TCP flags were set, and so on.*

> *Think of a flow as a phone call rather than a single spoken word. A whole conversation tells you much more than any one syllable. That is why flow-level statistics are powerful for detection. Attacks tend to have a distinctive conversational shape. A denial-of-service attack produces millions of very short flows. A brute-force password attack produces hundreds of small flows. A slow-loris attack produces a few flows that last hours. Each lives in a different corner of the flow-statistics space, and the machine-learning model only has to learn the boundaries.*

[SLIDE 6: Box plot of Flow Duration per attack class on log scale (figure from your thesis)]

> *This box plot shows the distribution of one feature — flow duration — across the attack classes in my dataset. Benign traffic spans six orders of magnitude. DoS Hulk and DDoS attacks cluster at very short durations. Slowloris attacks cluster at very long ones. Differences this stark are why flow-based detection can work.*

---

## Section 4 — Datasets (≈2 minutes)

[SLIDE 7: Two logos / labels — "CICIDS2017" and "UNSW-NB15"]

> *I use two public datasets. The first is CICIDS2017, captured by the Canadian Institute for Cybersecurity over five days in 2017. After cleaning, I have two-point-three million flow records labelled across fifteen classes: one benign class and fourteen attack subtypes including brute force, denial of service, DDoS, port scanning, and a botnet.*

> *The second dataset is UNSW-NB15, generated at the University of New South Wales using a different testbed and a different traffic generator. It has nine attack categories plus a normal class. I use the official train and test split.*

[SLIDE 8: Big text — "Why two datasets?" with three reasons]

> *Why two datasets and not one? Because a single-dataset result is not really a claim about machine learning — it is a claim about that one dataset. By running the same experiments on two corpora produced by different teams on different testbeds, I can tell the difference between findings that generalise and findings that are artefacts of one specific data collection. As you will see later in this video, the model ranking actually flips between the two datasets, which is exactly the kind of result you can only get from cross-dataset validation.*

[SLIDE 9: Class imbalance bar chart on log scale]

> *Both datasets are extremely imbalanced. In CICIDS, eighty-five percent of the flows are benign. The most common attack class has one hundred seventy-three thousand flows. The rarest attack — Heartbleed — has only eleven flows. That is a ratio of two hundred twenty thousand to one. Because of this imbalance, plain accuracy is a useless metric: a model that always predicts "benign" gets eighty-five percent accuracy and does no useful work. So I use macro F1, which weights every class equally regardless of how big it is.*

---

## Section 5 — The four models (≈3 minutes)

[SLIDE 10: Four labelled icons — LogReg, Random Forest, XGBoost, LightGBM]

> *I compare four classical machine-learning models. Each represents a different family of approaches.*

[SLIDE 11: Visual — straight line dividing two clouds of points]

> *The first is logistic regression. It is the simplest model possible. It learns one weight for each feature, adds them up, and squashes the result through the sigmoid function to produce a probability between zero and one. Geometrically it draws a single straight line through feature space. It is fast, interpretable, and a strong baseline whenever the decision boundary is roughly linear. As a bonus, it produces well-calibrated probabilities, which matters for threshold tuning later.*

[SLIDE 12: Visual — many small trees voting]

> *The second is random forest. It builds many decision trees on random subsets of the training data, and the prediction is the average of all the trees. The averaging makes it robust to noisy features and unusually stable across runs. That property turns out to be important for deployment, because a model that gives different answers every time you retrain it is hard to trust in production.*

[SLIDE 13: Visual — chain of trees with arrows pointing forward]

> *The third and fourth are XGBoost and LightGBM. Both are gradient-boosting libraries. They build trees one at a time, and each new tree tries to correct the mistakes the previous trees made. The result is usually a stronger learner than random forest, but it can over-fit the training distribution if the data has temporal structure — which I will show happens in my dataset.*

> *In plain words: logistic regression draws a single line. Random forest asks a hundred different decision-tree experts and takes a vote. XGBoost and LightGBM build a chain of small experts where each one corrects the previous one's mistakes.*

[SLIDE 14: One question on screen — "Why not deep learning?"]

> *A natural question is why I did not use deep learning. Two reasons. First, recent research, specifically a 2022 paper by Grinsztajn and colleagues, systematically showed that tree ensembles still match or beat deep models on tabular flow data, often by a clear margin, and they train in seconds rather than hours. Second, this is a fifteen-ECTS bachelor project. Adding a deep-learning comparison would have crowded out the adversarial study and the operational triage layer, which are the more distinctive parts of the thesis.*

---

## Section 6 — Evaluation protocols (≈2.5 minutes)

[SLIDE 15: Three labelled boxes — "Stratified", "Day-based", "LODO"]

> *Now the most important methodological choice. I compare three evaluation protocols, each more demanding than the last.*

[SLIDE 16: Cartoon — random shuffle of coloured flows into train/val/test]

> *The first is the stratified random split. I shuffle all the flows, then put sixty percent into training, twenty into validation, and twenty into testing. The proportions of each attack class are preserved in every partition. This is the protocol used in most published papers, and it produces optimistic numbers.*

[SLIDE 17: Diagram — Monday/Tuesday/Wednesday in train (blue), Thursday in validation (orange), Friday in test (red)]

> *The second protocol is the day-based split. CICIDS2017 was captured across Monday through Friday. I train on Monday to Wednesday, use Thursday for validation and threshold tuning, and test on Friday. This is the realistic protocol for a deployed intrusion detector, because in production you train on past traffic and evaluate on future traffic.*

[SLIDE 18: Five rotating diagrams — each day held out once]

> *The third protocol is leave-one-day-out cross-validation, or LODO. I hold out one day for testing, train on the remaining four, and rotate. That gives me five different held-out test scenarios. LODO is the strongest temporal generalisation test I apply, because it exercises the model on five disjoint test distributions instead of just one.*

[SLIDE 19: One big sentence — "If a model only sees Mon–Wed in training, it cannot predict labels that only exist in Thu–Fri."]

> *Here is the crucial structural point about the day-based split. Friday contains three attack classes — DDoS, PortScan, and Bot — that never appear in Monday through Wednesday training. A supervised classifier simply cannot output a label it has never seen during training. The day-based split is therefore both more realistic and more punishing than the stratified split. Stratified evaluation is the published practice; day-based evaluation is what production deployment actually looks like.*

---

## Section 7 — Results: the generalisation collapse (≈2 minutes)

[SLIDE 20: Table — "Multi-class macro F1 by split", showing stratified vs day for the four models, with the 0.86 → 0.44 drop highlighted]

> *Here is the central empirical finding of the thesis.*

> *On the stratified split, XGBoost reaches macro F1 of zero point eight six. Random forest reaches zero point eight four. These are the kinds of numbers that dominate the published literature.*

> *On the day-based split, every tree ensemble collapses to a macro F1 in the zero point four three to zero point four six range. The collapse is uniform: random forest, XGBoost, and LightGBM differ by less than zero point zero four macro F1 on the day split. The reason is the same for all of them. Friday's three attack classes are simply not in training. The classifier cannot produce a label it has never seen, no matter how well it is calibrated.*

[SLIDE 21: Bar chart "generalisation_gap_all_models.png" from the thesis]

> *This bar chart visualises the same finding. The blue bars are the stratified, published-style numbers. The red bars are the day-based, deployment-style numbers. Every tree ensemble loses about zero point four macro F1. This zero-point-four gap is the size of the memorisation tax that published papers pay when they use random stratified splits.*

> *Notice that logistic regression behaves differently. Its score actually improves from stratified to day-based. The reason is that its stratified score was already very low — zero point two seven — because logistic regression cannot capture the non-linear class boundaries that tree models exploit. On the day split, the unseen classes hurt all models equally, but the tree models also lose the spurious correlations they had memorised in training. The net effect for logistic regression is a small recovery in average F1, even though it still cannot predict the unseen classes.*

---

## Section 8 — Binary detection survives (≈1.5 minutes)

[SLIDE 22: Two-panel comparison "binary_vs_multiclass.png"]

> *Now the more hopeful side of the story. If I simplify the question to "is this an attack or not?" — without naming the specific attack — the picture changes completely.*

[SLIDE 23: Table — binary day F1 results, RF = 0.86 highlighted]

> *Random forest on the binary day task reaches F1 of zero point eight six. That is only zero point one four lower than its stratified score of zero point nine nine six. Binary detection survives the temporal shift in a way that multi-class classification does not.*

> *The reason is structural. The binary task only needs to separate attack from benign — it does not need to name which specific attack. Even though Friday's specific attacks were not in training, they still look different from benign traffic in feature space: shorter flows, irregular timing, asymmetric packet ratios. Random forest learns that "this looks attack-like" from the Monday through Wednesday training data and transfers that boundary to Friday.*

[SLIDE 24: Table "Recall @ FPR budget", highlighting RF 62% at 0.1% FPR]

> *Here is what that looks like in deployment terms. At a strict false-positive budget of zero point one percent — meaning about two thousand false alerts per two million flows per day, which one analyst can absorb — random forest still catches sixty-two percent of attacks. XGBoost at the same budget catches only three percent, because XGBoost is poorly calibrated on the day split. Calibration matters a lot when you operate at strict false-positive budgets.*

[SLIDE 25: LODO bar chart "lodo_folds.png"]

> *I confirmed this with leave-one-day-out cross-validation. Random forest gives ROC-AUC of zero point nine two six, with a standard deviation of only zero point zero five three across the four attack-bearing days. That tiny standard deviation is the strongest stability claim in the thesis. A model that gives consistent results across five very different test scenarios is much more trustworthy in production than a model that scores ninety-nine on one day and fifty-three on another.*

---

## Section 9 — Adversarial robustness (≈2.5 minutes)

[SLIDE 26: Drawing — attacker box with arrows to "shape packet timing"]

> *Now I ask a different question: how robust is this detector against an attacker who knows it is there?*

> *I assume what is called a score-query black-box attacker. The attacker can request the model's predicted probability for any input, but does not have access to gradients, model parameters, or training data. This is the realistic capability for an external attacker who only interacts with the deployed system through the network traffic they generate.*

> *Of the seventy-three features I use, only nineteen are actually under the attacker's control: timing statistics, packet lengths, packet counts, and flow duration. The remaining fifty-four — TCP flag counts, header lengths, initial window sizes, protocol identifiers — are determined by the network stack and cannot be freely changed by an application-level attacker.*

[SLIDE 27: Animation or static diagram — coordinate-ascent algorithm idea]

> *The attack itself is called greedy coordinate-ascent. At each step, the attacker tries six different magnitudes of change in each of the nineteen perturbable features, and keeps the single change that most reduces the model's predicted attack probability. The procedure repeats up to fifteen times. The total amount of change is bounded by a budget called epsilon, measured in units of standard deviations on the benign feature distribution.*

[SLIDE 28: Table — bounded evasion 21.4%, unbounded 24.4%, transferability 86.4%]

> *I want to be very precise about what I measured, because there are two different numbers and they mean different things. Under a strict budget of zero point two five sigma — meaning no feature changes by more than a quarter of a standard deviation — twenty-one point four percent of attack flows are flipped to look benign. That is the bounded evasion rate. If I let the attacker run without a strict cap, the same procedure flips twenty-four point four percent of flows. That is the unbounded headline rate. The two numbers describe the same phenomenon at two different attacker effort levels.*

> *Then I tested whether these adversarial flows transfer between models. They do. When I take the flows crafted against random forest and re-score them with XGBoost, eighty-six point four percent of them still fool XGBoost. That means an attacker does not need to know which model the defender deploys. A parallel-ensemble defence in which two models must agree before alerting offers almost no extra protection against this kind of attack.*

[SLIDE 29: Single sentence — "Naïve adversarial training: clean F1 0.86 → 0.73, evasion 24% → 27%"]

> *Then I tried the obvious defence: add adversarial examples to the training set and retrain. This is called naïve adversarial training. The result was a clear negative. Clean F1 dropped from zero point eight six to zero point seven three — a twelve-point loss — and the evasion rate actually increased from twenty-four percent to twenty-seven percent. The naïve defence made both clean and adversarial performance worse simultaneously.*

> *I report this negative result rather than hiding it because it is informative. It demonstrates that adversarial robustness for flow-based intrusion detection is not solved by adding noise to the training set. More sophisticated defences — curriculum-scheduled PGD, certified robustness, or network-layer mitigations like rate limiting — would be needed, and those are open work for a follow-on project.*

---

## Section 10 — MITRE ATT&CK and the OODA loop (≈1.5 minutes)

[SLIDE 30: Four-quadrant diagram — Observe / Orient / Decide / Act]

> *Finally, I built a deployment layer on top of the detector. The goal was to convert raw model output into alerts that a security analyst can act on without further translation.*

> *I use Boyd's OODA loop as the framework. Observe means alert ingestion: the detector scores each flow and emits an alert when the predicted probability exceeds the threshold. Orient means enrichment: each alert is annotated with a MITRE ATT&CK technique ID, a kill-chain phase, and a one-sentence justification. Decide means the analyst's escalate-or-close choice. Act means the response — block the IP, reset the credential, escalate the incident.*

[SLIDE 31: Table — Predicted class → ATT&CK ID mapping]

> *MITRE ATT&CK is the industry-standard taxonomy for attacker behaviour, with over two hundred technique IDs and an open knowledge base maintained by MITRE. I built a static mapping from each of my fifteen predicted classes to the appropriate technique ID. FTP-Patator maps to T-one-one-one-zero-point-zero-zero-one, brute-force password guessing. DDoS maps to T-one-four-nine-eight, network denial of service. PortScan maps to T-one-zero-four-six, network service discovery. And so on.*

> *The bottleneck in the OODA loop is the analyst's Decide step. The detector runs in microseconds, but the analyst takes minutes. The MITRE mapping is the single largest force multiplier in this design, because it converts "model says attack with probability point nine seven" into "this is T-one-one-one-zero, run the credential-stuffing playbook." Each minute saved per alert compounds across thousands of alerts per day.*

---

## Section 11 — What it all means (≈1.5 minutes)

[SLIDE 32: Four-box summary — collapse, binary survives, brittle, no universal winner]

> *Let me summarise the four headline findings.*

> *One. Multi-class classification collapses from zero point eight six macro F1 on the stratified split to zero point four four on the day-based split. The cause is structural: Friday's classes are not in training.*

> *Two. Binary detection survives the temporal shift. Random forest reaches F1 zero point eight six on the day split, with ROC-AUC zero point nine two six plus or minus zero point zero five three across leave-one-day-out folds. Random forest is the most stable detector across genuinely disjoint test scenarios.*

> *Three. The detector is brittle under low-effort evasion. Twenty-one point four percent of attack flows can be flipped at a tiny budget of zero point two five sigma. Adversarial examples transfer to XGBoost at eighty-six percent. Naïve adversarial training makes both clean and adversarial performance worse.*

> *Four. The model ranking flips between the CICIDS and UNSW datasets. XGBoost wins CICIDS multi-class, random forest wins CICIDS day-split binary, LightGBM wins UNSW multi-class. There is no universal winner. Single-dataset benchmarks measure dataset characteristics as much as model quality.*

[SLIDE 33: One sentence — "Originality = integration, not invention"]

> *Now let me address the question of originality, because that is what the examiner will ask about. The originality of this thesis is not in the choice of models — random forest, XGBoost, and LightGBM are standard. It is in the combination of three rarely-co-located components in one reproducible pipeline: honest temporal evaluation, score-query black-box adversarial robustness, and SOC-grade MITRE ATT&CK triage. Each component has been studied separately in the literature. Their integration into a single end-to-end study, with explicit negative results where they appear, is what this thesis contributes.*

---

## Section 12 — Limitations and future work (≈1 minute)

[SLIDE 34: Four bullets — synthetic data, no live traffic, feature-space attacks, static mapping]

> *Let me be honest about what this thesis does not do. Both CICIDS and UNSW are synthetic captures from controlled testbeds. Real enterprise traffic is messier — more diverse, more encrypted, with a long tail of rare attacks. My numbers are likely upper bounds on what would be achievable in production.*

> *The adversarial study perturbs feature vectors, not real network packets. There is a gap, formalised by Pierazzi and colleagues in 2020, between a flow vector that fools a model and a packet sequence that an attacker can actually transmit on the wire. Bridging that gap is open work.*

> *The MITRE mapping is a static lookup table rather than a learned mapping over richer telemetry. And there is no live deployment, no streaming evaluation, no adversary simulator.*

[SLIDE 35: Five future-work bullets]

> *Future work would extend in five directions. Cross-dataset LODO — train on CICIDS days, test on UNSW splits. A learned MITRE mapping with richer features. Curriculum-scheduled adversarial training that does not fail like the naïve version. Streaming deployment with concept-drift detection. And a comparison against modern tabular deep models like TabNet and FT-Transformer.*

---

## Section 13 — Closing (≈30 seconds)

[SLIDE 36: Single sentence — "Honest evaluation beats novel models."]

> *To close: flow-based machine learning for intrusion detection works in practice, but only when it is evaluated under protocols that respect the temporal structure of the data. Headline numbers from random stratified splits substantially overstate the deployment-relevant performance reported under day-based and LODO splits.*

> *The contribution that distinguishes this thesis is the integration: honest temporal evaluation, adversarial robustness analysis, and operational MITRE triage in one reproducible pipeline. The unsolved problem is adversarial robustness. Any deployed detector should assume that a determined adversary will eventually evade it, and the right response is defence in depth — the machine-learning detector alongside network-layer mitigations, host-based detection, and human analysts.*

[SLIDE 37: Final slide — name, supervisor, repository, contact]

> *Thank you for watching. The full thesis, the code, and the reproducible pipeline are available at the project repository. I would like to thank my supervisor, Gaurav Choudhary, and DTU Compute for supporting this work.*

*Smile. Hold for two seconds. Fade out.*

---

## Filming tips

- **Equipment.** A phone on a tripod is fine. Use the back camera, not the selfie camera (sharper). Sit two metres from a window for natural light; avoid filming with a bright window behind you.
- **Audio.** Voice is the most important part of the video. Use headphones with an inline mic if you have them, or any USB microphone. Phone built-in mic is acceptable in a quiet room but noticeable in editing. Test by recording thirty seconds first.
- **Slides.** Build slides in Keynote or PowerPoint. One idea per slide. Use the figures from your thesis directly — they are already polished. Dark text on white background reads best on phone screens.
- **Recording method.** Two options: (1) film yourself talking, then cut to slides in editing; (2) screen-record slides while reading the script aloud, then overlay your face in a small corner. Option 2 is easier and faster for a thesis-explainer style.
- **Editing software.** iMovie (Mac, free) or DaVinci Resolve (Mac/Windows, free) are both fine. You do not need anything more.
- **Pace.** Speak at around 150 words per minute. This script is roughly 3,500 words, which is around 23 minutes of speaking. If you want a shorter video, cut Sections 5 (the four models — already on slides), 6 (evaluation protocols — same), or 12 (limitations) by half each.
- **Rehearse.** Read the script aloud twice before you record. The first time you read it you will trip over numbers; the second time you will sound natural.
- **Mistakes.** Do not start over the whole take when you make a small mistake. Pause, breathe, and restart from the previous sentence. You can cut around the mistake in editing.

## A shorter five-minute version

If you need a five-minute explainer instead, keep only these sections and trim each to under one minute: Section 1 (cold open), Section 2 (the problem), Section 7 (the collapse), Section 8 (binary survives), Section 13 (closing). That gives you the headline story without the technical depth.

## A ten-minute version

Sections 1, 2, 4 (datasets), 6 (evaluation protocols), 7 (collapse), 8 (binary survives), 9 (adversarial), 13 (closing). Drops the model-detail and limitations sections.
