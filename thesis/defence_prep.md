# Defence Prep Pack — BSc Thesis

**Title:** Flow-Based Machine Learning for Network Intrusion Detection
**Author:** Ram Karki | **Supervisor:** Gaurav Choudhary | **DTU Compute, 15 ECTS**

This document is the defence preparation pack. It contains:
1. The opening summary you will give the examiner.
2. The numbers and formulas you must memorise.
3. Questions an examiner is likely to ask, with the best short answer for each.

Read it twice, then practise the opening summary out loud until it feels natural.

---

## PART 1 — How to explain the project

### 30-second elevator pitch (memorise this word-for-word)

> "My thesis is about how machine-learning intrusion detection systems are evaluated. Public benchmarks like CICIDS2017 report macro F1 scores near 0.95, but practitioners say these models do not work in production. I tested four classical models on two datasets under three evaluation protocols and showed that the published scores reflect in-distribution memorisation, not real generalisation. The honest, deployment-relevant number for the multi-class task is closer to 0.44, while binary detection survives at 0.86. I also showed the detector is brittle under low-cost evasion, and that the obvious defence makes things worse. The contribution is not a new model — it is honest evaluation, adversarial robustness, and MITRE ATT&CK triage in one reproducible pipeline."

### 3-minute supervisor-style explanation

Use this if your supervisor asks "tell me what you did".

> "I built a complete machine-learning pipeline for network intrusion detection using two public flow datasets: CICIDS2017 from the Canadian Institute for Cybersecurity, and UNSW-NB15 from the IXIA PerfectStorm testbed. I trained four classical models — logistic regression, random forest, XGBoost, and LightGBM — and compared them.
>
> The motivation came from a problem in the literature. Most papers split their data randomly and report macro F1 scores above 0.95. But practitioners say these models fail in production. I wanted to test whether the published numbers reflect real model quality or just memorisation of one specific dataset.
>
> I trained the models under three different evaluation protocols. First, the random stratified split that most papers use. Second, a day-based split, where I train on Monday to Wednesday and test on Friday — this matches how a real system would be evaluated. Third, leave-one-day-out cross-validation, which is the strongest temporal test.
>
> The headline finding was clear. Multi-class macro F1 dropped from 0.86 on the stratified split to 0.44 on the day-based split, because Friday's attack classes never appear in training. But when I reformulated the task as binary attack-versus-benign, the random forest still reached F1 = 0.86 on the day split, and ROC-AUC 0.926 ± 0.053 across LODO folds. Binary detection survives temporal shift; fine-grained attack classification does not.
>
> I then asked: how easily can an attacker evade the detector? I implemented a score-query black-box greedy evasion attack on a 19-feature subset and found 21.4% of attacks could be flipped at an L-infinity budget of 0.25 sigma. The adversarial flows transferred to XGBoost at 86%. I also tested naïve adversarial training as a defence and reported that it failed: clean F1 dropped from 0.86 to 0.73 while evasion rate went up.
>
> Finally, I mapped predicted attack types to MITRE ATT&CK technique IDs so that the detector output can be routed directly to existing SOC playbooks.
>
> Everything is reproducible from a single Makefile in 45 to 60 minutes."

### 10-minute defence opening

Slide-by-slide outline you can build slides from.

1. **Title slide** — name, supervisor, DTU Compute, May 2026.
2. **Motivation** — Hospital example. Millions of flows per day. Analyst cannot inspect manually. ML promises help, but practitioners say deployed systems fail.
3. **Research questions** — RQ1 temporal generalisation, RQ2 adversarial robustness, RQ3 operational triage.
4. **Datasets** — CICIDS2017 (2.31M flows after cleaning, 15 classes), UNSW-NB15 (175k test rows, 10 classes). Why both: cross-dataset sanity check.
5. **Method** — 4 models, 3 split protocols. Class imbalance handled with class weights. Threshold tuning via Youden's J. Statistical tests: bootstrap CIs (B=1000) and McNemar.
6. **Headline result 1** — Multi-class collapse: 0.86 → 0.44. Show generalisation-gap figure.
7. **Headline result 2** — Binary survives. RF day F1 = 0.86. LODO ROC-AUC = 0.926 ± 0.053. Show LODO folds figure.
8. **Headline result 3** — Adversarial: 21.4% evasion at ε=0.25σ, transfer 86% to XGBoost, naïve adv-training fails.
9. **Best model** — RF wins 6 of 11 criteria. Show the comparison table.
10. **MITRE mapping** — Predicted classes → ATT&CK technique IDs → SOC playbooks.
11. **Limitations** — CICIDS is synthetic, McNemar p-values affected by temporal correlation, attack lives in feature space not packet space.
12. **Conclusion** — Honest evaluation matters more than model choice. RF is the most deployable model under these benchmarks. Adversarial robustness remains unsolved.

---

## PART 2 — Numbers you must memorise

| Quantity | Value |
|---|---|
| CICIDS flow count after cleaning | 2,313,150 flows, 73 features |
| CICIDS classes | 1 Benign + 14 attack subtypes |
| UNSW test rows | 175,341 |
| Stratified XGBoost multi-class F1 | 0.86 |
| Day-split multi-class F1 (any tree model) | 0.43–0.46 |
| Day-split binary F1 (Random Forest) | 0.859 |
| LODO Random Forest ROC-AUC | 0.926 ± 0.053 |
| LODO best F1 model | Random Forest (0.549 ± 0.366) |
| Bounded evasion at ε = 0.25σ | 21.4% |
| Unbounded (headline) evasion rate | 24.4% |
| Median L∞ to flip a prediction | 0.250σ |
| Transferability RF → XGBoost | 86.4% |
| Adv-training: clean F1 before / after | 0.859 → 0.734 |
| Adv-training: evasion before / after | 24.4% → 26.8% |
| RF recall at 0.1% FPR (day split) | 62% |
| XGBoost recall at 0.1% FPR | 3% |
| RF ECE on day split | 0.18 |
| LogReg ECE on day split | 0.073 |
| RF inference latency at batch=10k | 3.19 µs / flow |
| RF throughput | 313,400 flows / second |
| Perturbable feature subset | 19 of 73 features |
| Most-moved evasion feature | Flow IAT Min (83.6% of successful attacks) |
| Random seed | 42 |
| Pipeline runtime | 45 – 60 minutes |
| McNemar χ² closest pair (XGB vs LGB) | 520.6 (p < 10⁻⁴) |
| Bootstrap resamples | B = 1000, seed 42 |

---

## PART 3 — Formulas you must know

You will not be asked to derive these. You only need to recognise them and explain what they mean in one sentence.

| Name | Formula | One-line meaning |
|---|---|---|
| **Sigmoid** | σ(z) = 1 / (1 + e⁻ᶻ) | Squashes any real number to a probability between 0 and 1. Used in logistic regression. |
| **Logistic regression** | P(y=1 \| x) = σ(wᵀx + b) | Linear combination of features, passed through sigmoid. |
| **Gini impurity** | 1 − Σᵢ pᵢ² | How mixed the classes are in a tree node; trees split to reduce it. |
| **F1 score** | F1 = 2·P·R / (P + R) | Harmonic mean of precision and recall. Punishes either weakness. |
| **Macro F1** | average of F1 across all classes | Treats every class equally regardless of size. |
| **Youden's J** | J = TPR − FPR | Threshold-selection criterion. Maximising J gives the best balance of sensitivity and specificity. Prevalence-invariant. |
| **McNemar χ²** | χ² = (\|b − c\| − 1)² / (b + c) | Compares two classifiers on the same test set. b = A wrong, B right. c = A right, B wrong. |
| **Bootstrap CI** | Resample test set with replacement B times; report 2.5th and 97.5th percentile | Gives the sampling-noise floor for any metric. |
| **ECE** | Expected Calibration Error: weighted average of \|confidence − accuracy\| per bin | Measures how close predicted probabilities are to observed frequencies. Lower is better. |
| **L∞ norm** | ‖x‖∞ = max(\|x₁\|, \|x₂\|, ...) | Largest absolute change across all dimensions. Bounds an adversarial perturbation. |

---

## PART 4 — Likely defence questions, with best answers

### Project overview

**Q1. Tell me about your project in two minutes.**
Use the 3-minute supervisor explanation from Part 1.

**Q2. What is the contribution of your thesis?**
The contribution is not a new model. It is the combination of three things in one reproducible pipeline: honest temporal evaluation, score-query black-box adversarial robustness analysis, and SOC-grade MITRE ATT&CK triage. Each component exists in prior literature, but their integration into a single end-to-end study, with explicit negative results, is what distinguishes the work.

**Q3. Why is this project interesting?**
Because there is a real gap between published intrusion-detection numbers and what works in production. Practitioners say machine-learning IDS fail in deployment. My thesis quantifies that gap on two public benchmarks and identifies its main causes: temporal distribution shift, threshold transfer, and adversarial brittleness.

**Q4. Why 15 ECTS is the right scope?**
I deliberately bounded the work. I used two public datasets, four classical models, and excluded deep learning. The reproducible pipeline runs in 45 to 60 minutes. A wider scope — for example, deep models or live traffic capture — would have crowded out the adversarial study and the operational triage layer, which are the most distinctive parts.

### Methodology questions

**Q5. Why these four models?**
Logistic regression is the linear baseline. Random forest is the standard bagging ensemble. XGBoost and LightGBM are the two leading gradient-boosting libraries. Together they span the spectrum from linear to non-linear, from single-tree to ensemble, from bagging to boosting.

**Q6. Why not deep learning?**
Two reasons. First, Grinsztajn et al. in 2022 showed that tree ensembles still match or beat deep models on tabular data, including flow-feature benchmarks. Second, a 15-ECTS project has limited compute; deep models would take hours per run and crowd out the adversarial and operational analyses that make the thesis interesting.

**Q7. Why two datasets, not one?**
Because a single-dataset claim is not really a claim about machine learning. It is a claim about that one dataset. CICIDS2017 and UNSW-NB15 were generated by different teams, on different testbeds, with different attack catalogues. Cross-dataset validation lets me distinguish robust findings from dataset artefacts. The model ranking flips between datasets, which is the strongest evidence that single-dataset benchmarks measure dataset characteristics as much as model quality.

**Q8. Why did you drop the four rate features?**
Engelen et al. in 2021 showed that the rate features in CICIDS2017 — Flow Bytes per second, Flow Packets per second, Forward and Backward Packets per second — leak the label through their denominators. They correlate strongly with the attack class because of how CICFlowMeter computes them. I dropped these four features explicitly and the leakage_check module enforces their absence on every run. Dropping them alone reduces stratified random-forest macro F1 by about four percentage points.

**Q9. Why Youden's J for threshold selection?**
The canonical 0.5 threshold hides large per-model differences and fails when the attack rate differs between training and test. Youden's J equals TPR minus FPR, and the threshold that maximises it is prevalence-invariant — it gives the same operating point whether the test set has 5% or 50% attacks. This is critical for the day-based splits where Monday is benign-only and Wednesday has 80% attacks.

**Q10. Why bootstrap and McNemar?**
Bootstrap gives me the sampling-noise floor. With 516,643 test flows the CIs are narrow, around ±0.001 on F1. McNemar compares two classifiers on the same test set: it counts where model A is wrong but B is right, and the reverse. The χ² statistic tests whether the two have equal error rates. Both tests support the model ranking — every pair is significant at p less than 10⁻⁴, and bootstrap CIs do not overlap.

**Q11. Three split protocols — why all three?**
Stratified is the protocol most published papers use. Day-based is the realistic protocol for deployed IDS, because a deployed model is trained on past traffic and evaluated on future traffic. LODO is the strongest temporal generalisation test because it exercises the model on five disjoint held-out distributions rather than one. Reporting all three exposes the gap between optimistic published numbers and realistic deployment performance.

**Q12. Why is class imbalance such a big deal?**
In CICIDS, 85% of flows are benign and the rarest attack class has only 11 examples. A classifier that always predicts benign already gets 85% accuracy and is useless. That is why macro F1 is the right metric: it averages F1 across classes weighting each equally, so a tiny class cannot be ignored. I also use class-weight balanced for LogReg and RF, is-unbalance for LightGBM, and scale-pos-weight for XGBoost on the binary task.

### Results questions

**Q13. Explain the multi-class collapse.**
Macro F1 drops from 0.86 on the stratified split to 0.44 on the day-based split. The reason is structural: Friday's test classes — DDoS, PortScan, Bot — never appear in Monday-to-Wednesday training. A supervised classifier cannot output a label it has never seen. Per-class F1 on the day split is exactly 0.0 for those three classes across all four models. No amount of hyperparameter tuning fixes this; the only fix is more training data that covers all the classes you care about, or reformulating the problem.

**Q14. Why does logistic regression's score improve on the day split?**
This looks strange but has a simple explanation. Its stratified score was already low (0.27) because it cannot capture the non-linear class boundaries that tree models exploit. On the day split, the unseen classes hurt all models equally. But the tree models also lose the spurious correlations they had memorised, while logistic regression had not learned them in the first place. The net effect is a small recovery in average F1 for LogReg even though it still cannot predict the unseen classes.

**Q15. Why does binary detection survive when multi-class fails?**
Because the binary task only needs to separate attack from benign. The decision boundary does not depend on naming the specific attack. Even though DDoS, PortScan, and Bot are unseen, they still look different from benign traffic in the feature space — short flows, irregular timing, asymmetric packet ratios. Random forest learns "this looks attack-like" from the Mon-Wed attack examples and transfers that boundary to Friday.

**Q16. Why does the random forest win LODO stability?**
Bagging. RF trains many trees on bootstrap samples and averages them, which smooths out the noise from any one fold. XGBoost has higher mean ROC-AUC on some folds but its threshold is sensitive to the day-specific class prevalence, so its F1 spread is much wider. Stability across days is more important than peak performance because production traffic shifts continuously.

**Q17. What does the SHAP summary plot tell you?**
For the binary random forest on the day split, the most influential feature is Flow IAT Min — the smallest inter-arrival time. Small Flow IAT Min values push predictions strongly toward attack, because attack flows have tight burst structure. This is operationally important because it tells the defender which feature an attacker would need to manipulate. It is also why Flow IAT Min is the most-moved feature in 83.6% of successful evasions in Chapter 10.

**Q18. What does ECE measure and why is it different across models?**
ECE — Expected Calibration Error — measures how close a model's predicted probabilities are to the observed frequencies. A model that says "probability 0.7" should be right 70% of the time at that score. Logistic regression on the day split has ECE = 0.073, which is well-calibrated. The tree ensembles are markedly worse: RF = 0.18, XGB and LGB = 0.25. Poor calibration is why a 0.5 threshold fails for tree ensembles under temporal shift — their probability surface drifts.

### Adversarial questions

**Q19. Explain your threat model in one sentence.**
A score-query black-box attacker can request predicted probabilities for any input flow, has no access to gradients or model parameters, and has no rate limit on queries. The attacker can only perturb 19 of the 73 features — timing, packet lengths, packet counts, and flow duration. The remaining 54 are determined by the network stack.

**Q20. What is greedy coordinate-ascent in plain English?**
At each iteration, try six different step magnitudes in each of the 19 perturbable features. Keep the single perturbation that reduces the predicted attack probability the most. Repeat for up to 15 iterations per flow. The total perturbation is bounded by ε in the L∞ norm, meaning no single feature changes by more than ε standard deviations.

**Q21. What is the difference between bounded evasion (21.4%) and unbounded headline evasion (24.4%)?**
At ε = 0.25σ, robust accuracy is 0.786, meaning 21.4% of attack flows are flipped under that hard cap. The 24.4% headline number is the evasion rate when the attack is allowed to run unbounded until the perturbation is minimised. The gap reflects flows that need a slightly larger perturbation than 0.25σ to flip. The median L∞ to flip a prediction is exactly 0.250σ.

**Q22. What is transferability and why is 86.4% a problem?**
Transferability means an adversarial example crafted against one model also fools a different model. When the same RF-adversarial flows are scored by XGBoost, 86.4% of them are still misclassified. Clean attack flows are misclassified by XGBoost at only 26%. This means an attacker does not need to know which model the defender deploys. A parallel-ensemble defence in which two models must agree before alerting offers very little extra protection.

**Q23. Why did naïve adversarial training fail?**
Three reasons. First, no curriculum scheduling: Madry's PGD uses gradually increasing perturbation budgets, but I just added 2,000 adversarial flows at one budget. Second, no class rebalancing: adding attack-only adversarial flows distorts the class prior. Third, simple augmentation: the defence learned to ignore the specific adversarial flows rather than generalising. The result is clean F1 dropped from 0.86 to 0.73 and evasion rate increased from 24.4% to 26.8%. The defence makes both metrics worse simultaneously, consistent with Tramer et al. 2020.

**Q24. What is the difference between feature space and problem space?**
Pierazzi et al. 2020 formalised this. Feature space attacks craft a flow vector that fools the model. Problem space attacks must generate a real packet sequence that an attacker could actually transmit on the wire. My attacks live in feature space — they might produce flow vectors that do not correspond to any physically realisable packet sequence. Bridging the gap is open work and a real limitation of this thesis.

**Q25. Why didn't you use PGD or AutoAttack?**
PGD needs gradients, and tree ensembles are non-differentiable in any useful sense. AutoAttack is an ensemble of stronger attacks but most still require gradients. Greedy coordinate-ascent is a fair upper bound on what a score-query black-box adversary can achieve. The Apruzzese et al. 2023 paper "Real Attackers Don't Compute Gradients" specifically argues that gradient-based attacks overstate the realistic adversarial threat for deployed IDS.

### Operational / deployment questions

**Q26. Why MITRE ATT&CK and not something else?**
ATT&CK is the industry-standard taxonomy for adversary behaviour, organised by tactic and technique. It is more granular than the Lockheed kill chain (over 200 techniques) and is maintained as an open knowledge base. SOC playbooks are usually keyed to ATT&CK technique IDs, so mapping the detector output to ATT&CK lets the analyst route an alert to an existing playbook without manual translation.

**Q27. What does OODA stand for and why does it matter here?**
Observe, Orient, Decide, Act. Boyd's decision cycle from military strategy. In a SOC, Observe is alert ingestion, Orient is enrichment with context — reputation, asset criticality, MITRE mapping — Decide is the analyst's escalate-or-close decision, and Act is containment. The cycle matters because the analyst's Decide step is the bottleneck. MITRE mapping is the largest force multiplier because it shrinks the Decide step from "interpret a probability" to "this is T1110, run the credential-stuffing playbook".

**Q28. What are the seven kill-chain stages?**
Reconnaissance, weaponisation, delivery, exploitation, installation, command-and-control, actions on objectives. Different CICIDS attacks sit at different stages — Patator brute force at credential access (stage 4), DDoS at actions on objectives (stage 7), Heartbleed at exploitation (stage 4).

**Q29. Why is recall at fixed FPR more useful than F1?**
A SOC has a finite alert budget. At 1% FPR on a two-million-flow day, a model raises 20,000 false alerts — too many for analysts. At 0.1% FPR, it raises 2,000 false alerts, which a single analyst team can absorb. The deployment question is "how much recall can I deliver under that constraint", not "what is my F1 at some arbitrary threshold". My table reports recall at three FPR budgets so the SOC can pick the right operating point.

**Q30. Could this be deployed in production today?**
No. The thesis is an offline benchmark study. Production deployment would require live traffic validation, continuous concept-drift detection, weekly retraining on a rolling window, integration with the existing SIEM and SOAR stack, and network-layer mitigations against the evasions I demonstrated. The thesis identifies which model is most ready for that path (random forest), but does not claim it has been deployed.

### Limitations / critical questions

**Q31. How representative is CICIDS2017 of real enterprise traffic?**
Limited. CICIDS was generated on a controlled testbed in 2017 with synthetic traffic. Real enterprise traffic is messier — more diverse user behaviour, more encrypted traffic, a long tail of rare attack variants. The numbers in this thesis are likely upper bounds on what would be achievable in production. The cross-dataset comparison with UNSW-NB15 partially mitigates this, but neither dataset is real enterprise traffic.

**Q32. What about encrypted (TLS) traffic?**
Modern enterprise traffic is mostly TLS-encrypted, which hides the payload from any inspection tool. Flow-based detection still works because it uses only flow-level statistics — durations, packet counts, inter-arrival times — that are visible regardless of encryption. So my method is encryption-resistant in principle. But CICIDS2017 contains relatively little encrypted traffic, so my numbers do not directly reflect performance on a TLS-heavy network.

**Q33. Could the models just be learning the lab environment?**
Possibly. The day-based split partially controls for this because the lab environment is consistent across days. But subtle artefacts — testbed clock skew, the specific attacker tooling used, the baseline traffic mix — could leak into the model. The fact that binary detection generalises across LODO folds suggests the signal is at least partly real, but a truly conclusive answer would require live enterprise validation.

**Q34. Why didn't you test on a third dataset?**
Scope. Two datasets already let me test cross-dataset rank reversal, which is the key cross-dataset finding. Adding a third would have crowded out the adversarial chapter, which is the more distinctive contribution. A natural extension would be cross-dataset LODO — train on CICIDS days, test on UNSW splits — but that needs feature-schema alignment between the two extractors and is non-trivial engineering.

**Q35. Are your McNemar p-values trustworthy?**
Not as exact significance levels. McNemar assumes paired samples are independent, but CICIDS flows are temporally correlated — a single attack often produces many related flows in quick succession. The p-values support the model ranking as supporting evidence, but they are not strictly correct. A more rigorous analysis would use a dependency-aware test, which is outside the scope of this thesis.

**Q36. Your adversarial training failed — does this hurt the thesis?**
No, the negative result is part of the contribution. The naïve adversarial-training approach is the simplest defence anyone would try first, and showing concretely that it fails — with a 12.5-point drop in clean F1 and a 2.4-point increase in evasion rate — is informative for any practitioner who would otherwise reach for that defence. It is consistent with Tramer et al. 2020, who documented similar failures across many adversarial-training schemes. A successor researcher could try curriculum-scheduled PGD, certified robustness, or network-layer mitigations.

**Q37. Why not use a deep model — could it have done better?**
On bigger datasets, yes — deep models like TabNet, FT-Transformer, or LSTMs over packet inter-arrival times might match or exceed tree ensembles. But on the dataset sizes here, Grinsztajn et al. 2022 systematically showed that tree ensembles still win on tabular data, often by a clear margin, and they train in seconds rather than hours. A deep-learning comparison is the natural future work direction.

**Q38. What is the biggest threat to the validity of your conclusions?**
The dataset itself. Both CICIDS2017 and UNSW-NB15 are synthetic captures from controlled testbeds. If a determined examiner attacks anywhere, this is where. My thesis acknowledges this explicitly in Chapter 16, includes the cross-dataset validation as partial mitigation, and frames every conclusion as "under these benchmarks", not as a universal claim.

### Statistical / theoretical questions

**Q39. What is the sigmoid function?**
σ(z) = 1 / (1 + e⁻ᶻ). It maps any real number to a value between 0 and 1. Logistic regression uses it to convert a linear combination of features into a probability.

**Q40. What is Gini impurity?**
Gini = 1 − Σᵢ pᵢ², where pᵢ is the fraction of class i in a node. It measures how mixed the classes are. A pure node (all one class) has Gini = 0. Trees split to minimise the weighted Gini of the children.

**Q41. What is the F1 score formula?**
F1 = 2 · precision · recall / (precision + recall). It is the harmonic mean of the two. It punishes whichever is lower, so a model with perfect precision but zero recall has F1 = 0.

**Q42. What is bootstrap in one sentence?**
Resample the test set with replacement B times, compute the metric on each resample, and report the 2.5th and 97.5th percentile to get a 95% confidence interval.

**Q43. What does L∞ norm mean physically in your adversarial setting?**
The L∞ norm of a perturbation vector is the maximum absolute change across all dimensions. In my setting it is measured in units of benign-feature standard deviations σ. An L∞ budget of 0.25σ means no single feature changes by more than 0.25 of its training standard deviation. Other features may change by less.

---

## PART 5 — Curveball questions that might catch you off-guard

Prepare these specifically. They are the questions that separate a B from an A at DTU.

**Q44. If you had three more months, what would you do next?**
First, cross-dataset LODO — train on CICIDS days, test on UNSW splits — to test whether my temporal-generalisation claim holds across capture environments. Second, stronger adversarial training: curriculum-scheduled PGD with class rebalancing. Third, a learned MITRE mapping over richer telemetry — logs, EDR, identity signals — rather than the static lookup table I use.

**Q45. What surprised you most during this project?**
Three things. First, how concentrated the adversarial brittleness is — 83.6% of successful evasions move a single feature, Flow IAT Min. Second, how narrow the LODO ROC-AUC variance is for random forest at 0.926 ± 0.053. Third, how sharply the model ranking flips between CICIDS and UNSW: XGBoost wins one, LightGBM wins the other. Single-dataset benchmarks measure dataset characteristics as much as model quality.

**Q46. What is the one thing you would change if you could start over?**
I would align the CICFlowMeter and UNSW NetMate feature schemas at the start, so I could run cross-dataset LODO. That would have made the cross-dataset claim much stronger. Aligning the schemas in retrospect is non-trivial because the two extractors compute different statistics on the same flows.

**Q47. What is the single most important number in your thesis?**
The 0.4-point macro F1 drop between stratified and day-based protocols for tree ensembles. From 0.86 to 0.44. This is the size of the memorisation tax. It says the published numbers reflect in-distribution memorisation, not deployment readiness.

**Q48. Is this contribution publishable?**
The integration is at the level of a strong workshop paper or a security-focused conference short paper, especially the combination of honest temporal evaluation plus adversarial robustness plus operational triage. The individual components — temporal evaluation, transferability, adversarial training failure — are not novel in isolation. The combination, with a fully reproducible pipeline, is what makes it interesting.

**Q49. Why didn't you collaborate with anyone or look at related student projects at DTU?**
The project was scoped as an individual BSc thesis. The relevant prior work is in the literature, which I cite extensively in Chapters 2 and 15. A related student project could have provided extra hands for cross-dataset alignment or for live-traffic validation, but the scope is appropriate for 15 ECTS.

**Q50. If I read only one chapter, which one should I read?**
Chapter 10, Adversarial Robustness. The combination of evasion measurement, transferability, and the documented failure of naïve adversarial training is the strongest standalone contribution and the chapter most likely to interest a security audience. Chapter 6 is the second choice — the multi-class collapse is the headline finding and the figure is the cleanest illustration of why protocol matters.

---

## PART 6 — How to handle questions you cannot answer

The examiner is not testing whether you know everything. They are testing whether you can think under pressure.

If you do not know an answer:
1. **Do not bluff.** Examiners can detect bluffing instantly and it costs more than honest uncertainty.
2. **Acknowledge the gap.** Say: "That is a good question. I did not test this directly, but my best guess based on [related result] would be [hypothesis]."
3. **Reframe to what you do know.** Connect the question to a result you can defend.
4. **Offer the path forward.** "If I had access to [data / time / tool], I would test it by [procedure]."

Practise the phrase "I did not test this directly, but..." until it feels natural. It is much better than silence or a wrong answer.

---

## PART 7 — Last-week defence checklist

- [ ] Memorise the 30-second elevator pitch word-for-word.
- [ ] Read the 3-minute supervisor explanation out loud three times.
- [ ] Memorise the 25 numbers in Part 2.
- [ ] Know the 10 formulas in Part 3 — what they mean, not how to derive them.
- [ ] Practise Q1, Q2, Q5, Q9, Q13, Q15, Q19, Q21, Q23 out loud — these are the most likely opening questions.
- [ ] Print this document. Read it before the defence.
- [ ] Sleep before the defence — alert beats prepared.

---

*Last updated for defence on [enter date].*
*Thesis file: `thesis/overleaf_project.zip`. Pipeline: `make full`. Repository: configured locally.*
