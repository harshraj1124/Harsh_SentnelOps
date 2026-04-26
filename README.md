# Anomaly Detector — SentnelOps Assignment

This is my submission for the SentnelOps internship assignment. I built a hybrid anomaly detection system that flags infrastructure resources which look anomalous or inefficient, explains why, and suggests what to do about it.

## Running it

```bash
pip install -r requirements.txt
python anomaly_detector.py
```

Prints a summary to the console and writes full JSON output to `sample_outputs.json`.

---

## Approach

I considered three options before settling on the hybrid:

**Pure rule-based:** Fast, transparent, easy to explain. Works well for obvious cases like a zombie instance or a CPU-pegged server. The problem is you have to think of every case yourself — anything you didn't define gets missed.

**Isolation Forest (ML):** Looks at all metrics together and flags things that don't fit the pattern of the group. Useful for catching weird combinations where no single number is alarming but together they're suspicious — like low CPU with very high network traffic. Harder to explain on its own.

**LLM-based:** Best explanation quality, but non-deterministic and can't run offline. I think it's better as a post-processing layer for generating readable summaries than as your actual detection logic.

**What I went with:** Hybrid — rules for interpretability and known patterns, Isolation Forest for statistical coverage of the edge cases rules miss. Each output has a `reason` from the rule that fired, and the ML score acts as a corroborating signal that adjusts the confidence up or down.

Confidence formula:
```
# rules fired
confidence = 0.70 × max_rule_confidence + 0.30 × ml_score

# only ML flagged it (no rule fired)
confidence = 0.80 × ml_score

# healthy
confidence ≈ 0.05 – 0.12
```

I weighted rules at 70% because they're more interpretable and calibrated. The ML part is more of a supporting signal than a decision-maker.

---

## Anomaly types

| Type | What it means |
|---|---|
| `idle_zombie` | CPU and network both near zero — paying for a resource doing nothing |
| `over_provisioned` | Very low CPU/p95 consistently — instance is too large for the workload |
| `cpu_spike` | CPU p95 ≥ 95% — critically overloaded at peak |
| `high_cpu` | CPU avg sustained above 80% — under-provisioned |
| `memory_pressure` | Memory above 85% — OOM risk |
| `network_anomaly` | High network + low CPU — unusual combination, worth investigating |
| `statistical_outlier` | ML flagged it but no individual rule fired |
| `healthy` | Everything looks fine |

---

## Security detection

I added a security module because the `internet_facing` and `identity_attached` fields are too important to ignore:

- **internet-facing + identity attached = HIGH RISK.** If IMDSv2 isn't enforced, an SSRF vulnerability can pull IAM credentials through the metadata API. This is a well-known AWS attack vector.
- **internet-facing only = MEDIUM RISK.** Not inherently dangerous but worth verifying inbound rules.
- **internet-facing + high outbound network + low CPU = SUSPICIOUS.** Could be a legit data transfer job, but it's the pattern you'd see in an exfiltration scenario too.

---

## Test cases

I ran it on the 2 provided resources plus 5 I added to cover a range of scenarios:

| Resource | Scenario | Result | Confidence |
|---|---|---|---|
| i-1 | Low CPU, internet-facing + identity | over_provisioned | 0.74 |
| i-2 | CPU p95 at 98% | cpu_spike | 0.87 |
| i-3 | Balanced metrics | healthy | 0.10 |
| i-4 | Everything near zero | idle_zombie | 0.90 |
| i-5 | High CPU + memory + internet-facing + identity | cpu_spike | 0.93 |
| i-6 | Low CPU but network at 78%, internet-facing | network_anomaly | 0.78 |
| i-7 | Balanced metrics | healthy | 0.09 |

Full output in `sample_outputs.json`.

---

## Tradeoffs and known limitations

**Isolation Forest needs batch context.** The ML scores compare each resource to its peers. If you send one resource at a time, there's no reference distribution and the score is meaningless. For single-resource analysis you'd fall back entirely on rules, which is fine but you lose the statistical signal.

**Thresholds are hand-tuned.** I picked values that made sense given typical cloud workload behavior (e.g. a 95% p95 CPU is almost always a problem), but in production these should be learned from each resource type's historical baseline — not hardcoded.

**Point-in-time data only.** A CPU spike that lasts 30 seconds is very different from one that's been sustained for 3 hours. Right now I can't distinguish between them because I'm working with snapshots, not time series.

**Missing fields default to 0.** The pipeline won't crash, but silent defaults can cause false negatives — a missing `cpu_p95` means the spike rule never fires. Production code should track data quality explicitly.

---

## What I'd improve with more time

**Time-series detection** — run the rules over rolling windows (e.g. 1h, 6h, 24h) so "sustained anomaly" is different from "brief spike." This is probably the most impactful change.

**Per-resource baselines** — instead of fixed thresholds, compute each resource's own rolling average and flag deviations from *its own* history. A bursty service and a steady-state service look very different and shouldn't share the same thresholds.

**Online ML** — replace batch Isolation Forest with something like `river`'s HalfSpaceTrees so you can do real-time detection without needing a full batch of peers to fit on.

**LLM explanation layer** — keep the rule + ML logic as-is for detection, but pass the signals to an LLM to generate better natural-language explanations for non-technical stakeholders. Detection and explanation are two different jobs.

---

## Project structure

```
anomaly_detector.py   main script, all logic
README.md             this file
requirements.txt      scikit-learn, numpy
sample_outputs.json   outputs from the 7 test cases
```
