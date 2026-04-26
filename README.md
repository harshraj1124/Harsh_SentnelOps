# SentnelOps Internship Assignment — Anomaly Detection

## Quick Start

```bash
pip install -r requirements.txt
python anomaly_detector.py
```

Outputs results to console and writes `sample_outputs.json`.

---

## Approach: Hybrid (Rule-Based + Isolation Forest)

### Why Hybrid?

I evaluated three approaches before deciding:

| Approach | Pros | Cons | Fit for this task |
|---|---|---|---|
| **Rule-Based** | Transparent, fast, deterministic, easy to explain | Misses multi-dimensional outliers; threshold tuning is manual | Good for clear-cut cases |
| **Isolation Forest (ML)** | Catches unusual metric combinations no single rule would flag; no labeled data needed | Less interpretable on its own; needs batch context | Good for statistical outliers |
| **LLM-Based Reasoning** | Highest explanation quality; handles ambiguity well | Non-deterministic, slow, expensive, no offline guarantee | Better for report generation, not real-time detection |

**The hybrid approach is the right choice here because:**

1. Infrastructure anomalies often fall into well-understood categories (zombie, over-provisioned, CPU spike) that rules handle perfectly with high confidence.
2. Some anomalies are subtle — a moderate CPU + moderate memory + high network on an internet-facing resource may not breach any single threshold but is statistically unusual. ML catches these.
3. The assignment specifically values *explainability*. Rules produce crisp human-readable reasons. ML adds a supporting statistical signal.
4. It avoids over-engineering while still being more robust than rules alone.

---

## Architecture

```
Input JSON
    │
    ├──► Rule-Based Detector     ──► signals: [(type, confidence, reason), ...]
    │        (per-resource)
    │
    ├──► Isolation Forest        ──► ml_score: [0–1] per resource
    │        (batch context)          (higher = more anomalous)
    │
    └──► Security Analyzer       ──► security_note: str | null
             (per-resource)
                    │
                    ▼
             Hybrid Merger
         ─────────────────────
         Primary type   = highest-severity signal
         Confidence     = 0.70 × rule_conf + 0.30 × ml_score
         Reason         = sorted signals, most severe first
                    │
                    ▼
             JSON Output
```

---

## How Each Component Works

### Rule-Based Detector

Five independent rule checks, each producing a typed signal with a pre-calibrated confidence:

| Rule | Condition | Type | Confidence |
|---|---|---|---|
| Idle/Zombie | CPU avg ≤ 3% AND network ≤ 5% | `idle_zombie` | 0.90 |
| Over-provisioned | CPU avg < 10% AND CPU p95 < 20% | `over_provisioned` | 0.82 |
| CPU spike | CPU p95 ≥ 95% | `cpu_spike` | 0.92 |
| High CPU | CPU avg ≥ 80% | `high_cpu` | 0.78 |
| Memory pressure | Memory avg ≥ 85% | `memory_pressure` | 0.75 |
| Network anomaly | Network ≥ 70% AND CPU < 30% | `network_anomaly` | 0.80 |

Multiple signals can fire simultaneously (e.g. `cpu_spike` + `memory_pressure`).
The primary anomaly type is the highest-severity signal.

### Isolation Forest

Isolation Forest works by randomly partitioning the feature space and counting how many splits it takes to isolate a data point. Anomalous points are isolated quickly (few splits); normal points require many. The raw score is a negative float — more negative means more anomalous.

I normalize it to `[0, 1]` where `1 = most anomalous` and use it as a supporting signal. With `contamination=0.3`, the model expects roughly 30% of the batch to be anomalous.

**Features used:** `cpu_avg`, `cpu_p95`, `memory_avg`, `network_pct`

### Confidence Scoring

```
if rule signals fired:
    confidence = 0.70 × max(rule_confidence) + 0.30 × ml_score

elif ml_score > 0.65:
    confidence = 0.80 × ml_score   # ML-only: lower weight

else (healthy):
    confidence = small value ≈ 0.05–0.12
```

The 70/30 split was chosen deliberately: rules are more interpretable and calibrated, so they dominate. ML acts as a corroborating signal that bumps confidence when it agrees — or as a lone detector when rules don't fire.

### Security Analyzer

Maps exposure flags to risk levels:

| Condition | Risk Level |
|---|---|
| `internet_facing=true` + `identity_attached=true` | HIGH — SSRF / credential exposure via metadata API |
| `internet_facing=true` only | MEDIUM — verify security groups |
| Internet-facing + network > 50% + CPU < 30% | SUSPICIOUS — possible data exfiltration |

---

## Anomaly Type Reference

| Type | What it means | Action |
|---|---|---|
| `idle_zombie` | CPU and network both near zero; instance consuming resources doing nothing | Terminate or review for decommission |
| `over_provisioned` | Very low CPU/p95; paying for capacity that isn't used | Downsize instance type |
| `cpu_spike` | CPU p95 at critical levels; system is overwhelmed at peak | Scale up or fix the workload |
| `high_cpu` | Sustained high average CPU | Monitor; consider scaling |
| `memory_pressure` | Memory critically high | Increase allocation or fix leak |
| `network_anomaly` | High network + low CPU (unusual combination) | Audit traffic flows |
| `statistical_outlier` | ML flags unusual metric combination; no single rule fired | Manual review |
| `healthy` | All metrics within normal ranges | No action |

---

## Sample Outputs (7 Test Cases)

| Resource | Anomalous | Type | Confidence | Security |
|---|---|---|---|---|
| i-1 | ✓ | over_provisioned | 0.74 | HIGH RISK |
| i-2 | ✓ | cpu_spike | 0.87 | — |
| i-3 | ✗ | healthy | 0.10 | — |
| i-4 | ✓ | idle_zombie | 0.90 | — |
| i-5 | ✓ | cpu_spike + memory_pressure | 0.93 | HIGH RISK |
| i-6 | ✓ | network_anomaly | 0.78 | MEDIUM + SUSPICIOUS |
| i-7 | ✗ | healthy | 0.09 | — |

See `sample_outputs.json` for full JSON.

---

## Handling Ambiguity

Real infrastructure data is imperfect. Here is how the system handles each case:

**Missing fields:** `resource.get("cpu_avg", 0)` — defaults to 0, so the system degrades gracefully rather than crashing. A missing metric reduces the accuracy of ML scoring but does not break the pipeline.

**Borderline metrics (e.g. cpu_avg = 11%, just above the "low" threshold):** The ML score acts as a tiebreaker. A borderline rule non-trigger with a high ML score will still surface as a `statistical_outlier` with moderate confidence.

**Small batches:** Isolation Forest needs a reference distribution to work meaningfully. With fewer than 5 resources, the ML scores should be treated as supplementary only. The rule-based signals are still reliable.

**Conflicting signals:** When `over_provisioned` and `network_anomaly` both fire on the same resource, the severity ranking resolves the primary type (`network_anomaly` wins at severity 3 vs 1). The reason string includes all signals, most important first.

---

## Approach Comparison (Bonus)

### Rule-Based vs ML vs LLM

```
                    Rule-Based    Isolation Forest    LLM (GPT/Claude)
─────────────────────────────────────────────────────────────────────
Interpretability        ★★★★★            ★★★               ★★★★
Speed                   ★★★★★            ★★★★              ★★
Cost                    Free             Free              API cost
Needs labels            No               No                No
Catches subtle outliers ★★               ★★★★★             ★★★★
Deterministic           Yes              Yes (seed)        No
Handles ambiguity       Poor             Moderate          Excellent
Best for                Clear thresholds Statistical noise Explanation gen.
```

**My recommendation for production:**
- Use rules for known anomaly types with clear thresholds (fast, auditable)
- Use Isolation Forest for catching the long tail of unusual combinations
- Use an LLM as a post-hoc explanation layer only (not in the detection path)

This assignment's hybrid approach reflects that recommendation.

---

## Tradeoffs

**What I chose and why:**
- Hybrid over pure ML: explainability matters more than marginal accuracy gains in ops contexts
- Isolation Forest over supervised ML: no labeled anomaly data exists (realistic constraint)
- Flat script over microservices: the problem is a script-level task; over-architecting would obscure the reasoning

**Known limitations:**
- Isolation Forest needs a meaningful batch size; single-resource analysis loses statistical context
- Thresholds are hand-tuned; in production these should be learned from historical baselines
- `contamination=0.3` is a guess; in practice you'd tune this against known anomaly rates
- No time-series context: a CPU spike that lasts 5 minutes is different from one that lasts 3 hours — this system can't distinguish them from snapshot data

---

## What I Would Improve With More Time

1. **Time-series integration** — Run detection over rolling windows (e.g. 1h, 24h) rather than point-in-time snapshots. Most real anomalies are patterns, not instants.

2. **Learned baselines** — Instead of fixed thresholds, compute per-resource moving averages and flag deviations from the resource's own history (z-score or MAD-based).

3. **Severity scoring** — Add a composite severity score (not just anomaly type) that accounts for criticality of the resource (production vs dev) and blast radius.

4. **LLM explanation layer** — Feed the rule signals and ML score into a small LLM call to generate better natural-language explanations for non-technical stakeholders.

5. **Feedback loop** — Let operators mark detections as false positives; use that signal to adjust thresholds automatically over time.

6. **Streaming support** — Replace batch Isolation Forest with an online variant (e.g. `river` library's `HalfSpaceTrees`) for real-time processing.

---

## Project Structure

```
.
├── anomaly_detector.py   # Main script — all logic, runnable
├── README.md             # This file
├── requirements.txt      # Dependencies (scikit-learn, numpy)
└── sample_outputs.json   # Pre-computed outputs for 7 test cases
```

---

## Why This Approach Fits SentnelOps

> "At SentnelOps, we build systems where AI doesn't just predict — it explains and helps decide what should happen next."

This is exactly what the hybrid approach delivers:
- The **rule engine** provides the *explanation* (clear, auditable, deterministic)
- The **ML layer** provides the *prediction* (catches what rules miss)
- The **security analyzer** provides the *decision context* (what to do next)
- Every output includes a `suggested_action` — the system doesn't just flag, it recommends

The architecture is also intentionally extensible: adding a new anomaly type means adding one rule block and one entry in `ACTION_MAP`.
