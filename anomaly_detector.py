#!/usr/bin/env python3
"""
SentnelOps Internship Assignment — AI/ML Anomaly Detection
Approach : Hybrid (Rule-Based + Isolation Forest)

Run:
    pip install -r requirements.txt
    python anomaly_detector.py
"""

import json
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Optional, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all tunable thresholds in one place
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "cpu_idle"                 : 3,    # % — below this + low network = zombie
    "cpu_low"                  : 10,   # % — over-provisioned signal
    "cpu_high"                 : 80,   # % — sustained high load
    "cpu_critical"             : 95,   # % — p95 spike
    "memory_high"              : 85,   # % — memory pressure
    "network_high"             : 70,   # % — high throughput
    "network_suspicious_cpu"   : 30,   # % — cpu cap; above = normal burst
    "ml_anomaly_threshold"     : 0.65, # normalized score; above = flag it
}

# Severity ranking: determines primary anomaly type when multiple signals fire
SEVERITY = {
    "cpu_spike"          : 5,
    "idle_zombie"        : 4,
    "network_anomaly"    : 3,
    "memory_pressure"    : 3,
    "high_cpu"           : 2,
    "over_provisioned"   : 1,
}

ACTION_MAP = {
    "idle_zombie"        : "Terminate or schedule decommission review — instance is idle and incurring cost",
    "over_provisioned"   : "Downsize to a smaller instance type; review 7–30 day usage patterns to right-size",
    "cpu_spike"          : "Scale up or distribute load; profile workload to identify CPU hotspots",
    "high_cpu"           : "Monitor closely; consider vertical scaling or load distribution",
    "memory_pressure"    : "Increase memory allocation or investigate for memory leaks and unbounded caches",
    "network_anomaly"    : "Audit network flows immediately; check for unauthorized data transfer or misconfigured services",
    "statistical_outlier": "Manually review — ML flagged an unusual metric combination not matched by any single rule",
    "healthy"            : "No action required; continue standard monitoring",
}


# ─────────────────────────────────────────────────────────────────────────────
# Test Dataset  (2 from assignment + 5 extended cases)
# ─────────────────────────────────────────────────────────────────────────────

TEST_RESOURCES: List[Dict[str, Any]] = [
    # ── From assignment ───────────────────────────────────────────────────────
    {
        "resource_id": "i-1",
        "cpu_avg": 2, "cpu_p95": 5, "memory_avg": 70, "network_pct": 10,
        "internet_facing": True, "identity_attached": True,
        # Expected: over_provisioned + HIGH security risk
    },
    {
        "resource_id": "i-2",
        "cpu_avg": 85, "cpu_p95": 98, "memory_avg": 40, "network_pct": 60,
        "internet_facing": False, "identity_attached": False,
        # Expected: cpu_spike
    },
    # ── Extended test cases ───────────────────────────────────────────────────
    {
        "resource_id": "i-3",
        "cpu_avg": 45, "cpu_p95": 65, "memory_avg": 55, "network_pct": 30,
        "internet_facing": False, "identity_attached": True,
        # Expected: healthy — all metrics balanced
    },
    {
        "resource_id": "i-4",
        "cpu_avg": 1, "cpu_p95": 2, "memory_avg": 5, "network_pct": 1,
        "internet_facing": False, "identity_attached": False,
        # Expected: idle_zombie — everything near zero
    },
    {
        "resource_id": "i-5",
        "cpu_avg": 78, "cpu_p95": 96, "memory_avg": 92, "network_pct": 85,
        "internet_facing": True, "identity_attached": True,
        # Expected: cpu_spike + memory_pressure, critical security risk
    },
    {
        "resource_id": "i-6",
        "cpu_avg": 8, "cpu_p95": 12, "memory_avg": 30, "network_pct": 78,
        "internet_facing": True, "identity_attached": False,
        # Expected: network_anomaly — high outbound, low CPU, internet-facing
    },
    {
        "resource_id": "i-7",
        "cpu_avg": 50, "cpu_p95": 70, "memory_avg": 60, "network_pct": 40,
        "internet_facing": False, "identity_attached": False,
        # Expected: healthy
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Component 1 — Rule-Based Detector
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_analyze(resource: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Applies threshold-based rules to a single resource.

    Returns a list of (anomaly_type, confidence, human_readable_reason) tuples.
    Multiple signals can fire simultaneously (e.g. cpu_spike + memory_pressure).

    Design choices:
    - idle_zombie and over_provisioned are mutually exclusive (elif) because
      a near-zero-CPU instance can only be one or the other.
    - cpu_spike and high_cpu are mutually exclusive (elif) to avoid redundancy.
    - memory_pressure and network_anomaly are independent checks.
    """
    T           = THRESHOLDS
    cpu_avg     = resource.get("cpu_avg", 0)
    cpu_p95     = resource.get("cpu_p95", 0)
    memory_avg  = resource.get("memory_avg", 0)
    network_pct = resource.get("network_pct", 0)

    signals: List[Tuple[str, float, str]] = []

    # ── 1. Idle / Zombie ──────────────────────────────────────────────────────
    if cpu_avg <= T["cpu_idle"] and network_pct <= 5:
        signals.append((
            "idle_zombie", 0.90,
            f"CPU avg {cpu_avg}% and network {network_pct}% are both near zero — "
            f"instance appears completely idle or is a zombie process"
        ))

    # ── 2. Over-provisioned ───────────────────────────────────────────────────
    elif cpu_avg < T["cpu_low"] and cpu_p95 < 20:
        signals.append((
            "over_provisioned", 0.82,
            f"CPU avg {cpu_avg}% and p95 {cpu_p95}% are consistently very low — "
            f"resource is significantly over-provisioned relative to actual workload"
        ))

    # ── 3. CPU Spike (p95 is the better signal than avg for bursting) ─────────
    if cpu_p95 >= T["cpu_critical"]:
        signals.append((
            "cpu_spike", 0.92,
            f"CPU p95 at {cpu_p95}% — instance is critically overloaded at peak; "
            f"risk of latency spikes, request queuing, and OOM"
        ))
    elif cpu_avg >= T["cpu_high"]:
        signals.append((
            "high_cpu", 0.78,
            f"CPU avg {cpu_avg}% is consistently high — sustained under-provisioning risk"
        ))

    # ── 4. Memory Pressure ────────────────────────────────────────────────────
    if memory_avg >= T["memory_high"]:
        signals.append((
            "memory_pressure", 0.75,
            f"Memory avg {memory_avg}% is critically high — "
            f"risk of OOM kills, swap thrashing, or memory leak"
        ))

    # ── 5. Network Anomaly: high bytes with low CPU ───────────────────────────
    if network_pct >= T["network_high"] and cpu_avg < T["network_suspicious_cpu"]:
        signals.append((
            "network_anomaly", 0.80,
            f"Network utilization {network_pct}% is high while CPU is only {cpu_avg}% — "
            f"disproportionate network activity may indicate data exfiltration, "
            f"rogue process, or misconfigured service"
        ))

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Component 2 — ML Detector (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────

def ml_anomaly_scores(resources: List[Dict[str, Any]]) -> List[float]:
    """
    Fits an Isolation Forest on the full resource batch and returns a
    per-resource anomaly score normalized to [0, 1] where 1 = most anomalous.

    Why Isolation Forest?
    ─ No labeled data required — ideal for unsupervised infra monitoring
    ─ Isolates outliers by how few splits it takes to isolate a point
    ─ Handles multi-dimensional outliers rules might miss (e.g. unusual
      combination of moderate metrics that together look abnormal)
    ─ Fast, interpretable, industry-standard for anomaly detection

    Limitation: needs a reasonably sized batch for meaningful context.
    With very small batches (< 5 resources), interpret ML scores loosely.
    """
    features = ["cpu_avg", "cpu_p95", "memory_avg", "network_pct"]
    X = np.array(
        [[r.get(f, 0) for f in features] for r in resources],
        dtype=float
    )

    model = IsolationForest(
        contamination=0.3,   # ~30% of resources expected to be anomalous
        n_estimators=200,
        random_state=42      # reproducible
    )
    model.fit(X)

    # score_samples: more negative = more anomalous
    raw = model.score_samples(X)

    lo, hi = raw.min(), raw.max()
    if hi == lo:
        return [0.5] * len(resources)

    # Normalize: anomalous → 1, normal → 0
    return [float((hi - s) / (hi - lo)) for s in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Component 3 — Security Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def security_analyze(resource: Dict[str, Any]) -> Optional[str]:
    """
    Evaluates security risk from exposure metadata and behavioral signals.

    Risk model:
    ─ internet_facing + identity_attached = HIGH RISK (SSRF/credential exposure)
    ─ internet_facing alone              = MEDIUM RISK (verify inbound rules)
    ─ internet_facing + high network + low CPU = potential data exfiltration
    """
    internet_facing   = resource.get("internet_facing", False)
    identity_attached = resource.get("identity_attached", False)
    network_pct       = resource.get("network_pct", 0)
    cpu_avg           = resource.get("cpu_avg", 0)

    notes: List[str] = []

    if internet_facing and identity_attached:
        notes.append(
            "HIGH RISK: Internet-facing with IAM/identity attached — "
            "SSRF or metadata API abuse (e.g. IMDSv1) could expose cloud credentials"
        )
    elif internet_facing:
        notes.append(
            "MEDIUM RISK: Internet-facing instance — "
            "verify security groups and restrict inbound rules to minimum required"
        )

    if internet_facing and network_pct > 50 and cpu_avg < 30:
        notes.append(
            "SUSPICIOUS: High outbound network with low CPU on internet-facing resource — "
            "possible data exfiltration vector; recommend VPC flow log analysis"
        )

    return " | ".join(notes) if notes else None


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Detector — Combines all components
# ─────────────────────────────────────────────────────────────────────────────

def analyze_resources(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hybrid analysis pipeline:
        1. Rule-based scan  → explicit signals with individual confidences
        2. Isolation Forest → statistical anomaly score across all resources
        3. Security scan    → risk flags from exposure metadata
        4. Merge + score    → weighted confidence, primary anomaly type

    Confidence formula:
        Rules fired   :  0.70 × max(rule_confidence) + 0.30 × ml_score
        ML-only flag  :  0.80 × ml_score  (lower weight — no rule support)
        Healthy       :  small value reflecting certainty of normalcy

    The 70/30 split favors interpretable rules while letting ML bump confidence
    when the statistical signal agrees, and catch things rules miss entirely.
    """
    ml_scores = ml_anomaly_scores(resources)
    results: List[Dict[str, Any]] = []

    for resource, ml_score in zip(resources, ml_scores):
        signals  = rule_based_analyze(resource)
        sec_note = security_analyze(resource)

        # ── Determine primary anomaly type ────────────────────────────────────
        if signals:
            is_anomalous = True
            # Pick the highest-severity signal as primary type
            primary_type = max(signals, key=lambda s: SEVERITY.get(s[0], 0))[0]
        elif ml_score > THRESHOLDS["ml_anomaly_threshold"]:
            is_anomalous = True
            primary_type = "statistical_outlier"
        else:
            is_anomalous = False
            primary_type = "healthy"

        # ── Confidence ────────────────────────────────────────────────────────
        if signals:
            rule_conf  = max(s[1] for s in signals)
            confidence = round(0.70 * rule_conf + 0.30 * ml_score, 2)
        elif is_anomalous:
            confidence = round(0.80 * ml_score, 2)
        else:
            # Healthy: low score; closer to 0 means "very likely healthy"
            confidence = round(max(0.05, (1.0 - ml_score) * 0.12), 2)

        # ── Build reason string ───────────────────────────────────────────────
        if signals:
            # Sort by severity descending so the most important signal leads
            sorted_sigs = sorted(signals, key=lambda s: SEVERITY.get(s[0], 0), reverse=True)
            reason = ". ".join(s[2] for s in sorted_sigs)
        elif is_anomalous:
            reason = (
                f"No individual metric breached a rule threshold, but Isolation Forest "
                f"identified this resource as a statistical outlier (ML score: {ml_score:.2f}). "
                f"The combination of metrics is unusual relative to the peer group."
            )
        else:
            reason = "All metrics are within normal operating ranges for this resource group."

        results.append({
            "resource_id"     : resource["resource_id"],
            "is_anomalous"    : is_anomalous,
            "anomaly_type"    : primary_type,
            "reason"          : reason,
            "suggested_action": ACTION_MAP[primary_type],
            "confidence"      : confidence,
            "security_note"   : sec_note,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Console Output + JSON Export
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("  SentnelOps — Infrastructure Anomaly Detection Report")
    print("  Approach: Hybrid (Rule-Based + Isolation Forest)")
    print("=" * 70)

    for r in results:
        status = "ANOMALOUS" if r["is_anomalous"] else "HEALTHY  "
        print(f"\n[{status}]  {r['resource_id']}  —  {r['anomaly_type'].upper()}")
        print(f"  Confidence : {r['confidence']}")
        print(f"  Reason     : {r['reason'][:120]}" + ("..." if len(r['reason']) > 120 else ""))
        print(f"  Action     : {r['suggested_action']}")
        if r.get("security_note"):
            note = r["security_note"]
            print(f"  Security   : {note[:110]}" + ("..." if len(note) > 110 else ""))

    print("\n\n" + "=" * 70)
    print("  FULL JSON OUTPUT")
    print("=" * 70 + "\n")
    print(json.dumps(results, indent=2))


def main() -> None:
    results = analyze_resources(TEST_RESOURCES)
    print_results(results)

    with open("sample_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Saved] sample_outputs.json")


if __name__ == "__main__":
    main()
