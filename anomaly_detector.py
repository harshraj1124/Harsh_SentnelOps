#!/usr/bin/env python3
"""
anomaly_detector.py

My solution for the SentnelOps internship assignment.
Hybrid approach: rule-based thresholds + Isolation Forest + security checks.

Usage:
    pip install -r requirements.txt
    python anomaly_detector.py
"""

import json
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Optional, Dict, Any, Tuple


# thresholds are hand-tuned, not learned. in production you'd want
# per-resource-type baselines derived from historical data
THRESHOLDS = {
    "cpu_idle":        3,    # below this + low network = zombie
    "cpu_low":         10,   # below this = over-provisioned
    "cpu_high":        80,   # above this = under-provisioned
    "cpu_critical":    95,   # p95 at this level = basically on fire
    "memory_high":     85,
    "network_high":    70,
    "net_suspect_cpu": 30,   # high network only suspicious when cpu is also low
    "ml_threshold":    0.65,
}

# when multiple signals fire on the same resource, pick the highest-severity
# one as the primary anomaly type for the output
SEVERITY = {
    "cpu_spike":       5,
    "idle_zombie":     4,
    "network_anomaly": 3,
    "memory_pressure": 3,
    "high_cpu":        2,
    "over_provisioned":1,
}

ACTIONS = {
    "idle_zombie":         "Terminate or schedule for review — this instance is doing nothing and costing money",
    "over_provisioned":    "Downsize the instance type; check 7-30 day usage to right-size",
    "cpu_spike":           "Scale up or distribute load; profile what's consuming CPU at peak",
    "high_cpu":            "Monitor closely, consider scaling up or spreading the load",
    "memory_pressure":     "Increase memory or investigate for leaks — OOM risk is real here",
    "network_anomaly":     "Audit traffic flows — high outbound with low CPU is worth investigating",
    "statistical_outlier": "Worth a manual look — ML flagged this but no single rule fired",
    "healthy":             "Nothing to do here",
}


# 7 test cases: 2 from the assignment + 5 I added to cover edge cases
TEST_DATA = [
    # from the assignment
    {
        "resource_id": "i-1",
        "cpu_avg": 2, "cpu_p95": 5, "memory_avg": 70, "network_pct": 10,
        "internet_facing": True, "identity_attached": True
    },
    {
        "resource_id": "i-2",
        "cpu_avg": 85, "cpu_p95": 98, "memory_avg": 40, "network_pct": 60,
        "internet_facing": False, "identity_attached": False
    },
    # healthy, well-balanced instance
    {
        "resource_id": "i-3",
        "cpu_avg": 45, "cpu_p95": 65, "memory_avg": 55, "network_pct": 30,
        "internet_facing": False, "identity_attached": True
    },
    # complete zombie — all metrics near zero
    {
        "resource_id": "i-4",
        "cpu_avg": 1, "cpu_p95": 2, "memory_avg": 5, "network_pct": 1,
        "internet_facing": False, "identity_attached": False
    },
    # worst case: high everything + internet-facing + identity attached
    {
        "resource_id": "i-5",
        "cpu_avg": 78, "cpu_p95": 96, "memory_avg": 92, "network_pct": 85,
        "internet_facing": True, "identity_attached": True
    },
    # interesting one: low cpu but high network — possible exfil or just a transfer job
    {
        "resource_id": "i-6",
        "cpu_avg": 8, "cpu_p95": 12, "memory_avg": 30, "network_pct": 78,
        "internet_facing": True, "identity_attached": False
    },
    # another healthy instance
    {
        "resource_id": "i-7",
        "cpu_avg": 50, "cpu_p95": 70, "memory_avg": 60, "network_pct": 40,
        "internet_facing": False, "identity_attached": False
    },
]


def check_rules(r: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Runs threshold checks on a single resource.
    Returns list of (anomaly_type, confidence, reason) — multiple can fire at once.
    """
    cpu  = r.get("cpu_avg", 0)
    p95  = r.get("cpu_p95", 0)
    mem  = r.get("memory_avg", 0)
    net  = r.get("network_pct", 0)
    T    = THRESHOLDS
    hits = []

    # zombie: both cpu and network near zero — instance is probably just sitting there
    if cpu <= T["cpu_idle"] and net <= 5:
        hits.append((
            "idle_zombie", 0.90,
            f"CPU avg {cpu}% and network {net}% are both near zero — instance looks completely idle"
        ))
    # over-provisioned: low cpu consistently, not just a quiet moment
    elif cpu < T["cpu_low"] and p95 < 20:
        hits.append((
            "over_provisioned", 0.82,
            f"CPU avg {cpu}%, p95 {p95}% — consistently underutilized, instance is too big for the workload"
        ))

    # using p95 for spike detection rather than avg — avg can mask short bursts
    if p95 >= T["cpu_critical"]:
        hits.append((
            "cpu_spike", 0.92,
            f"CPU p95 at {p95}% — hitting critical load at peak, latency and OOM risk are real"
        ))
    elif cpu >= T["cpu_high"]:
        hits.append((
            "high_cpu", 0.78,
            f"CPU avg {cpu}% is consistently high — under-provisioned for this workload"
        ))

    if mem >= T["memory_high"]:
        hits.append((
            "memory_pressure", 0.75,
            f"Memory at {mem}% — getting dangerous, check for leaks or unbounded caches"
        ))

    # high network with low cpu is the one I find most interesting —
    # could be a legit data transfer job, but on an internet-facing machine it's suspicious
    if net >= T["network_high"] and cpu < T["net_suspect_cpu"]:
        hits.append((
            "network_anomaly", 0.80,
            f"Network at {net}% while CPU is only {cpu}% — disproportionate, worth investigating"
        ))

    return hits


def get_ml_scores(resources: List[Dict[str, Any]]) -> List[float]:
    """
    Fits Isolation Forest on the batch and returns anomaly scores in [0, 1].
    Higher = more anomalous.

    I went with Isolation Forest because it doesn't need labeled data,
    handles multi-dimensional outliers well, and is fast enough for this use case.
    The main weakness here is batch size — with only a handful of resources,
    these scores are more like "relative weirdness" than hard anomaly calls.
    """
    cols = ["cpu_avg", "cpu_p95", "memory_avg", "network_pct"]
    X = np.array([[r.get(c, 0) for c in cols] for r in resources], dtype=float)

    model = IsolationForest(contamination=0.3, n_estimators=200, random_state=42)
    model.fit(X)
    raw = model.score_samples(X)  # more negative = more anomalous

    lo, hi = raw.min(), raw.max()
    if lo == hi:
        return [0.5] * len(resources)

    # flip and normalize so 1.0 = most anomalous
    return [float((hi - s) / (hi - lo)) for s in raw]


def check_security(r: Dict[str, Any]) -> Optional[str]:
    """Flags security concerns based on exposure metadata."""
    inet   = r.get("internet_facing", False)
    has_id = r.get("identity_attached", False)
    net    = r.get("network_pct", 0)
    cpu    = r.get("cpu_avg", 0)

    notes = []

    if inet and has_id:
        # this combo is the dangerous one — if IMDSv2 isn't enforced,
        # an SSRF can drain your IAM credentials through the metadata API
        notes.append(
            "HIGH RISK: internet-facing with identity attached — "
            "SSRF or metadata API abuse could expose cloud credentials"
        )
    elif inet:
        notes.append(
            "MEDIUM RISK: internet-facing — verify security group inbound rules"
        )

    if inet and net > 50 and cpu < 30:
        notes.append(
            "SUSPICIOUS: high outbound traffic + low CPU on internet-facing resource — "
            "possible exfiltration, recommend reviewing VPC flow logs"
        )

    return " | ".join(notes) if notes else None


def analyze(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs all three components and merges into final output."""
    ml_scores = get_ml_scores(resources)
    results = []

    for resource, ml_score in zip(resources, ml_scores):
        rule_hits = check_rules(resource)
        sec_note  = check_security(resource)

        if rule_hits:
            is_anomalous = True
            primary = max(rule_hits, key=lambda x: SEVERITY.get(x[0], 0))[0]
        elif ml_score > THRESHOLDS["ml_threshold"]:
            is_anomalous = True
            primary = "statistical_outlier"
        else:
            is_anomalous = False
            primary = "healthy"

        # blend rule + ml confidence, weighted toward rules since they're more explainable
        if rule_hits:
            rule_conf  = max(h[1] for h in rule_hits)
            confidence = round(0.70 * rule_conf + 0.30 * ml_score, 2)
        elif is_anomalous:
            confidence = round(0.80 * ml_score, 2)
        else:
            confidence = round(max(0.05, (1.0 - ml_score) * 0.12), 2)

        # build reason — most severe signal leads
        if rule_hits:
            sorted_hits = sorted(rule_hits, key=lambda x: SEVERITY.get(x[0], 0), reverse=True)
            reason = ". ".join(h[2] for h in sorted_hits)
        elif is_anomalous:
            reason = (
                f"No single threshold was breached, but Isolation Forest flagged this as "
                f"a statistical outlier (score: {ml_score:.2f}). "
                f"The metric combination is unusual relative to the rest of the group."
            )
        else:
            reason = "All metrics look normal."

        results.append({
            "resource_id":      resource["resource_id"],
            "is_anomalous":     is_anomalous,
            "anomaly_type":     primary,
            "reason":           reason,
            "suggested_action": ACTIONS[primary],
            "confidence":       confidence,
            "security_note":    sec_note,
        })

    return results


def main():
    results = analyze(TEST_DATA)

    for r in results:
        tag = "ANOMALOUS" if r["is_anomalous"] else "healthy  "
        print(f"[{tag}] {r['resource_id']:5s}  {r['anomaly_type']:<22}  conf={r['confidence']}")
        if r.get("security_note"):
            print(f"         sec: {r['security_note'][:90]}...")

    print("\n--- full JSON output ---\n")
    print(json.dumps(results, indent=2))

    with open("sample_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to sample_outputs.json")


if __name__ == "__main__":
    main()
