import streamlit as st
import pandas as pd
import json
from anomaly_detector import analyze, TEST_DATA

st.set_page_config(
    page_title="Infrastructure Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar: inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Resource Metrics")
    st.caption("Adjust sliders to analyze a resource in real time")
    st.markdown("---")

    resource_id = st.text_input("Resource ID", value="i-custom", placeholder="e.g. i-abc123")

    st.markdown("**Compute**")
    cpu_avg    = st.slider("CPU Average (%)",       0, 100, 45)
    cpu_p95    = st.slider("CPU p95 (%)",           0, 100, 65,
                           help="95th percentile — better for catching spikes than the average")
    memory_avg = st.slider("Memory Average (%)",    0, 100, 55)
    network_pct= st.slider("Network Utilization (%)", 0, 100, 30)

    st.markdown("**Exposure**")
    internet_facing   = st.toggle("Internet Facing",          value=False)
    identity_attached = st.toggle("Identity / IAM Attached",  value=False)

    st.markdown("---")
    st.markdown("**Quick presets**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Zombie",     use_container_width=True):
            st.session_state.preset = "zombie"
    with col_b:
        if st.button("CPU Spike",  use_container_width=True):
            st.session_state.preset = "spike"
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Healthy",    use_container_width=True):
            st.session_state.preset = "healthy"
    with col_d:
        if st.button("Exfil Risk", use_container_width=True):
            st.session_state.preset = "exfil"

# Apply presets via session state
preset = st.session_state.get("preset")
if preset == "zombie":
    cpu_avg = 1;  cpu_p95 = 2;  memory_avg = 5;  network_pct = 1
    internet_facing = False; identity_attached = False
elif preset == "spike":
    cpu_avg = 88; cpu_p95 = 97; memory_avg = 40; network_pct = 55
    internet_facing = False; identity_attached = False
elif preset == "healthy":
    cpu_avg = 45; cpu_p95 = 62; memory_avg = 55; network_pct = 28
    internet_facing = False; identity_attached = True
elif preset == "exfil":
    cpu_avg = 7;  cpu_p95 = 11; memory_avg = 30; network_pct = 80
    internet_facing = True;  identity_attached = False

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Infrastructure Anomaly Detector")
st.caption(
    "Hybrid detection — rule-based thresholds + Isolation Forest + security analysis. "
    "Results update live as you adjust the sliders."
)
st.divider()

# Build and analyze user resource
user_resource = {
    "resource_id":      resource_id,
    "cpu_avg":          cpu_avg,
    "cpu_p95":          cpu_p95,
    "memory_avg":       memory_avg,
    "network_pct":      network_pct,
    "internet_facing":  internet_facing,
    "identity_attached":identity_attached,
}

# Append to test data so Isolation Forest has peer context
results = analyze(TEST_DATA + [user_resource])
r = results[-1]  # user's resource is always last

# ── Status banner ─────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns([2, 1, 1])
with m1:
    if r["is_anomalous"]:
        st.error(f"ANOMALOUS  ·  {r['anomaly_type'].replace('_', ' ').upper()}")
    else:
        st.success("HEALTHY  ·  No anomalies detected")
with m2:
    st.metric("Confidence", f"{r['confidence']:.0%}")
with m3:
    st.metric("Type", r["anomaly_type"].replace("_", " "))

st.markdown("---")

# ── Detail columns ────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.markdown("**Why flagged**")
    st.info(r["reason"])

    st.markdown("**Suggested action**")
    st.write(r["suggested_action"])

    st.markdown("**Confidence**")
    st.progress(r["confidence"])
    st.caption(
        "Blended score: 70% rule confidence + 30% Isolation Forest anomaly score. "
        "ML-only flags are capped at 80% to reflect lower certainty."
    )

with right:
    st.markdown("**Security assessment**")
    sec = r.get("security_note")
    if sec:
        if "HIGH" in sec:
            st.error(sec)
        elif "SUSPICIOUS" in sec or "MEDIUM" in sec:
            st.warning(sec)
        else:
            st.info(sec)
    else:
        st.success("No security concerns detected")

    st.markdown("**Input snapshot**")
    st.json({
        "cpu_avg":          cpu_avg,
        "cpu_p95":          cpu_p95,
        "memory_avg":       memory_avg,
        "network_pct":      network_pct,
        "internet_facing":  internet_facing,
        "identity_attached":identity_attached,
    })

with st.expander("Full JSON output"):
    st.json(r)

# ── Example test cases ────────────────────────────────────────────────────────
st.divider()
st.subheader("All 7 Test Cases")
st.caption("Results from the full evaluation run — the ML model was fitted on these.")

test_results = analyze(TEST_DATA)

rows = []
for res in test_results:
    rows.append({
        "Resource":   res["resource_id"],
        "Status":     "Anomalous" if res["is_anomalous"] else "Healthy",
        "Type":       res["anomaly_type"],
        "Confidence": f"{res['confidence']:.0%}",
        "Security":   "Yes" if res.get("security_note") else "—",
        "Action":     res["suggested_action"][:60] + "...",
    })

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SentnelOps Internship Assignment · Harsh Raj · github.com/harshraj1124/Harsh_SentnelOps")
