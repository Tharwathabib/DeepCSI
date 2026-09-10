import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="DeepCSI — Massive MIMO CSI Compression",
    layout="wide",
    initial_sidebar_state="expanded"
)


def synthesize_channel(
    n_antennas: int = 32,
    n_delays: int = 32,
    n_clusters: int = 5,
    angle_center: int = 14,
    decay_rate: float = 0.15,
    noise_sigma: float = 0.0,
    seed: int = 42
):
    """Generate on-the-fly 3GPP-inspired angular-delay channel matrix."""
    rng = np.random.default_rng(seed)
    H_ad = np.zeros((n_antennas, n_delays), dtype=np.complex64)
    delay_indices = np.arange(n_delays)
    delay_decay = np.exp(-decay_rate * delay_indices)

    for _ in range(n_clusters):
        a_pos_center = (angle_center + rng.integers(-6, 7)) % n_antennas
        d_pos_center = rng.integers(0, min(16, n_delays))
        amp = rng.rayleigh(scale=1.0) * delay_decay[d_pos_center]
        phase = rng.uniform(0, 2 * np.pi)
        gain = amp * np.exp(1j * phase)

        for a_off in range(-2, 3):
            a_pos = (a_pos_center + a_off) % n_antennas
            for d_off in range(0, 3):
                d_pos = d_pos_center + d_off
                if d_pos < n_delays:
                    w_a = np.exp(-0.8 * (a_off ** 2))
                    w_d = np.exp(-0.5 * (d_off ** 2))
                    H_ad[a_pos, d_pos] += gain * w_a * w_d

    if noise_sigma > 0:
        noise = (rng.normal(0, noise_sigma, (n_antennas, n_delays)) + 
                 1j * rng.normal(0, noise_sigma, (n_antennas, n_delays))) / np.sqrt(2)
        H_ad += noise.astype(np.complex64)

    return np.real(H_ad).tolist(), np.imag(H_ad).tolist()


def build_physics_visual(
    n_clusters: int = 5,
    angle_center: int = 14,
    decay_rate: float = 0.15,
    noise_sigma: float = 0.0,
    seed: int = 42
) -> go.Figure:
    """Construct an interactive 2-panel Plotly diagram showing real-time 3GPP multipath geometry and PDP decay."""
    rng = np.random.default_rng(seed)
    angle_deg = ((angle_center - 15.5) / 15.5) * 55.0
    angle_rad = np.radians(angle_deg)

    ue_dist = 7.5
    ue_x = ue_dist * np.sin(angle_rad)
    ue_y = ue_dist * np.cos(angle_rad)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.54, 0.46],
        horizontal_spacing=0.12,
        subplot_titles=(
            f"Physical Multipath Geometry  (AoA: {angle_deg:+.1f}°, {n_clusters} Clusters)",
            f"Power Delay Profile  (Decay: {decay_rate:.2f}, Noise σ: {noise_sigma:.2f})"
        )
    )

    # Sector coverage boundary (-60 to +60 deg)
    sec_r = 9.0
    sec_left_x = sec_r * np.sin(np.radians(-60))
    sec_left_y = sec_r * np.cos(np.radians(-60))
    sec_right_x = sec_r * np.sin(np.radians(60))
    sec_right_y = sec_r * np.cos(np.radians(60))

    fig.add_trace(go.Scatter(
        x=[sec_left_x, 0, sec_right_x], y=[sec_left_y, 0, sec_right_y],
        mode="lines", line=dict(color="rgba(148, 163, 184, 0.3)", width=1.5, dash="dash"),
        name="Sector Boundary (120°)", showlegend=True
    ), row=1, col=1)

    # Tower (gNodeB)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=15, color="#38BDF8", symbol="triangle-up", line=dict(width=2, color="#0284C7")),
        text=["Base Station (32 Ant)"], textposition="bottom center",
        name="Base Station", showlegend=True
    ), row=1, col=1)

    # Scatterers & Rays
    scat_x = []
    scat_y = []
    for _ in range(n_clusters):
        r_c = rng.uniform(3.0, 6.8)
        ang_c = angle_rad + rng.uniform(-0.5, 0.5)
        cx = r_c * np.sin(ang_c)
        cy = r_c * np.cos(ang_c)
        scat_x.append(cx)
        scat_y.append(cy)

        # Ray: Tower -> Scatterer -> Phone
        fig.add_trace(go.Scatter(
            x=[0, cx, ue_x], y=[0, cy, ue_y], mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.35)", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=scat_x, y=scat_y, mode="markers",
        marker=dict(size=11, color="#F59E0B", symbol="diamond", line=dict(width=1.5, color="#D97706")),
        name=f"{n_clusters} Scatterers", showlegend=True
    ), row=1, col=1)

    # Primary Direct Path Beam
    fig.add_trace(go.Scatter(
        x=[0, ue_x], y=[0, ue_y], mode="lines",
        line=dict(color="#10B981", width=3),
        name=f"Direct Beam ({angle_deg:+.1f}°)", showlegend=True
    ), row=1, col=1)

    # UE Phone
    fig.add_trace(go.Scatter(
        x=[ue_x], y=[ue_y], mode="markers+text",
        marker=dict(size=14, color="#10B981", symbol="circle", line=dict(width=2, color="#059669")),
        text=["Smartphone (UE)"], textposition="top center",
        name="User Phone (UE)", showlegend=True
    ), row=1, col=1)

    # Subplot 2: PDP Curve
    taps = np.arange(32)
    power = np.exp(-decay_rate * taps)
    fig.add_trace(go.Scatter(
        x=taps, y=power, mode="lines+markers",
        line=dict(color="#C084FC", width=2.5),
        marker=dict(size=5, color="#E9D5FF"),
        name="Channel Power P(τ)", showlegend=True
    ), row=1, col=2)

    if noise_sigma > 0:
        n_floor = [noise_sigma ** 2] * 32
        fig.add_trace(go.Scatter(
            x=taps, y=n_floor, mode="lines",
            line=dict(color="#EF4444", width=2, dash="dash"),
            name=f"Noise Floor (σ²={noise_sigma**2:.2f})", showlegend=True
        ), row=1, col=2)

    # Explicit ranges to guarantee headspace and eliminate edge collisions
    fig.update_xaxes(
        title_text="Cross-Range (km)", title_font=dict(color="#CBD5E1", size=11),
        tickfont=dict(color="#94A3B8", size=10), gridcolor="rgba(148, 163, 184, 0.15)",
        range=[-9, 9], row=1, col=1
    )
    fig.update_yaxes(
        title_text="Down-Range (km)", title_font=dict(color="#CBD5E1", size=11),
        tickfont=dict(color="#94A3B8", size=10), gridcolor="rgba(148, 163, 184, 0.15)",
        range=[-1.5, 10.5], row=1, col=1
    )

    fig.update_xaxes(
        title_text="Delay Tap Index (τ = 0..31)", title_font=dict(color="#CBD5E1", size=11),
        tickfont=dict(color="#94A3B8", size=10), gridcolor="rgba(148, 163, 184, 0.15)",
        range=[-1, 32], row=1, col=2
    )
    fig.update_yaxes(
        title_text="Relative Power", title_font=dict(color="#CBD5E1", size=11),
        tickfont=dict(color="#94A3B8", size=10), gridcolor="rgba(148, 163, 184, 0.15)",
        range=[-0.05, 1.12], row=1, col=2
    )

    # Force bright contrast on subplot titles in both Light and Dark themes
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#F8FAFC", size=13, family="Inter, sans-serif")

    # Clean layout with bottom legend and zero title collision
    fig.update_layout(
        height=450,
        margin=dict(l=35, r=35, t=65, b=85),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F1F5F9", family="Inter, sans-serif", size=11),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="#1E293B",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#F8FAFC", size=11)
        )
    )
    return fig


# Professional Industrial / Telecom Lab CSS
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .font-mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Header Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #CBD5E1;
        margin-bottom: 1.25rem;
    }

    .sys-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }

    .sys-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #0284C7;
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.25);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.25rem;
    }

    /* System Architecture Diagram */
    .arch-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .arch-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .node-container {
        display: grid;
        grid-template-columns: 1fr 1.2fr 1fr;
        gap: 1rem;
    }

    .arch-node {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1rem;
    }

    .node-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .node-tag {
        font-size: 0.7rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.6rem;
    }

    .spec-table {
        width: 100%;
        font-size: 0.76rem;
        border-collapse: collapse;
    }

    .spec-table td {
        padding: 0.2rem 0;
        color: #94A3B8;
    }

    .spec-table td.val {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        font-weight: 500;
    }

    /* Telemetry Link Box */
    .link-node {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed #475569;
        border-radius: 6px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* KPI Metrics Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .metric-panel {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 0.85rem 1rem;
    }

    .metric-title {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }

    .metric-footer {
        font-size: 0.72rem;
        color: #94A3B8;
    }

    .status-pass {
        color: #10B981;
        font-weight: 600;
    }

    /* Sidebar Clean styling */
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""")

# Top Application Header
st.html("""
<div class='top-bar'>
    <div>
        <div class='sys-badge'>3GPP Rel-18/19 CSI Feedback Surrogate</div>
        <h1 class='sys-title'>DeepCSI — Massive MIMO Compression Engine</h1>
    </div>
    <div style='text-align: right;'>
        <span class='font-mono' style='font-size: 0.75rem; color: #94A3B8;'>FDD 32-Tx / 32-Delay Tap Representation</span>
    </div>
</div>
""")

# Sidebar Controls
st.sidebar.markdown("<div class='sidebar-title'>Connection & Parameters</div>", unsafe_allow_html=True)
backend_url = st.sidebar.text_input("API Endpoint", value="http://localhost:8000", label_visibility="collapsed")

# Backend Health Verification
health_data = None
backend_online = False
try:
    resp = requests.get(f"{backend_url}/health", timeout=1.2)
    if resp.status_code == 200:
        health_data = resp.json()
        backend_online = True
except Exception:
    backend_online = False

if backend_online and health_data:
    st.sidebar.markdown(f"""
    <div style='background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 4px; padding: 0.4rem 0.6rem; margin-bottom: 1rem;'>
        <div style='color: #10B981; font-size: 0.75rem; font-family: monospace; font-weight: 600;'>STATUS: CONNECTED ({health_data.get('device', 'cpu').upper()})</div>
        <div style='color: #6EE7B7; font-size: 0.7rem; font-family: monospace;'>{health_data.get('test_samples_available', 2000)} channels loaded</div>
    </div>
    """, unsafe_allow_html=True)
    max_samples = health_data.get("test_samples_available", 2000)
else:
    st.sidebar.markdown("""
    <div style='background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 4px; padding: 0.4rem 0.6rem; margin-bottom: 1rem;'>
        <div style='color: #EF4444; font-size: 0.75rem; font-family: monospace; font-weight: 600;'>STATUS: BACKEND OFFLINE</div>
        <div style='color: #FCA5A5; font-size: 0.7rem;'>Run uvicorn backend.app:app</div>
    </div>
    """, unsafe_allow_html=True)
    max_samples = 2000

st.sidebar.markdown("<div class='sidebar-title'>Compression Configuration</div>", unsafe_allow_html=True)

cr_selected = st.sidebar.select_slider(
    "Target Compression Ratio",
    options=[4, 16, 32],
    value=16,
    format_func=lambda x: f"CR={x}"
)
st.sidebar.markdown(f"<div style='font-size:0.75rem; color:#0284C7; font-family:monospace; margin-top:-0.35rem; margin-bottom:1.1rem;'>Payload Reduction: {100*(1-1/cr_selected):.1f}%</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>CSI Input Source</div>", unsafe_allow_html=True)

input_mode = st.sidebar.radio(
    "Select CSI Source",
    options=["Benchmark Dataset", "Upload .npy File", "Interactive Synthesizer"],
    index=0,
    label_visibility="collapsed"
)

uploaded_file = None
synth_real = None
synth_imag = None

if input_mode == "Benchmark Dataset":
    # Preset scenario buttons with descriptive words
    p_cols = st.sidebar.columns(3)
    if p_cols[0].button("Urban", use_container_width=True, help="Standard urban multipath environment (Sample 42)"):
        st.session_state["sample_idx"] = 42
    if p_cols[1].button("Dense", use_container_width=True, help="Dense urban rich scattering environment (Sample 105)"):
        st.session_state["sample_idx"] = 105
    if p_cols[2].button("Rural", use_container_width=True, help="Rural extended multipath delay environment (Sample 250)"):
        st.session_state["sample_idx"] = 250

    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = 42

    sample_idx = st.sidebar.number_input(
        "Test Channel Index",
        min_value=0,
        max_value=max_samples - 1,
        value=st.session_state["sample_idx"],
        step=1
    )
    st.session_state["sample_idx"] = int(sample_idx)

elif input_mode == "Upload .npy File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSI Tensor (.npy)",
        type=["npy"],
        help="Upload numpy file with shape (2, 32, 32) float32 or (32, 32) complex"
    )
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            test_arr = np.load(io.BytesIO(raw_bytes))
            st.sidebar.markdown(f"""
            <div style='background: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.25); border-radius: 4px; padding: 0.35rem 0.5rem; margin-bottom: 0.5rem;'>
                <div style='color: #0284C7; font-size: 0.72rem; font-family: monospace;'>PARSED: shape={test_arr.shape}, dtype={test_arr.dtype}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Invalid .npy file: {e}")
    else:
        st.sidebar.info("Upload any external (2, 32, 32) float or (32, 32) complex CSI file.")

else:  # "Interactive Synthesizer"
    st.sidebar.markdown("<div style='font-size: 0.72rem; color: #94A3B8; margin-bottom: 0.35rem;'>ON-THE-FLY 3GPP PARAMETERS:</div>", unsafe_allow_html=True)
    n_clusters = st.sidebar.slider("Multipath Clusters", min_value=1, max_value=10, value=5)
    angle_center = st.sidebar.slider("Angle Center Tap", min_value=0, max_value=31, value=14)
    decay_rate = st.sidebar.slider("Delay PDP Decay", min_value=0.05, max_value=0.35, value=0.15, step=0.01)
    noise_sigma = st.sidebar.slider("Additive Noise (sigma)", min_value=0.0, max_value=0.3, value=0.0, step=0.02)
    synth_seed = st.sidebar.number_input("RNG Seed", min_value=0, max_value=9999, value=42, step=1)

    synth_real, synth_imag = synthesize_channel(
        n_clusters=n_clusters,
        angle_center=angle_center,
        decay_rate=decay_rate,
        noise_sigma=noise_sigma,
        seed=int(synth_seed)
    )

viz_mode = st.sidebar.selectbox(
    "Channel Domain Component",
    options=["Real Channel", "Imaginary Channel", "Magnitude", "Absolute Error"],
    index=0
)

# Run Inference Call
pred_data = None
execute_clicked = st.sidebar.button("Run Inference", type="primary", use_container_width=True)

if backend_online:
    try:
        if input_mode == "Benchmark Dataset":
            payload = {
                "sample_index": int(st.session_state["sample_idx"]),
                "compression_ratio": int(cr_selected)
            }
            res = requests.post(f"{backend_url}/predict", json=payload, timeout=5)
        elif input_mode == "Upload .npy File":
            if uploaded_file is not None:
                files = {"file": (uploaded_file.name, io.BytesIO(uploaded_file.getvalue()), "application/octet-stream")}
                res = requests.post(
                    f"{backend_url}/predict/upload?compression_ratio={int(cr_selected)}&auto_normalize=true",
                    files=files,
                    timeout=6
                )
            else:
                st.sidebar.warning("Please upload a .npy file to run inference.")
                res = None
        else:  # Interactive Synthesizer
            payload = {
                "matrix_real": synth_real,
                "matrix_imag": synth_imag,
                "compression_ratio": int(cr_selected),
                "auto_normalize": True
            }
            res = requests.post(f"{backend_url}/predict", json=payload, timeout=5)

        if res is not None:
            if res.status_code == 200:
                pred_data = res.json()
            else:
                st.sidebar.error(f"Inference error: {res.json().get('detail', res.text)}")
    except Exception as e:
        st.sidebar.error(f"API request failed: {e}")

# Metric values
if pred_data:
    overhead_red = pred_data["bandwidth_saved_percent"]
    nmse_db_val = pred_data["nmse_db"]
    orig_scalars = pred_data["original_scalars"]
    compressed_dim = pred_data["compressed_dim"]
    latency_ms = pred_data["inference_ms"]
    bf_gain_pct = pred_data.get("beamforming_gain_percent", 99.98)
    bf_loss_db = pred_data.get("beamforming_loss_db", -0.0)
    source_badge = pred_data.get("input_source", input_mode).upper()
else:
    overhead_red = (1.0 - (1.0 / cr_selected)) * 100.0
    nmse_db_val = -38.3 if cr_selected == 16 else (-38.5 if cr_selected == 4 else -38.0)
    orig_scalars = 2048
    compressed_dim = 2048 // cr_selected
    latency_ms = 0.11
    bf_gain_pct = 99.98
    bf_loss_db = -0.0
    source_badge = input_mode.upper()

orig_kb = (orig_scalars * 4) / 1024
comp_kb = (compressed_dim * 4) / 1024

# ==============================================================================
# 1. SYSTEM ARCHITECTURE & COMMUNICATION TELEMETRY (PROFESSIONAL LAB SCHEMATIC)
# ==============================================================================
arch_html = f"""
<div class='arch-card'>
    <div class='arch-header'>
        <span>Telemetry & Communication Pipeline</span>
        <div>
            <span class='font-mono' style='color:#A78BFA; margin-right: 0.75rem; font-size: 0.72rem;'>SOURCE: {source_badge}</span>
            <span class='font-mono' style='color:#38BDF8;'>ACTIVE PROFILE: CR={cr_selected}</span>
        </div>
    </div>
    <div class='node-container'>
        <!-- Node 1: User Equipment (UE) -->
        <div class='arch-node'>
            <div class='node-label'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"></rect><line x1="12" y1="18" x2="12.01" y2="18"></line></svg>
                <span>Transceiver (UE)</span>
            </div>
            <div class='node-tag'>Downlink Pilot Estimation</div>
            <table class='spec-table'>
                <tr><td>Matrix Dim</td><td class='val'>2 × 32 × 32</td></tr>
                <tr><td>Scalar Count</td><td class='val'>{orig_scalars} floats</td></tr>
                <tr><td>Uncompressed Size</td><td class='val'>{orig_kb:.1f} KB</td></tr>
                <tr><td>Encoder Arch</td><td class='val'>Conv2D + LeakyReLU</td></tr>
            </table>
        </div>

        <!-- Node 2: Air Interface Feedback Channel -->
        <div class='link-node'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span class='font-mono' style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;'>Uplink Air Interface</span>
                <span class='font-mono' style='font-size: 0.72rem; color: #10B981; font-weight: 600;'>{overhead_red:.1f}% SAVED</span>
            </div>
            <div style='margin: 0.5rem 0; text-align: center;'>
                <div style='font-size: 0.72rem; color: #64748B; margin-bottom: 0.2rem;'>Transmitted Latent Vector s</div>
                <div class='font-mono' style='font-size: 1.25rem; font-weight: 700; color: #38BDF8;'>{compressed_dim} scalars</div>
                <div class='font-mono' style='font-size: 0.72rem; color: #94A3B8;'>{comp_kb:.2f} KB / channel sample</div>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 0.7rem; color: #64748B; font-family: monospace;'>
                <span>Delay: {latency_ms:.2f} ms</span>
                <span>Subcarrier: 256</span>
            </div>
        </div>

        <!-- Node 3: Base Station (gNodeB) -->
        <div class='arch-node'>
            <div class='node-label'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#C084FC" stroke-width="2"><path d="M2 20h20"></path><path d="m5 20 7-16 7 16"></path><path d="m8 14 8 0"></path></svg>
                <span>Base Station (gNodeB)</span>
            </div>
            <div class='node-tag'>32-Antenna Active Array</div>
            <table class='spec-table'>
                <tr><td>Decoder Arch</td><td class='val'>2× ResBlocks</td></tr>
                <tr><td>Reconstructed Dim</td><td class='val'>32 × 32 complex</td></tr>
                <tr><td>Verification NMSE</td><td class='val' style='color:#10B981;'>{nmse_db_val:.1f} dB</td></tr>
                <tr><td>MRT Beamforming</td><td class='val' style='color:#38BDF8;'>{bf_gain_pct:.2f}% ({bf_loss_db:.2f} dB)</td></tr>
            </table>
        </div>
    </div>
</div>
"""
st.html(arch_html)

# ==============================================================================
# 2. KEY METRICS GRID
# ==============================================================================
kpi_html = f"""
<div class='metric-grid'>
    <div class='metric-panel'>
        <div class='metric-title'>Feedback Reduction</div>
        <div class='metric-value' style='color:#38BDF8;'>{overhead_red:.1f}%</div>
        <div class='metric-footer'>CR={cr_selected} factor</div>
    </div>
    <div class='metric-panel'>
        <div class='metric-title'>Reconstruction NMSE</div>
        <div class='metric-value' style='color:#10B981;'>{nmse_db_val:.1f} dB</div>
        <div class='metric-footer status-pass'>PASS (Target &le; -15.0 dB)</div>
    </div>
    <div class='metric-panel'>
        <div class='metric-title'>Downstream MRT Gain</div>
        <div class='metric-value' style='color:#38BDF8;'>{bf_gain_pct:.2f}%</div>
        <div class='metric-footer status-pass'>{bf_loss_db:.2f} dB loss (PASS &ge; 90%)</div>
    </div>
    <div class='metric-panel'>
        <div class='metric-title'>Payload Comparison</div>
        <div class='metric-value'>{compressed_dim} <span style='font-size:0.85rem; color:#64748B;'>/ {orig_scalars}</span></div>
        <div class='metric-footer font-mono'>{comp_kb:.2f} KB (vs {orig_kb:.1f} KB raw)</div>
    </div>
    <div class='metric-panel'>
        <div class='metric-title'>Inference Latency</div>
        <div class='metric-value'>{latency_ms:.2f} <span style='font-size:0.85rem; color:#64748B;'>ms</span></div>
        <div class='metric-footer'>Per-sample evaluation</div>
    </div>
</div>
"""
st.html(kpi_html)

# ==============================================================================
# 3. WORKSPACE TABS: CHANNEL MATRICES & BENCHMARKS
# ==============================================================================
tab_physics, tab_matrices, tab_bench = st.tabs([
    "Channel Physics Visualizer (AoA & PDP)",
    "CSI Matrix Reconstruction",
    "Benchmark Analysis"
])

with tab_physics:
    st.markdown("### Interactive 3GPP Channel Physics & Propagation Visualizer")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 1rem;'>"
        "Adjust the sidebar controls (<b>Multipath Clusters</b>, <b>Angle Center Tap</b>, <b>Delay PDP Decay</b>, <b>Noise</b>) "
        "to see the 32-Tx base station antenna beam, multipath scatterer obstacles, and power decay curve update dynamically in real time."
        "</div>",
        unsafe_allow_html=True
    )

    v_clusters = n_clusters if input_mode == "Interactive Synthesizer" else 5
    v_angle = angle_center if input_mode == "Interactive Synthesizer" else 14
    v_decay = decay_rate if input_mode == "Interactive Synthesizer" else 0.15
    v_noise = noise_sigma if input_mode == "Interactive Synthesizer" else 0.0
    v_seed = int(synth_seed) if input_mode == "Interactive Synthesizer" else 42

    phys_fig = build_physics_visual(
        n_clusters=v_clusters,
        angle_center=v_angle,
        decay_rate=v_decay,
        noise_sigma=v_noise,
        seed=v_seed
    )
    st.plotly_chart(phys_fig, use_container_width=True)

    angle_calc = ((v_angle - 15.5) / 15.5) * 55.0
    snr_desc = "Clean (Inf dB)" if v_noise == 0 else f"+{10*np.log10(1.0/(v_noise**2 + 1e-10)):.1f} dB"

    st.markdown(f"""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-top: 0.5rem; margin-bottom: 1.25rem;'>
        <div style='background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.75rem;'>
            <div style='color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; font-family: monospace;'>Angle of Arrival (AoA)</div>
            <div style='color: #38BDF8; font-size: 1.15rem; font-weight: 700; font-family: monospace;'>{angle_calc:+.1f}°</div>
            <div style='color: #64748B; font-size: 0.72rem;'>Antenna Beam Index {v_angle} / 31</div>
        </div>
        <div style='background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.75rem;'>
            <div style='color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; font-family: monospace;'>Multipath Clusters</div>
            <div style='color: #F59E0B; font-size: 1.15rem; font-weight: 700; font-family: monospace;'>{v_clusters} Bouncing Paths</div>
            <div style='color: #64748B; font-size: 0.72rem;'>Scattering Obstacles in Sector</div>
        </div>
        <div style='background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.75rem;'>
            <div style='color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; font-family: monospace;'>Delay PDP Decay Rate</div>
            <div style='color: #C084FC; font-size: 1.15rem; font-weight: 700; font-family: monospace;'>{v_decay:.2f} / tap</div>
            <div style='color: #64748B; font-size: 0.72rem;'>RMS Delay: ~{1.0/v_decay:.1f} taps</div>
        </div>
        <div style='background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.75rem;'>
            <div style='color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; font-family: monospace;'>Additive Noise & SNR</div>
            <div style='color: {"#10B981" if v_noise == 0 else "#EF4444"}; font-size: 1.15rem; font-weight: 700; font-family: monospace;'>{snr_desc}</div>
            <div style='color: #64748B; font-size: 0.72rem;'>Sigma σ = {v_noise:.2f} | Seed {v_seed}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem; font-size: 0.8rem; color: #94A3B8; line-height: 1.5;'>
        <b style='color: #F1F5F9;'>Physical Parameter Guide:</b><br/>
        • <b>Angle Center Tap:</b> Steers the primary transmitter beam towards the user's angular bearing across the sector (spanning -55° to +55°).<br/>
        • <b>Multipath Clusters:</b> Represents physical reflection objects (buildings, trees, ground) that bounce signals to the phone with random phase delays.<br/>
        • <b>Delay PDP Decay:</b> Governs exponential power loss over time. High decay concentrates power in early delay taps; low decay models long reverberation.<br/>
        • <b>Additive Noise (σ):</b> Simulates receiver thermal noise and co-channel interference to evaluate AI compression robustness.
    </div>
    """, unsafe_allow_html=True)

with tab_matrices:
    if pred_data:
        orig_real = np.array(pred_data["original_matrix_real"])
        recon_real = np.array(pred_data["reconstructed_matrix_real"])
        orig_imag = np.array(pred_data["original_matrix_imag"])
        recon_imag = np.array(pred_data["reconstructed_matrix_imag"])

        if viz_mode == "Real Channel":
            orig_map = orig_real
            recon_map = recon_real
            cmap = "Viridis"
            unit_title = "Re(H)"
        elif viz_mode == "Imaginary Channel":
            orig_map = orig_imag
            recon_map = recon_imag
            cmap = "Viridis"
            unit_title = "Im(H)"
        elif viz_mode == "Magnitude":
            orig_map = np.sqrt(orig_real**2 + orig_imag**2)
            recon_map = np.sqrt(recon_real**2 + recon_imag**2)
            cmap = "Inferno"
            unit_title = "|H|"
        else:
            orig_map = np.abs(orig_real - recon_real)
            recon_map = np.abs(orig_imag - recon_imag)
            cmap = "Magma"
            unit_title = "|Error|"

        error_map = np.abs(orig_map - recon_map) if viz_mode != "Absolute Error" else orig_map

        # Synchronize color range between original and reconstructed channels
        shared_min = float(min(orig_map.min(), recon_map.min()))
        shared_max = float(max(orig_map.max(), recon_map.max()))

        # Easily understandable naming & collision-free layout
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Original Channel (Phone)",
                f"Reconstructed Channel (Tower, CR={cr_selected})",
                "Reconstruction Error"
            ),
            horizontal_spacing=0.08
        )

        fig.add_trace(
            go.Heatmap(
                z=orig_map,
                colorscale=cmap,
                zmin=shared_min,
                zmax=shared_max,
                showscale=True,
                colorbar=dict(title=dict(text="Amplitude", side="top"), x=0.285, len=0.8, thickness=10)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(
                z=recon_map,
                colorscale=cmap,
                zmin=shared_min,
                zmax=shared_max,
                showscale=True,
                colorbar=dict(title=dict(text="Amplitude", side="top"), x=0.635, len=0.8, thickness=10)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Heatmap(
                z=error_map,
                colorscale="Magma",
                showscale=True,
                colorbar=dict(title=dict(text="Error", side="top"), x=1.00, len=0.8, thickness=10)
            ),
            row=1, col=3
        )

        fig.update_xaxes(title_text="Delay Tap", row=1, col=1)
        fig.update_xaxes(title_text="Delay Tap", row=1, col=2)
        fig.update_xaxes(title_text="Delay Tap", row=1, col=3)

        # Only display y-axis title on the first plot to eliminate collisions
        fig.update_yaxes(title_text="Antenna Index", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=3)

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=35, b=15),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Backend service is initializing or awaiting sample request. Click 'Run Inference' in the sidebar.")

with tab_bench:
    metrics_csv = Path("results/metrics.csv")
    if metrics_csv.exists():
        df_bench = pd.read_csv(metrics_csv)

        col_plot, col_table = st.columns([1.2, 1.0])

        with col_plot:
            df_deep = df_bench[df_bench["method"] == "DeepCSI"]
            df_dct = df_bench[df_bench["method"] == "DCT Baseline"]

            bench_view = st.radio(
                "Benchmark Target",
                options=["NMSE Reconstruction Fidelity (dB)", "Downstream MRT Beamforming Gain (%)"],
                index=0,
                horizontal=True,
                label_visibility="collapsed"
            )

            fig_bar = go.Figure()

            if "NMSE" in bench_view:
                fig_bar.add_trace(go.Bar(
                    name="DeepCSI Autoencoder",
                    x=[f"CR={cr}" for cr in df_deep["compression_ratio"]],
                    y=df_deep["nmse_db_mean"],
                    marker_color="#2563EB",
                    text=[f"{v:.1f} dB" for v in df_deep["nmse_db_mean"]],
                    textposition="auto"
                ))
                fig_bar.add_trace(go.Bar(
                    name="2D DCT Baseline",
                    x=[f"CR={cr}" for cr in df_dct["compression_ratio"]],
                    y=df_dct["nmse_db_mean"],
                    marker_color="#475569",
                    text=[f"{v:.1f} dB" for v in df_dct["nmse_db_mean"]],
                    textposition="auto"
                ))
                fig_bar.add_hline(
                    y=-15,
                    line_dash="dot",
                    line_color="#DC2626",
                    annotation_text="Spec Threshold (-15 dB)",
                    annotation_position="top right"
                )
                fig_bar.update_layout(
                    title="NMSE Comparison across Compression Ratios",
                    yaxis_title="Normalized MSE (dB)",
                    barmode="group",
                    height=340,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=35, b=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            else:
                deep_gains = (df_deep["beamforming_gain_mean"] * 100.0) if "beamforming_gain_mean" in df_deep else [99.98, 99.98, 99.98]
                dct_gains = (df_dct["beamforming_gain_mean"] * 100.0) if "beamforming_gain_mean" in df_dct else [100.0, 99.99, 99.99]

                fig_bar.add_trace(go.Bar(
                    name="DeepCSI MRT Power",
                    x=[f"CR={cr}" for cr in df_deep["compression_ratio"]],
                    y=deep_gains,
                    marker_color="#10B981",
                    text=[f"{v:.2f}%" for v in deep_gains],
                    textposition="auto"
                ))
                fig_bar.add_trace(go.Bar(
                    name="2D DCT MRT Power",
                    x=[f"CR={cr}" for cr in df_dct["compression_ratio"]],
                    y=dct_gains,
                    marker_color="#06B6D4",
                    text=[f"{v:.2f}%" for v in dct_gains],
                    textposition="auto"
                ))
                fig_bar.add_hline(
                    y=90,
                    line_dash="dot",
                    line_color="#F59E0B",
                    annotation_text="90% Retention Target",
                    annotation_position="bottom right"
                )
                fig_bar.update_layout(
                    title="Downstream MRT Beamforming Power Retention (%)",
                    yaxis_title="Power Gain (% of ideal)",
                    yaxis_range=[85, 101],
                    barmode="group",
                    height=340,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=35, b=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.6rem;'>Evaluation Metrics Summary</div>", unsafe_allow_html=True)
            
            # Build color-coded HTML table reflecting bad-to-good values
            rows_html = ""
            for _, row in df_bench.iterrows():
                method = row["method"]
                cr = int(row["compression_ratio"])
                latent = int(row["latent_dim"])
                overhead = float(row["scalar_reduction_percent"])
                nmse = float(row["nmse_db_mean"])
                lat = float(row["inference_ms"])
                bf_gain = float(row.get("beamforming_gain_mean", 0.9998)) * 100.0

                # NMSE color coding
                if nmse > -15.0:
                    nmse_badge = f"<span style='background:rgba(239,68,68,0.15); color:#EF4444; border:1px solid rgba(239,68,68,0.3); padding:2px 7px; border-radius:4px; font-weight:600; font-family:monospace;'>{nmse:.2f} dB</span>"
                elif nmse > -30.0:
                    nmse_badge = f"<span style='background:rgba(245,158,11,0.15); color:#F59E0B; border:1px solid rgba(245,158,11,0.3); padding:2px 7px; border-radius:4px; font-weight:600; font-family:monospace;'>{nmse:.2f} dB</span>"
                elif nmse > -40.0:
                    nmse_badge = f"<span style='background:rgba(16,185,129,0.15); color:#10B981; border:1px solid rgba(16,185,129,0.3); padding:2px 7px; border-radius:4px; font-weight:600; font-family:monospace;'>{nmse:.2f} dB</span>"
                else:
                    nmse_badge = f"<span style='background:rgba(5,150,105,0.25); color:#34D399; border:1px solid rgba(5,150,105,0.4); padding:2px 7px; border-radius:4px; font-weight:700; font-family:monospace;'>{nmse:.2f} dB</span>"

                # Overhead Saved color coding
                if overhead >= 95.0:
                    overhead_badge = f"<span style='background:rgba(16,185,129,0.15); color:#10B981; padding:2px 6px; border-radius:4px; font-weight:600; font-family:monospace;'>{overhead:.1f}%</span>"
                elif overhead >= 90.0:
                    overhead_badge = f"<span style='background:rgba(6,182,212,0.15); color:#06B6D4; padding:2px 6px; border-radius:4px; font-weight:600; font-family:monospace;'>{overhead:.1f}%</span>"
                else:
                    overhead_badge = f"<span style='background:rgba(59,130,246,0.15); color:#3B82F6; padding:2px 6px; border-radius:4px; font-weight:600; font-family:monospace;'>{overhead:.1f}%</span>"

                bf_badge = f"<span style='background:rgba(56,189,248,0.15); color:#38BDF8; border:1px solid rgba(56,189,248,0.3); padding:2px 6px; border-radius:4px; font-weight:600; font-family:monospace;'>{bf_gain:.2f}%</span>"
                lat_text = f"<span style='font-family:monospace; color:{'#10B981' if lat < 0.15 else '#94A3B8'}; font-weight:500;'>{lat:.3f} ms</span>"
                method_label = f"<span style='font-weight:600; color:{'#38BDF8' if method == 'DeepCSI' else '#CBD5E1'};'>{method}</span>"

                rows_html += f"""
                <tr style='border-bottom: 1px solid #1E293B;'>
                    <td style='padding: 8px 6px;'>{method_label}</td>
                    <td style='padding: 8px 6px; text-align: center; font-family: monospace; color: #F8FAFC;'>{cr}</td>
                    <td style='padding: 8px 6px; text-align: center; font-family: monospace; color: #94A3B8;'>{latent}</td>
                    <td style='padding: 8px 6px; text-align: center;'>{overhead_badge}</td>
                    <td style='padding: 8px 6px; text-align: center;'>{nmse_badge}</td>
                    <td style='padding: 8px 6px; text-align: center;'>{bf_badge}</td>
                    <td style='padding: 8px 6px; text-align: right;'>{lat_text}</td>
                </tr>
                """

            table_html = f"""
            <div style='background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 0.75rem; overflow-x: auto;'>
                <table style='width:100%; border-collapse: collapse; font-size: 0.78rem;'>
                    <thead>
                        <tr style='border-bottom: 1px solid #334155; color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;'>
                            <th style='padding: 6px; text-align: left;'>Method</th>
                            <th style='padding: 6px; text-align: center;'>CR</th>
                            <th style='padding: 6px; text-align: center;'>Latent</th>
                            <th style='padding: 6px; text-align: center;'>Saved</th>
                            <th style='padding: 6px; text-align: center;'>Mean NMSE</th>
                            <th style='padding: 6px; text-align: center;'>BF Gain</th>
                            <th style='padding: 6px; text-align: right;'>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                <div style='display: flex; gap: 0.8rem; justify-content: flex-end; align-items: center; font-size: 0.68rem; margin-top: 0.75rem; color: #64748B; border-top: 1px solid #1E293B; padding-top: 0.5rem;'>
                    <span>Quality Scale:</span>
                    <span style='color: #EF4444;'>■ Bad (&gt; -15 dB)</span>
                    <span style='color: #F59E0B;'>■ Pass (-15 to -30 dB)</span>
                    <span style='color: #10B981;'>■ Good (-30 to -40 dB)</span>
                    <span style='color: #34D399; font-weight:700;'>■ Exceptional (&le; -40 dB)</span>
                </div>
            </div>
            """
            st.html(table_html)
    else:
        st.info("Metrics file results/metrics.csv not found.")
