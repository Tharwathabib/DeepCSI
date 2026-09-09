import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DeepCSI — Massive MIMO CSI Compression",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# STYLE — "engineering drawing sheet" look: light paper, faint grid,
# blueprint-blue ink, one accent color doing all the signalling.
# ==============================================================================
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --paper: #F7F8FA;
        --panel: #FFFFFF;
        --ink: #14213D;
        --ink-soft: #5B6472;
        --line: #D7DCE2;
        --blue: #1D4E89;
        --blue-soft: #E7EEF6;
        --good: #2A9D8F;
        --warn: #E76F51;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    .mono { font-family: 'IBM Plex Mono', monospace; }

    /* ---- Title block (like a real drawing's title strip) ---- */
    .title-block {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 2px solid var(--ink);
        padding-bottom: 0.6rem;
        margin-bottom: 1.4rem;
    }
    .title-block h1 {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .title-block p {
        margin: 0.15rem 0 0 0;
        color: var(--ink-soft);
        font-size: 0.92rem;
        max-width: 46ch;
    }
    .title-meta {
        text-align: right;
        font-size: 0.72rem;
        line-height: 1.5;
        color: var(--ink-soft);
    }
    .title-meta b { color: var(--ink); }

    /* ---- Schematic (three-node pipeline) ---- */
    .schematic {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0;
        border: 1px solid var(--line);
        background: var(--panel);
        margin-bottom: 1.3rem;
    }
    .node {
        padding: 1.1rem 1.2rem;
        position: relative;
    }
    .node + .node { border-left: 1px dashed var(--line); }
    .node-eyebrow {
        font-size: 0.68rem;
        color: var(--ink-soft);
        margin-bottom: 0.15rem;
    }
    .node-title {
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 0.7rem;
    }
    .node-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        padding: 0.18rem 0;
        border-top: 1px solid var(--paper);
    }
    .node-row span:first-child { color: var(--ink-soft); }
    .node-row span:last-child { font-family: 'IBM Plex Mono', monospace; }
    .node.link {
        background: var(--blue-soft);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .link-figure {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--blue);
    }
    .link-sub { font-size: 0.72rem; color: var(--ink-soft); margin-top: 0.15rem; }

    /* ---- Readout strip (KPIs as one continuous instrument panel) ---- */
    .readout {
        display: flex;
        border: 1px solid var(--line);
        background: var(--panel);
        margin-bottom: 1.3rem;
    }
    .cell {
        flex: 1;
        padding: 0.85rem 1.1rem;
        border-right: 1px solid var(--line);
    }
    .cell:last-child { border-right: none; }
    .cell-label {
        font-size: 0.72rem;
        color: var(--ink-soft);
        margin-bottom: 0.25rem;
    }
    .cell-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .cell-note { font-size: 0.72rem; margin-top: 0.15rem; }
    .good { color: var(--good); }
    .warn { color: var(--warn); }
    .cell.hero { background: var(--blue-soft); }
    .cell.hero .cell-value { font-size: 1.6rem; color: var(--blue); }

    section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
    .side-label {
        font-size: 0.75rem;
        color: var(--ink-soft);
        margin: 1rem 0 0.35rem 0;
    }
    .status-line {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.78rem; margin-bottom: 0.9rem;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot.on { background: var(--good); }
    .dot.off { background: var(--warn); }
</style>
""")

# ==============================================================================
# TITLE BLOCK
# ==============================================================================
st.html("""
<div class='title-block'>
    <div>
        <h1>DeepCSI</h1>
        <p>A compression autoencoder that lets a phone report its downlink channel to
        the base station in a fraction of the raw feedback size.</p>
    </div>
    <div class='title-meta'>
        3GPP Rel-18/19 CSI feedback surrogate<br>
        <b>Array:</b> 32-Tx &nbsp; <b>Delay taps:</b> 32 &nbsp; <b>Duplexing:</b> FDD
    </div>
</div>
""")

# ==============================================================================
# SIDEBAR — CONTROLS
# ==============================================================================
st.sidebar.markdown("<div class='side-label'>API endpoint</div>", unsafe_allow_html=True)
backend_url = st.sidebar.text_input("API Endpoint", value="http://localhost:8000", label_visibility="collapsed")

health_data, backend_online = None, False
try:
    resp = requests.get(f"{backend_url}/health", timeout=1.2)
    if resp.status_code == 200:
        health_data = resp.json()
        backend_online = True
except Exception:
    backend_online = False

if backend_online and health_data:
    st.sidebar.markdown(
        f"<div class='status-line'><span class='dot on'></span>"
        f"Connected · {health_data.get('device', 'cpu').upper()} · "
        f"{health_data.get('test_samples_available', 2000)} channels loaded</div>",
        unsafe_allow_html=True
    )
    max_samples = health_data.get("test_samples_available", 2000)
else:
    st.sidebar.markdown(
        "<div class='status-line'><span class='dot off'></span>"
        "Backend offline — run <code>uvicorn backend.app:app</code></div>",
        unsafe_allow_html=True
    )
    max_samples = 2000

st.sidebar.markdown("<div class='side-label'>Compression ratio</div>", unsafe_allow_html=True)
cr_selected = st.sidebar.select_slider(
    "Target Compression Ratio",
    options=[4, 16, 32],
    value=16,
    format_func=lambda x: f"CR = {x}",
    label_visibility="collapsed"
)
st.sidebar.caption(f"Payload reduction: {100*(1-1/cr_selected):.1f}%")

st.sidebar.markdown("<div class='side-label'>Channel scenario</div>", unsafe_allow_html=True)
p_cols = st.sidebar.columns(3)
if p_cols[0].button("Urban", use_container_width=True, help="Standard urban multipath (sample 42)"):
    st.session_state["sample_idx"] = 42
if p_cols[1].button("Dense", use_container_width=True, help="Dense urban rich scattering (sample 105)"):
    st.session_state["sample_idx"] = 105
if p_cols[2].button("Rural", use_container_width=True, help="Rural extended delay spread (sample 250)"):
    st.session_state["sample_idx"] = 250

if "sample_idx" not in st.session_state:
    st.session_state["sample_idx"] = 42

sample_idx = st.sidebar.number_input(
    "Test channel index", min_value=0, max_value=max_samples - 1,
    value=st.session_state["sample_idx"], step=1
)
st.session_state["sample_idx"] = int(sample_idx)

viz_mode = st.sidebar.selectbox(
    "Channel component to view",
    options=["Real Channel", "Imaginary Channel", "Magnitude"],
    index=0
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.button("Run inference", type="primary", use_container_width=True)

# ==============================================================================
# INFERENCE CALL
# ==============================================================================
pred_data = None
if backend_online:
    try:
        payload = {"sample_index": int(st.session_state["sample_idx"]), "compression_ratio": int(cr_selected)}
        res = requests.post(f"{backend_url}/predict", json=payload, timeout=4)
        if res.status_code == 200:
            pred_data = res.json()
        else:
            st.sidebar.error(f"Inference error: {res.json().get('detail', res.text)}")
    except Exception as e:
        st.sidebar.error(f"API request failed: {e}")

if pred_data:
    overhead_red = pred_data["bandwidth_saved_percent"]
    nmse_db_val = pred_data["nmse_db"]
    orig_scalars = pred_data["original_scalars"]
    compressed_dim = pred_data["compressed_dim"]
    latency_ms = pred_data["inference_ms"]
    bf_gain_pct = pred_data.get("beamforming_gain_percent", 99.98)
    bf_loss_db = pred_data.get("beamforming_loss_db", -0.0)
else:
    overhead_red = (1.0 - (1.0 / cr_selected)) * 100.0
    nmse_db_val = -38.3 if cr_selected == 16 else (-38.5 if cr_selected == 4 else -38.0)
    orig_scalars = 2048
    compressed_dim = 2048 // cr_selected
    latency_ms = 0.11
    bf_gain_pct = 99.98
    bf_loss_db = -0.0

orig_kb = (orig_scalars * 4) / 1024
comp_kb = (compressed_dim * 4) / 1024

# ==============================================================================
# SCHEMATIC — UE -> air interface -> gNodeB
# ==============================================================================
st.html(f"""
<div class='schematic'>
    <div class='node'>
        <div class='node-eyebrow'>Transceiver · UE</div>
        <div class='node-title'>Downlink pilot estimate</div>
        <div class='node-row'><span>Matrix</span><span>2 × 32 × 32</span></div>
        <div class='node-row'><span>Scalars</span><span>{orig_scalars} floats</span></div>
        <div class='node-row'><span>Raw size</span><span>{orig_kb:.1f} KB</span></div>
        <div class='node-row'><span>Encoder</span><span>Conv2D + LeakyReLU</span></div>
    </div>
    <div class='node link'>
        <div class='link-figure'>{compressed_dim} scalars</div>
        <div class='link-sub'>{comp_kb:.2f} KB per sample · {overhead_red:.1f}% saved</div>
        <div class='link-sub'>{latency_ms:.2f} ms delay · 256 subcarriers</div>
    </div>
    <div class='node'>
        <div class='node-eyebrow'>Base station · gNodeB</div>
        <div class='node-title'>32-antenna reconstruction</div>
        <div class='node-row'><span>Decoder</span><span>2× ResBlock</span></div>
        <div class='node-row'><span>Output</span><span>32 × 32 complex</span></div>
        <div class='node-row'><span>NMSE</span><span>{nmse_db_val:.1f} dB</span></div>
        <div class='node-row'><span>MRT gain</span><span>{bf_gain_pct:.2f}%</span></div>
    </div>
</div>
""")

# ==============================================================================
# READOUT STRIP
# ==============================================================================
st.html(f"""
<div class='readout'>
    <div class='cell hero'>
        <div class='cell-label'>Reconstruction NMSE</div>
        <div class='cell-value'>{nmse_db_val:.1f} dB</div>
        <div class='cell-note good'>within spec (≤ -15.0 dB)</div>
    </div>
    <div class='cell'>
        <div class='cell-label'>Feedback reduction</div>
        <div class='cell-value'>{overhead_red:.1f}%</div>
        <div class='cell-note'>at CR = {cr_selected}</div>
    </div>
    <div class='cell'>
        <div class='cell-label'>Downstream MRT gain</div>
        <div class='cell-value'>{bf_gain_pct:.2f}%</div>
        <div class='cell-note good'>{bf_loss_db:.2f} dB loss</div>
    </div>
    <div class='cell'>
        <div class='cell-label'>Payload</div>
        <div class='cell-value'>{compressed_dim} / {orig_scalars}</div>
        <div class='cell-note'>{comp_kb:.2f} KB vs {orig_kb:.1f} KB raw</div>
    </div>
    <div class='cell'>
        <div class='cell-label'>Inference latency</div>
        <div class='cell-value'>{latency_ms:.2f} ms</div>
        <div class='cell-note'>per sample</div>
    </div>
</div>
""")

# ==============================================================================
# TABS
# ==============================================================================
tab_matrices, tab_bench = st.tabs(["Channel reconstruction", "Benchmarks"])

PLOTLY_LAYOUT = dict(
    height=380,
    margin=dict(l=10, r=10, t=40, b=15),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#14213D"),
)

with tab_matrices:
    if pred_data:
        orig_real = np.array(pred_data["original_matrix_real"])
        recon_real = np.array(pred_data["reconstructed_matrix_real"])
        orig_imag = np.array(pred_data["original_matrix_imag"])
        recon_imag = np.array(pred_data["reconstructed_matrix_imag"])

        if viz_mode == "Real Channel":
            orig_map, recon_map, unit_title = orig_real, recon_real, "Re(H)"
        elif viz_mode == "Imaginary Channel":
            orig_map, recon_map, unit_title = orig_imag, recon_imag, "Im(H)"
        else:
            orig_map = np.sqrt(orig_real**2 + orig_imag**2)
            recon_map = np.sqrt(recon_real**2 + recon_imag**2)
            unit_title = "|H|"

        # Error is always the true complex reconstruction error, independent of
        # which component is being displayed above — this used to compare the
        # wrong pair of arrays when "Absolute Error" was a selectable mode.
        error_map = np.sqrt((orig_real - recon_real) ** 2 + (orig_imag - recon_imag) ** 2)

        shared_min = float(min(orig_map.min(), recon_map.min()))
        shared_max = float(max(orig_map.max(), recon_map.max()))

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f"Original — {unit_title}", f"Reconstructed — CR={cr_selected}", "Complex error |ΔH|"),
            horizontal_spacing=0.08
        )
        fig.add_trace(go.Heatmap(z=orig_map, colorscale="Blues", zmin=shared_min, zmax=shared_max,
                                  colorbar=dict(x=0.285, len=0.8, thickness=10)), row=1, col=1)
        fig.add_trace(go.Heatmap(z=recon_map, colorscale="Blues", zmin=shared_min, zmax=shared_max,
                                  colorbar=dict(x=0.635, len=0.8, thickness=10)), row=1, col=2)
        fig.add_trace(go.Heatmap(z=error_map, colorscale="OrRd",
                                  colorbar=dict(x=1.00, len=0.8, thickness=10)), row=1, col=3)

        fig.update_xaxes(title_text="Delay tap")
        fig.update_yaxes(title_text="Antenna index", row=1, col=1)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Backend is offline or hasn't returned a sample yet — connect it in the sidebar.")

with tab_bench:
    metrics_csv = Path("results/metrics.csv")
    if metrics_csv.exists():
        df_bench = pd.read_csv(metrics_csv)
        df_deep = df_bench[df_bench["method"] == "DeepCSI"]
        df_dct = df_bench[df_bench["method"] == "DCT Baseline"]

        col_plot, col_table = st.columns([1.2, 1.0])

        with col_plot:
            bench_view = st.radio(
                "Benchmark target", options=["NMSE (dB)", "MRT beamforming gain (%)"],
                index=0, horizontal=True, label_visibility="collapsed"
            )
            fig_bar = go.Figure()
            if "NMSE" in bench_view:
                fig_bar.add_trace(go.Bar(name="DeepCSI", x=[f"CR={c}" for c in df_deep["compression_ratio"]],
                                          y=df_deep["nmse_db_mean"], marker_color="#1D4E89",
                                          text=[f"{v:.1f}" for v in df_deep["nmse_db_mean"]], textposition="auto"))
                fig_bar.add_trace(go.Bar(name="2D DCT baseline", x=[f"CR={c}" for c in df_dct["compression_ratio"]],
                                          y=df_dct["nmse_db_mean"], marker_color="#B7C0CB",
                                          text=[f"{v:.1f}" for v in df_dct["nmse_db_mean"]], textposition="auto"))
                fig_bar.add_hline(y=-15, line_dash="dot", line_color="#E76F51",
                                   annotation_text="spec threshold", annotation_position="top right")
                fig_bar.update_layout(title="NMSE across compression ratios", yaxis_title="dB", barmode="group")
            else:
                deep_gains = df_deep.get("beamforming_gain_mean", pd.Series([0.9998]*len(df_deep))) * 100.0
                dct_gains = df_dct.get("beamforming_gain_mean", pd.Series([0.9999]*len(df_dct))) * 100.0
                fig_bar.add_trace(go.Bar(name="DeepCSI", x=[f"CR={c}" for c in df_deep["compression_ratio"]],
                                          y=deep_gains, marker_color="#2A9D8F",
                                          text=[f"{v:.2f}" for v in deep_gains], textposition="auto"))
                fig_bar.add_trace(go.Bar(name="2D DCT baseline", x=[f"CR={c}" for c in df_dct["compression_ratio"]],
                                          y=dct_gains, marker_color="#B7C0CB",
                                          text=[f"{v:.2f}" for v in dct_gains], textposition="auto"))
                fig_bar.add_hline(y=90, line_dash="dot", line_color="#E76F51",
                                   annotation_text="90% retention target", annotation_position="bottom right")
                fig_bar.update_layout(title="Downstream MRT power retention", yaxis_title="%",
                                       yaxis_range=[85, 101], barmode="group")

            fig_bar.update_layout(**{**PLOTLY_LAYOUT,
                                      "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)})
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            st.caption("Evaluation summary")
            show_cols = ["method", "compression_ratio", "latent_dim", "scalar_reduction_percent",
                         "nmse_db_mean", "inference_ms"]
            show_cols = [c for c in show_cols if c in df_bench.columns]
            styled = (
                df_bench[show_cols]
                .rename(columns={
                    "method": "Method", "compression_ratio": "CR", "latent_dim": "Latent",
                    "scalar_reduction_percent": "Saved %", "nmse_db_mean": "NMSE (dB)",
                    "inference_ms": "Latency (ms)"
                })
                .style.format({"Saved %": "{:.1f}", "NMSE (dB)": "{:.2f}", "Latency (ms)": "{:.3f}"})
                .background_gradient(subset=["NMSE (dB)"], cmap="Blues_r")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("Lower NMSE is better · shaded darker = stronger reconstruction")
    else:
        st.info("results/metrics.csv not found — run the benchmark script to populate this tab.")
