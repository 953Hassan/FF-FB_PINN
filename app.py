import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from functools import lru_cache

# --- DATA MANAGEMENT ---
PRESET_DIR = "presets"
preset_files = sorted([f for f in os.listdir(PRESET_DIR) if f.endswith('.npz')], reverse=True)
preset_names = {f.replace(".npz", "").replace("_", " ").title(): f for f in preset_files}

all_options = []
for mode_name in preset_names.keys():
    for model in ["PINN", "FFPINN", "FBPINN", "FFFBPINN"]:
        all_options.append(f"{mode_name}: {model}")

@lru_cache(maxsize=32)
def load_data_cached(mode_name):
    filename = preset_names[mode_name]
    return np.load(os.path.join(PRESET_DIR, filename), allow_pickle=True)

def create_dashboard(selected_combinations, t_val):
    if not selected_combinations:
        return [None]*5 + ["### Please select at least one Architectural Solution at Run combination."]

    plot_data = [] 
    x, t, ref = None, None, None

    for combo in selected_combinations:
        mode_part, model_part = combo.split(": ")
        data = load_data_cached(mode_part)
        
        if x is None:
            x, t, ref = data['x_coords'], data['t_coords'], data['reference']
        
        plot_data.append({
            "label": combo,
            "u": data[f"u_{model_part}"],
            "metrics": data['metrics'][0][model_part],
            "runtime": data['runtimes'][0].get(model_part, 0)
        })

    # Colorbar Normalization
    z_min, z_max = float(np.min(ref)), float(np.max(ref))

    # --- FIGURE 1: SOLUTION HEATMAPS ---
    total_plots = len(plot_data) + 1
    cols = 2
    rows = int(np.ceil(total_plots / cols))
    fig_sol = make_subplots(rows=rows, cols=cols, subplot_titles=["Reference"] + [p["label"] for p in plot_data])
    
    fig_sol.add_trace(go.Contour(z=ref, x=x, y=t, colorscale='Jet', zmin=z_min, zmax=z_max, showscale=True), row=1, col=1)
    for i, p in enumerate(plot_data):
        r, c = (i+1) // cols + 1, (i+1) % cols + 1
        fig_sol.add_trace(go.Contour(z=p["u"], x=x, y=t, colorscale='Jet', zmin=z_min, zmax=z_max, showscale=False), row=r, col=c)
    
    fig_sol.update_xaxes(title_text="x ∈ [-1, 1]")
    fig_sol.update_yaxes(title_text="t ∈ [0, 1]")
    fig_sol.update_layout(height=400*rows, title_text="Solution Field Comparison")

    # --- FIGURE 2: ERROR HEATMAPS (100 levels) ---
    num_err = len(plot_data)
    cols_err = 2 if num_err > 1 else 1
    rows_err = int(np.ceil(num_err / cols_err))
    fig_err = make_subplots(rows=rows_err, cols=cols_err, subplot_titles=[f"{p['label']} Absolute Error" for p in plot_data])
    
    for i, p in enumerate(plot_data):
        r, c = i // cols_err + 1, i % cols_err + 1
        error_map = np.abs(p["u"] - ref)
        fig_err.add_trace(go.Contour(z=error_map, x=x, y=t, colorscale='Inferno', ncontours=100), row=r, col=c)
    
    fig_err.update_xaxes(title_text="x ∈ [-1, 1]")
    fig_err.update_yaxes(title_text="t ∈ [0, 1]")
    fig_err.update_layout(height=400*rows_err, title_text="Error Field Comparison")

    # --- FIGURE 3: 1D TIME SLICES (Solution & Error) ---
    t_idx = np.argmin(np.abs(t - t_val))
    fig_slice = make_subplots(rows=1, cols=2, subplot_titles=(f"Solution Slice at t={t_val}", f"Absolute Error Slice at t={t_val}"))
    
    fig_slice.add_trace(go.Scatter(x=x, y=ref[t_idx, :], name="Reference", line=dict(color='black', width=3)), row=1, col=1)
    styles = ['dash', 'dot', 'dashdot']
    for i, p in enumerate(plot_data):
        # Solution
        fig_slice.add_trace(go.Scatter(x=x, y=p["u"][t_idx, :], name=p["label"], line=dict(dash=styles[i % len(styles)])), row=1, col=1)
        # Error
        err_slice = np.abs(p["u"][t_idx, :] - ref[t_idx, :])
        fig_slice.add_trace(go.Scatter(x=x, y=err_slice, name=f"{p['label']} Error"), row=1, col=2)
    
    fig_slice.update_xaxes(title_text="x ∈ [-1, 1]")
    fig_slice.update_layout(title_text="1D Cross-Section Analysis")

    # --- FIGURE 4: BAR CHARTS ---
    fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Rel. L2 Error", "Rel. L∞ Error"))
    labels = [p["label"] for p in plot_data]
    l2_vals = [p["metrics"]['l2'] for p in plot_data]
    inf_vals = [p["metrics"]['linf'] for p in plot_data]

    fig_bar.add_trace(go.Bar(x=labels, y=l2_vals, name="L2", marker_color='blue'), row=1, col=1)
    fig_bar.add_trace(go.Bar(x=labels, y=inf_vals, name="L∞", marker_color='crimson'), row=1, col=2)
    fig_bar.update_yaxes(type="log", row=1, col=1)
    fig_bar.update_yaxes(type="log", row=1, col=2)
    fig_bar.update_layout(title_text="Error Metrics (Log Scale)")

    # --- ANALYTICS TABLE ---
    baseline_l2, baseline_rt = None, None
    for p in plot_data:
        if "10K" in p["label"].upper() and "PINN" in p["label"].upper() and "FFPINN" not in p["label"].upper():
            baseline_l2, baseline_rt = p["metrics"]['l2'], p["runtime"]
            break
    if baseline_l2 is None:
        baseline_l2, baseline_rt = plot_data[0]["metrics"]['l2'], plot_data[0]["runtime"]

    stats_md = f"## Comparative Analytics (Baseline: 10k PINN if selected)\n"
    stats_md += "| Combination | Rel L2 | Rel L∞ | Runtime | % Improv. | % Slower |\n|---|---|---|---|---|---|\n"
    
    for p in plot_data:
        m, rt = p["metrics"], p["runtime"]
        l2_improv = ((baseline_l2 - m['l2']) / (baseline_l2 + 1e-12)) * 100
        slowdown = ((rt - baseline_rt) / (baseline_rt + 1e-12)) * 100 if rt > 0 and baseline_rt > 0 else 0
        slowdown_str = f"{slowdown:.1f}% slower" if slowdown > 0 else f"{abs(slowdown):.1f}% faster"
        if p['metrics']['l2'] == baseline_l2: slowdown_str = "Base"
        
        stats_md += f"| {p['label']} | {m['l2']:.2e} | {m['linf']:.2e} | {rt:.2f}s | {l2_improv:.2f}% | {slowdown_str} |\n"

    return fig_sol, fig_err, fig_slice, fig_bar, stats_md

with gr.Blocks(theme=gr.themes.Soft(), title="FFFBPINN Architecture vs Existing Architecture Demo") as demo:
    with gr.Sidebar():
        gr.Markdown("# Architecture Selection")
        model_check = gr.CheckboxGroup(choices=all_options, label="Select Architectural Solution at Run", value=[all_options[0]])
        t_slide = gr.Slider(0, 1, step=0.01, value=0.5, label="Time Slice (t)")
        gr.Markdown("---")
        gr.Info("Comparing 1k/2.5k/5k FFFBPINN vs 10k PINN baseline.")

    with gr.Column():
        with gr.Tabs():
            with gr.Tab("Solution Heatmaps"):
                sol_out = gr.Plot()
            with gr.Tab("Error Heatmaps"):
                err_out = gr.Plot()
            with gr.Tab("1D Cross-Sections"):
                slice_out = gr.Plot()
            with gr.Tab("Error Metrics"):
                bar_out = gr.Plot()
            with gr.Tab("Statistics"):
                stats_out = gr.Markdown()
            with gr.Tab("Methodology"):
                gr.Markdown("""
                ## Architectural Overview
                This project evaluates three primary Physics-Informed Neural Network (PINN) architectures for Computational Fluid Dynamics (CFD).

                * **PINN (Global MLP):** A standard multilayer perceptron that takes (x, t) as inputs and outputs the solution u.
                * **FBPINN (Finite Basis PINN):** Utilizes **Time-domain decomposition**. The domain is split into subdomains handled by local networks.
                * **FFFBPINN:** Enhances domain decomposition with **Fourier Features**. This architecture embeds spatial inputs into a higher-dimensional frequency space to capture sharp shock fronts.
                """)

    inputs = [model_check, t_slide]
    outputs = [sol_out, err_out, slice_out, bar_out, stats_out]
    model_check.change(create_dashboard, inputs, outputs)
    t_slide.change(create_dashboard, inputs, outputs)
    demo.load(create_dashboard, inputs, outputs)

demo.launch()