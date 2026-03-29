
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"

COLORS = {
    "prompt_only": "#7A7F87",
    "direct": "#376996",
    "soar": "#2A8A78",
    "scaffold": "#7B61A8",
    "overlay_j": "#376996",
    "overlay_q": "#2A8A78",
    "soft": "#F4F7FB",
    "edge": "#CBD5E1",
    "text": "#1F2937",
}

def rounded_box(ax, xy, w, h, text="", fc="white", ec="#CBD5E1", lw=1.2, radius=0.025, fontsize=9, color=None, ha='left', va='top'):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + 0.015, y + h - 0.015, text, fontsize=fontsize, color=color or COLORS["text"], va=va, ha=ha)
    return patch

def arrow(ax, x1, y1, x2, y2, color="#64748B", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=2, shrinkB=2))

def make_protocol_figure():
    fig, ax = plt.subplots(figsize=(12.8, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0,0),1,1,facecolor="white", edgecolor="none"))

    # Column titles
    cols = [0.03, 0.28, 0.52, 0.78]
    titles = [
        "1. Same QA semantics,\nthree output contracts",
        "2. Train one scaffold and\ndirect overlay adapters",
        "3. Residualize in effective\ndense-delta space",
        "4. Compare recovery under\nthe same strict contract",
    ]
    for x, t in zip(cols, titles):
        ax.text(x, 0.95, t, fontsize=11, fontweight='bold', color=COLORS["text"], va='top')

    # Column 1: example
    rounded_box(ax, (0.03, 0.55), 0.19, 0.31,
                "Question\nWhat is the warranty period?\n\n"
                "Context\n[S1] The battery lasts 12 hours.\n"
                "[S2] The device includes a 24-month warranty.\n"
                "[S3] Accessories sold separately.",
                fc="#F8FAFC", ec=COLORS["edge"], fontsize=8.5)
    rounded_box(ax, (0.03, 0.34), 0.19, 0.12,
                "Plain\n24 months", fc="#FFFFFF", ec=COLORS["edge"], fontsize=8.5)
    rounded_box(ax, (0.03, 0.19), 0.19, 0.12,
                'J\n{"answer":"24 months"}', fc="#EFF6FF", ec="#BFDBFE", fontsize=8.5, color=COLORS["overlay_j"])
    rounded_box(ax, (0.03, 0.04), 0.19, 0.12,
                'Q\nANSWER: 24 months\nQUOTE: "The device includes a 24-month warranty."',
                fc="#ECFDF5", ec="#A7F3D0", fontsize=7.6, color=COLORS["overlay_q"])

    # Column 2: training
    rounded_box(ax, (0.29, 0.64), 0.17, 0.16, "Frozen base LM\nQwen2.5-7B-Instruct", fc="#F8FAFC", ec=COLORS["edge"], fontsize=9)
    rounded_box(ax, (0.29, 0.41), 0.17, 0.12, r"Scaffold $\mathbf{S}$" + "\nplain-answer targets",
                fc="#F5F3FF", ec="#DDD6FE", fontsize=9, color=COLORS["scaffold"])
    rounded_box(ax, (0.29, 0.23), 0.17, 0.11, r"Direct $\mathbf{F_J}$" + "\nJSON targets",
                fc="#EFF6FF", ec="#BFDBFE", fontsize=9, color=COLORS["overlay_j"])
    rounded_box(ax, (0.29, 0.08), 0.17, 0.11, r"Direct $\mathbf{F_Q}$" + "\nquote targets",
                fc="#ECFDF5", ec="#A7F3D0", fontsize=9, color=COLORS["overlay_q"])
    for y in [0.47, 0.29, 0.14]:
        arrow(ax, 0.375, 0.64, 0.375, y)

    # Column 3: residualization
    rounded_box(ax, (0.53, 0.55), 0.2, 0.27,
                r"Per adapted module $m$" "\n\n"
                r"$\Delta_m = (\alpha_m/r_m)\, B_mA_m$" "\n\n"
                r"$\Gamma_{k,m} = \Delta_{F_k,m} - \beta_k \Delta_{S,m}$" "\n"
                r"$\Delta_{R_k,m} = \mathrm{TSVD}_r(\Gamma_{k,m})$",
                fc="#F8FAFC", ec=COLORS["edge"], fontsize=9)
    rounded_box(ax, (0.53, 0.28), 0.2, 0.11,
                r"Validation selects $\beta$ from $\{0.5,0.75,1.0,1.25\}$" "\nunder the semantic-retention guardrail",
                fc="#FFFBEB", ec="#FDE68A", fontsize=8.5, color="#7C5E10")
    rounded_box(ax, (0.53, 0.08), 0.2, 0.12,
                r"Recovered residuals" "\n" r"$\mathbf{R_J}$ and $\mathbf{R_Q}$",
                fc="#FFFFFF", ec=COLORS["edge"], fontsize=9)
    arrow(ax, 0.63, 0.54, 0.63, 0.40)
    arrow(ax, 0.63, 0.27, 0.63, 0.21)

    # Column 4: evaluation
    rounded_box(ax, (0.79, 0.62), 0.18, 0.15,
                "Prompt-only\n" + r"$M_0 + \Delta_S$",
                fc="#F9FAFB", ec=COLORS["edge"], fontsize=9)
    rounded_box(ax, (0.79, 0.40), 0.18, 0.15,
                "Direct\n" + r"$M_0 + \Delta_{F_k}$",
                fc="#F9FAFB", ec=COLORS["edge"], fontsize=9)
    rounded_box(ax, (0.79, 0.18), 0.18, 0.15,
                r"SOAR" "\n" r"$M_0 + \Delta_S + \Delta_{R_k}$",
                fc="#F0FDF4", ec="#86EFAC", fontsize=9, color=COLORS["overlay_q"])
    arrow(ax, 0.74, 0.60, 0.79, 0.69)
    arrow(ax, 0.74, 0.14, 0.79, 0.25)
    ax.text(0.79, 0.06,
            "Parser-based metrics\n- strict JSON schema validity\n- exact quote match\n- answer EM/F1",
            fontsize=8.7, color=COLORS["text"], va='bottom')

    fig.savefig(FIGDIR / "fig_protocol_overview.pdf", bbox_inches="tight")
    plt.close(fig)

def make_results_figure():
    df = pd.read_csv(DATADIR / "figure1_single_overlay_recovery.csv")
    systems = ["prompt_only", "direct", "soar"]
    labels = {"prompt_only": "Prompt-only", "direct": "Direct", "soar": "SOAR"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.0), constrained_layout=True)

    for ax, panel, title in zip(
        axes,
        ["primary", "semantic"],
        ["Active operational metric", "Answer F1 after parser extraction"],
    ):
        sub = df[df["panel"] == panel].copy()
        overlays = ["J", "Q"]
        y_positions = {"J": 1, "Q": 0}
        offsets = {"prompt_only": -0.18, "direct": 0.0, "soar": 0.18}
        for ov in overlays:
            ax.axhline(y_positions[ov], color="#E5E7EB", lw=1.0, zorder=0)
            for sys in systems:
                val = float(sub[(sub["overlay"] == ov) & (sub["system"] == sys)]["value"].iloc[0])
                y = y_positions[ov] + offsets[sys]
                ax.plot(val, y, marker='o', ms=8.5, mec='white', mew=1.2,
                        color=COLORS[sys], zorder=3)
                ax.hlines(y, 0, val, color=COLORS[sys], lw=2.2, alpha=0.95, zorder=2)
                ha = 'left' if val < 0.92 else 'right'
                x_text = val + 0.018 if ha == 'left' else val - 0.018
                ax.text(x_text, y, f"{val:.4f}", fontsize=8.2, va='center', ha=ha,
                        color=COLORS["text"])
        ax.set_xlim(-0.02, 1.03)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([1,0])
        ax.set_yticklabels(["JSON (J)", "Quote (Q)"], fontsize=10)
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
        ax.grid(axis='x', color="#E5E7EB", lw=0.8)
        ax.set_axisbelow(True)
        for spine in ["top","right","left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D1D5DB")
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.set_xlabel("Score", fontsize=10)

    legend_elements = [
        Line2D([0],[0], marker='o', color=COLORS[s], markerfacecolor=COLORS[s], markeredgecolor='white',
               markersize=8, lw=2.2, label=labels[s])
        for s in systems
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=9.5)
    fig.savefig(FIGDIR / "fig_results_overview.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    make_protocol_figure()
    make_results_figure()
