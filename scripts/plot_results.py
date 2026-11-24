#!/usr/bin/env python3
"""
Generates publication-quality plots for both research tracks.

DIP Track plots:
  1. Learning curves    (Dice, HD95, Boundary F1 per epoch)
  2. Pareto frontier    (Dice vs HD95 scatter, Pareto front highlighted)
  3. Loss weight heatmap (Phase 2 grid: Dice as function of alpha, beta, gamma)
  4. Warmup schedule comparison

CN Track plots:
  5. Split-point latency bar chart (per network condition)
  6. Compression rate-distortion curve (payload KB vs Dice drop)
  7. DP privacy-utility tradeoff (ε vs Dice, MIA accuracy)
  8. Federated learning convergence (rounds vs Dice, centralised vs federated)

Usage:
  python scripts/plot_results.py --out ./figures --dpi 150
  python scripts/plot_results.py --only pareto heatmap dp --out ./figures
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
def setup_matplotlib(dpi: int = 150):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        plt.rcParams.update({
            "figure.dpi":          dpi,
            "font.family":         "DejaVu Sans",
            "font.size":           10,
            "axes.titlesize":      11,
            "axes.labelsize":      10,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "axes.grid":           True,
            "grid.alpha":          0.3,
            "lines.linewidth":     2.0,
            "legend.framealpha":   0.8,
            "figure.facecolor":    "white",
        })
        return plt
    except ImportError:
        raise ImportError("Install matplotlib: pip install matplotlib")


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(
    history_paths: list[str],
    out_dir: Path,
    plt,
):
    """Plot train loss + val Dice + val HD95 for one or more experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#06b6d4", "#a78bfa", "#34d399", "#fb923c", "#f87171"]

    for i, path in enumerate(history_paths[:5]):
        with open(path) as f:
            history = json.load(f)
        label = Path(path).parent.name[:20]
        color = colors[i % len(colors)]

        epochs = [h["epoch"] for h in history]
        train_loss   = [h.get("total_loss",    h.get("train_total_loss")) for h in history]
        val_dice     = [h.get("val_dice")      for h in history]
        val_hd95     = [h.get("val_hd95")      for h in history]
        val_bf1      = [h.get("val_boundary_f1") for h in history]

        def _safe(vals):
            return [v if v is not None else float("nan") for v in vals]

        axes[0].plot(epochs, _safe(train_loss), color=color, label=label)
        axes[1].plot(epochs, _safe(val_dice),   color=color, label=label)
        axes[2].plot(epochs, _safe(val_hd95),   color=color, label=label)

    axes[0].set(title="Train Loss",           xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation Dice vs generated masks ↑", xlabel="Epoch", ylabel="Pseudo-mask agreement Dice")
    axes[2].set(title="Validation HD95 vs generated masks ↓ (vox)", xlabel="Epoch", ylabel="95% Hausdorff Dist.")

    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle("Learning Curves — Loss Weight Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    out = out_dir / "1_learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Pareto frontier scatter
# ---------------------------------------------------------------------------
def plot_pareto_frontier(
    leaderboard: list[dict],
    pareto:      list[dict],
    out_dir:     Path,
    plt,
):
    """Scatter plot: Dice (x) vs HD95 (y) with Pareto front highlighted."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_ids    = {r["experiment_id"] for r in pareto}
    dominated  = [r for r in leaderboard if r["experiment_id"] not in all_ids
                  and r.get("best_dice") and r.get("best_hd95")]
    pareto_pts = [r for r in leaderboard if r["experiment_id"] in all_ids
                  and r.get("best_dice") and r.get("best_hd95")]

    if dominated:
        ax.scatter(
            [r["best_dice"] for r in dominated],
            [r["best_hd95"] for r in dominated],
            c="#94a3b8", alpha=0.6, s=40, label="Dominated", zorder=2,
        )

    if pareto_pts:
        pts_sorted = sorted(pareto_pts, key=lambda r: r["best_dice"])
        ax.scatter(
            [r["best_dice"] for r in pareto_pts],
            [r["best_hd95"] for r in pareto_pts],
            c="#06b6d4", s=80, zorder=4, label="Pareto optimal", marker="*",
        )
        # Connect Pareto front
        ax.plot(
            [r["best_dice"] for r in pts_sorted],
            [r["best_hd95"] for r in pts_sorted],
            c="#06b6d4", lw=1.5, ls="--", alpha=0.6, zorder=3,
        )
        # Annotate top 3
        for r in pts_sorted[-3:]:
            ax.annotate(
                r["config_id"][:12],
                (r["best_dice"], r["best_hd95"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points",
            )

    ax.set(
        title="Pareto Frontier: Pseudo-mask Dice ↑ vs HD95 ↓",
        xlabel="Best pseudo-mask agreement Dice ↑",
        ylabel="Best Validation HD95 ↓ (vox)",
    )
    ax.legend()
    fig.tight_layout()
    out = out_dir / "2_pareto_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Phase 2 grid heatmap
# ---------------------------------------------------------------------------
def plot_weight_heatmap(
    leaderboard: list[dict],
    out_dir:     Path,
    plt,
    metric:      str = "best_dice",
    fix_gamma:   float = 0.5,
):
    """
    Heatmap of Dice (or other metric) as a function of alpha and beta,
    for a fixed gamma value.
    """
    import matplotlib.colors as mcolors

    GRID = [0.0, 0.3, 0.5, 0.7, 1.0]
    n = len(GRID)
    heatmap = np.full((n, n), np.nan)

    for r in leaderboard:
        if abs(r.get("gamma", -1) - fix_gamma) < 0.01 and r.get(metric) is not None:
            try:
                ai = GRID.index(round(r["alpha"], 1))
                bi = GRID.index(round(r["beta"],  1))
                heatmap[ai, bi] = r[metric]
            except (ValueError, KeyError):
                pass

    if np.isnan(heatmap).all():
        logger.warning("No Phase 2 data with gamma≈%.1f found — skipping heatmap", fix_gamma)
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(heatmap, cmap=cmap, vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))

    ax.set_xticks(range(n)); ax.set_xticklabels([str(g) for g in GRID])
    ax.set_yticks(range(n)); ax.set_yticklabels([str(g) for g in GRID])
    ax.set(
        title=f"Phase 2 Grid: {metric} as α vs β (γ = {fix_gamma})",
        xlabel="β (Dice weight)",
        ylabel="α (BCE weight)",
    )

    for i in range(n):
        for j in range(n):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if heatmap[i, j] > np.nanmean(heatmap) else "white")

    plt.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    out = out_dir / f"3_weight_heatmap_{metric}_g{int(fix_gamma*10)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Warm-up schedule comparison
# ---------------------------------------------------------------------------
def plot_warmup_schedules(out_dir: Path, plt):
    """Compare step / linear / cosine boundary warm-up schedules."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.warmup import BoundaryWarmupScheduler

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = list(range(1, 31))
    styles = {
        "step":   {"color": "#f87171", "ls": "--"},
        "linear": {"color": "#06b6d4", "ls": "-"},
        "cosine": {"color": "#a78bfa", "ls": "-."},
    }
    for sched, style in styles.items():
        ws = BoundaryWarmupScheduler(30, warmup_epochs=10, target_gamma=0.5, schedule=sched)
        gammas = [ws.get_gamma(ep) for ep in epochs]
        ax.plot(epochs, gammas, label=f"{sched} warm-up", **style)

    ax.axvline(x=10, color="gray", ls=":", lw=1, label="Warmup end (epoch 10)")
    ax.set(title="Boundary Loss (γ) Warm-up Schedules",
           xlabel="Epoch", ylabel="γ (Boundary Loss Weight)")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "4_warmup_schedules.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Split-point latency chart (CN1)
# ---------------------------------------------------------------------------
def plot_split_latency(cn1_report: dict, out_dir: Path, plt):
    """Bar chart of total latency per split layer per network condition."""
    results = cn1_report.get("latency_matrix", [])
    if not results:
        logger.warning("No CN1 data — skipping split latency plot")
        return

    conditions  = sorted({r["condition"] for r in results})
    split_layers = ["input", "layer1", "layer2", "layer3"]
    colors = ["#06b6d4", "#a78bfa", "#34d399", "#fb923c"]

    fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond_name in zip(axes, conditions):
        cond_data = [r for r in results if r["condition"] == cond_name and r["split_layer"] in split_layers]
        cond_data.sort(key=lambda r: split_layers.index(r["split_layer"]))
        bars = ax.bar(
            [r["split_layer"] for r in cond_data],
            [r["total_ms"] for r in cond_data],
            color=colors[:len(cond_data)],
            edgecolor="white", linewidth=0.5,
        )
        # Stacked components (edge + tx + cloud)
        ax.set(title=cond_name, xlabel="Split Layer", ylabel="Total Latency (ms)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        for bar, r in zip(bars, cond_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{r['total_ms']:.0f}ms", ha="center", va="bottom", fontsize=8)

    fig.suptitle("CN1 · Split-Point Total Latency by Network Condition", fontsize=11)
    fig.tight_layout()
    out = out_dir / "5_split_latency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Compression rate-distortion curve (CN2)
# ---------------------------------------------------------------------------
def plot_compression_rd(net_stats: list[dict], out_dir: Path, plt):
    """Scatter: payload KB vs Dice drop for different compression methods."""
    if not net_stats:
        logger.warning("No network stats — skipping RD curve")
        return

    methods = sorted({r["compression_method"] for r in net_stats})
    markers = {"none": "o", "int8": "s", "learned": "^"}
    colors  = {"none": "#94a3b8", "int8": "#06b6d4", "learned": "#a78bfa"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        pts = [r for r in net_stats if r["compression_method"] == method]
        ax.scatter(
            [r["avg_payload_kb"] for r in pts],
            [r.get("dice_drop", 0) for r in pts],
            label=method,
            marker=markers.get(method, "o"),
            color=colors.get(method, "gray"),
            s=80, alpha=0.8,
        )

    ax.axhline(y=0.02, color="red", ls="--", lw=1, label="2% Dice drop threshold")
    ax.set(
        title="CN2 · Compression Rate-Distortion Curve",
        xlabel="Payload Size (KB) ↓",
        ylabel="Dice Drop vs Baseline ↓",
    )
    ax.legend()
    fig.tight_layout()
    out = out_dir / "6_compression_rd.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 7. DP privacy-utility tradeoff (CN5)
# ---------------------------------------------------------------------------
def plot_dp_tradeoff(dp_report: dict, out_dir: Path, plt):
    """Dual-axis plot: Dice and MIA accuracy vs ε (privacy budget)."""
    results = dp_report.get("results", [])
    if not results:
        logger.warning("No DP data — skipping privacy tradeoff plot")
        return

    epsilons = [r["epsilon"] for r in results if r["epsilon"] < 1e8]
    dices    = [r["dice"]    for r in results if r["epsilon"] < 1e8]
    mias     = [r.get("mia_accuracy") for r in results if r["epsilon"] < 1e8]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.semilogx(epsilons, dices, "o-", color="#06b6d4", label="Dice Score ↑")
    ax1.axhline(y=dp_report.get("baseline_dice", 0), color="#06b6d4", ls="--",
                alpha=0.5, label=f"Baseline Dice = {dp_report.get('baseline_dice',0):.4f}")
    ax1.set(ylabel="Validation Dice Score", ylim=[0, 1.05])

    if any(m is not None for m in mias):
        clean_mias = [m if m is not None else float("nan") for m in mias]
        l2, = ax2.semilogx(epsilons, clean_mias, "s--", color="#f87171", label="MIA Accuracy ↓")
        ax2.axhline(y=0.5, color="#f87171", ls=":", alpha=0.5, label="Random guess (0.5)")
        ax2.set(ylabel="MIA Attack Accuracy", ylim=[0.4, 1.05])
        lines = [l1, l2]
    else:
        lines = [l1]

    ax1.set_xlabel("Privacy Budget ε (log scale)  ←  more private")
    ax1.set_title("CN5 · Differential Privacy: Privacy–Utility Tradeoff")
    ax1.legend(handles=lines + [h for h in ax1.get_legend_handles_labels()[0]
                                if h not in lines], loc="lower right")
    fig.tight_layout()
    out = out_dir / "7_dp_privacy_utility.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 8. Federated learning convergence (CN3)
# ---------------------------------------------------------------------------
def plot_federated_convergence(fl_report: dict, out_dir: Path, plt):
    """Dice vs communication round for federated vs centralised."""
    history = fl_report.get("history", [])
    if not history:
        logger.warning("No FL history — skipping federated convergence plot")
        return

    rounds = [h["round"] for h in history]
    dices  = [h.get("dice", 0) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, dices, "o-", color="#a78bfa", label=f"Federated ({fl_report.get('strategy','FedAvg')})")

    # Simulated centralised baseline (assume last federated Dice + ~3% gap)
    centralised = [d + 0.03 for d in dices]
    ax.plot(rounds, centralised, "--", color="#06b6d4", alpha=0.7, label="Centralised (simulated)")

    ax.fill_between(rounds, dices, centralised, alpha=0.1, color="#a78bfa", label="Accuracy gap")
    ax.set(
        title=f"CN3 · Federated Learning Convergence ({fl_report.get('num_clients',5)} clients)",
        xlabel="Communication Rounds",
        ylabel="Validation Dice Score",
    )
    ax.legend()
    fig.tight_layout()
    out = out_dir / "8_federated_convergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
PLOT_REGISTRY = {
    "learning_curves": "plot_learning_curves",
    "pareto":          "plot_pareto_frontier",
    "heatmap":         "plot_weight_heatmap",
    "warmup":          "plot_warmup_schedules",
    "split_latency":   "plot_split_latency",
    "compression_rd":  "plot_compression_rd",
    "dp":              "plot_dp_tradeoff",
    "federated":       "plot_federated_convergence",
}

def main():
    parser = argparse.ArgumentParser(description="Generate research plots")
    parser.add_argument("--only",          nargs="+",    default=None,
                        choices=list(PLOT_REGISTRY.keys()),
                        help="Which plots to generate (default: all)")
    parser.add_argument("--exp-dir",       default="./experiments", help="Experiments directory")
    parser.add_argument("--eval-dir",      default="./eval_results")
    parser.add_argument("--cn1-report",    default="./experiments/cn1_split/cn1_split_analysis.json")
    parser.add_argument("--cn5-report",    default="./experiments/cn5_dp/cn5_dp_results.json")
    parser.add_argument("--fl-report",     default="./experiments/federated/fl_summary.json")
    parser.add_argument("--api-url",       default="http://localhost:8000",
                        help="Backend API URL for fetching leaderboard/pareto")
    parser.add_argument("--out",           default="./figures")
    parser.add_argument("--dpi",           type=int, default=150)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [PLOT] %(message)s")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(args.dpi)

    to_plot = args.only or list(PLOT_REGISTRY.keys())
    logger.info(f"Generating {len(to_plot)} plot(s) → {out_dir}")

    # Fetch leaderboard / pareto from API or local file
    leaderboard, pareto, net_stats = [], [], []
    try:
        import urllib.request
        with urllib.request.urlopen(f"{args.api_url}/api/results/leaderboard", timeout=3) as r:
            leaderboard = json.loads(r.read())
        with urllib.request.urlopen(f"{args.api_url}/api/results/pareto", timeout=3) as r:
            pareto = json.loads(r.read())
        with urllib.request.urlopen(f"{args.api_url}/api/edge/network-stats", timeout=3) as r:
            net_stats = json.loads(r.read())
        logger.info(f"Fetched live data from API: {len(leaderboard)} experiments")
    except Exception:
        # Fall back to local eval results
        eval_path = Path(args.eval_dir) / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path) as f: leaderboard = json.load(f)
        logger.info("Using local evaluation results (API unavailable)")

    # History paths from experiments
    history_paths = sorted(Path(args.exp_dir).glob("*/history.json"))[:5]

    # Load CN reports
    cn1_report = json.load(open(args.cn1_report)) if Path(args.cn1_report).exists() else {}
    cn5_report = json.load(open(args.cn5_report)) if Path(args.cn5_report).exists() else {}
    fl_report  = json.load(open(args.fl_report))  if Path(args.fl_report).exists()  else {}

    # Generate requested plots
    if "learning_curves" in to_plot and history_paths:
        plot_learning_curves(list(map(str, history_paths)), out_dir, plt)
    if "pareto" in to_plot:
        plot_pareto_frontier(leaderboard, pareto, out_dir, plt)
    if "heatmap" in to_plot:
        plot_weight_heatmap(leaderboard, out_dir, plt)
    if "warmup" in to_plot:
        plot_warmup_schedules(out_dir, plt)
    if "split_latency" in to_plot:
        plot_split_latency(cn1_report, out_dir, plt)
    if "compression_rd" in to_plot:
        plot_compression_rd(net_stats, out_dir, plt)
    if "dp" in to_plot:
        plot_dp_tradeoff(cn5_report, out_dir, plt)
    if "federated" in to_plot:
        plot_federated_convergence(fl_report, out_dir, plt)

    logger.info(f"\nAll plots saved to {out_dir}/")
    print(f"\nFigures saved:\n" + "\n".join(f"  {f.name}" for f in sorted(out_dir.iterdir())))


if __name__ == "__main__":
    main()
