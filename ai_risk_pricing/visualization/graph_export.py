import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


def export_graphml(graph: nx.DiGraph, path: str | Path) -> None:
    """
    Export the dependency network to GraphML for serialization
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out)


def export_gexf(graph: nx.DiGraph, path: str | Path) -> None:
    """
    Export the dependency network to GEXF for visualization
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, out)

def export_png(graph: nx.DiGraph, path: str | Path) -> None:
    """
    Export the dependency network directly to png
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    node_types = {n: str(attrs.get("node_type", "unknown")) for n, attrs in graph.nodes(data=True)}
    exposures = {n: float(attrs.get("exposure", 0.0)) for n, attrs in graph.nodes(data=True)}

    type_palette = {
        "foundation_model": "#123A63",  # institutional blue
        "saas_provider": "#2E6F95",
        "enterprise": "#7A8A99",
        "unknown": "#4B5563",
    }
    node_colors = [type_palette.get(node_types[n], type_palette["unknown"]) for n in graph.nodes()]

    exp_vals = [max(0.0, exposures.get(n, 0.0)) for n in graph.nodes()]
    exp_max = max(exp_vals) if exp_vals else 1.0
    # scale node areas conservatively (points^2)
    node_sizes = [250.0 + 1750.0 * (e / exp_max) ** 0.85 for e in exp_vals]

    edge_weights = [float(attrs.get("weight", 1.0)) for _, _, attrs in graph.edges(data=True)]
    w_max = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.8 + 2.4 * (w / w_max) ** 0.9 for w in edge_weights]

    #tiering the layout
    tier_order = ["foundation_model", "saas_provider", "enterprise", "unknown"]
    tier_x = {tier: i for i, tier in enumerate(tier_order)}

    nodes_by_tier: dict[str, list[str]] = {t: [] for t in tier_order}
    for n in graph.nodes():
        t = node_types.get(n, "unknown")
        if t not in nodes_by_tier:
            t = "unknown"
        nodes_by_tier[t].append(str(n))

    for t in tier_order:
        nodes_by_tier[t].sort(
            key=lambda n: (
                -float(exposures.get(n, 0.0)),
                -float(graph.out_degree(n)),
                str(n),
            )
        )

    # create tiered positions with spacing that accounts for node size.
    pos: dict[str, tuple[float, float]] = {}
    for t in tier_order:
        tier_nodes = nodes_by_tier[t]
        if not tier_nodes:
            continue

        tier_sizes = [node_sizes[list(graph.nodes()).index(n)] for n in tier_nodes]
        base_gap = 1.35
        size_gap = 0.55
        gaps = [base_gap + size_gap * (s / max(tier_sizes)) ** 0.5 for s in tier_sizes]

        y = 0.0
        ys: list[float] = []
        for g in gaps:
            ys.append(y)
            y -= g

        y_mid = (max(ys) + min(ys)) / 2.0
        ys = [yy - y_mid for yy in ys]

        x = float(tier_x[t]) * 3.2
        for n, yy in zip(tier_nodes, ys, strict=True):
            pos[n] = (x, yy)

    fig, ax = plt.subplots(figsize=(15, 9.5), facecolor="white")
    ax.set_title("AI Supply Chain Dependency Network", fontweight="bold", pad=14)
    ax.axis("off")

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=edge_widths,
        edge_color="#6B7280",
        alpha=0.55,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="white",
        alpha=0.95,
    )

    labels = {n: str(n).replace("_", " ") for n in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=labels,
        font_size=9,
        font_color="#111827",
        font_weight="medium",
        ax=ax,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
    )

    edge_labels = {(u, v): f"{float(attrs.get('weight', 1.0)):.2f}" for u, v, attrs in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="#374151",
        rotate=False,
        label_pos=0.55,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.3},
        ax=ax,
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=type_palette["foundation_model"], markersize=10, label="Foundation Model"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=type_palette["saas_provider"], markersize=10, label="SaaS Provider"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=type_palette["enterprise"], markersize=10, label="Enterprise"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_concentration_index(graph: nx.DiGraph) -> float:
    """
    Compute a systemic concentration index (H) using the Herfindahl index.

    Definition:
        Let w_i be each node's share of *systemic weighted exposure*.
        Then the concentration (Herfindahl) index is:

            H = sum_i w_i^2

        This is a portfolio-wide systemic concentration score, analogous to a
        market HHI, but applied to "systemic weighted exposure" rather than
        market share. Higher H indicates that losses are dominated by a
        small set of nodes (single points of failure / high criticality),
        implying higher systemic risk under correlated AI failure modes.

        Weighting scheme:
            systemic_weight_i = exposure_i * criticality_i * dependency_weight_i
            w_i = systemic_weight_i / sum_j systemic_weight_j

        This aligns with the catastrophe interpretation that exposure is the
        gross loss base, while criticality and dependency weight scale the
        expected transmitted loss under cascades.
    """
    systemic_weights: list[float] = []
    for _, attrs in graph.nodes(data=True):
        exposure = float(attrs.get("exposure", 0.0))
        criticality = float(attrs.get("criticality_score", attrs.get("criticality", 1.0)))
        dependency_weight = float(attrs.get("dependency_weight", 1.0))
        w = max(0.0, exposure) * max(0.0, criticality) * max(0.0, dependency_weight)
        systemic_weights.append(w)

    total = sum(systemic_weights)
    if total <= 0.0:
        return 0.0

    shares = [w / total for w in systemic_weights]
    return float(sum(s * s for s in shares))

