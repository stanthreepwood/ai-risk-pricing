"""
Exceedance probability curve visualization.

Generates professional-quality loss exceedance plots for catastrophe
model outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Sequence


class ExceedanceCurve:
    """
    Generator for loss exceedance probability curves.
    
    The Exceedance Probability (EP) curve is the fundamental output
    visualization for catastrophe models. It shows the relationship
    between loss severity and the probability of exceeding that loss.
    
    Actuarial interpretation:
        The EP curve answers: "What is the probability of exceeding
        loss amount X in any given year?"
        
        Reading the curve:
        - Y-axis (probability) → X-axis (loss): "There is a Y% chance
          of exceeding $X loss"
        - X-axis (loss) → Y-axis (probability): "$X loss has a Y%
          annual exceedance probability"
        
        Return periods are derived from exceedance probability:
        T = 1 / P(exceed)
        
        For example, if P(exceed $100M) = 1%, then $100M is a
        1-in-100 year loss.
    """
    
    def __init__(
        self,
        year_loss_table: pd.DataFrame,
        title: str = "AI Catastrophe Loss Exceedance Curve",
    ) -> None:
        """
        Initialize the exceedance curve generator.
        
        Args:
            year_loss_table: DataFrame with 'loss' column.
            title: Title for the plot.
        """
        if "loss" not in year_loss_table.columns:
            raise ValueError("Year Loss Table must have 'loss' column")
        
        self.losses = np.sort(year_loss_table["loss"].values)[::-1]  # Sort descending
        self.n_years = len(self.losses)
        self.title = title
    
    def compute_exceedance_probabilities(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute exceedance probabilities for all loss values.
        
        Uses the plotting position formula: P = rank / (n + 1)
        This provides an unbiased estimate of exceedance probability.
        
        Returns:
            Tuple of (losses, exceedance_probabilities) arrays.
        """
        # Rank-based exceedance probability
        ranks = np.arange(1, self.n_years + 1)
        exceedance_probs = ranks / (self.n_years + 1)
        
        return self.losses, exceedance_probs
    
    def get_return_period_losses(
        self,
        return_periods: Sequence[float] = (10, 50, 100, 200, 500),
    ) -> dict[float, float]:
        """
        Get losses at specific return periods.
        
        Args:
            return_periods: Return periods in years to calculate.
        
        Returns:
            Dictionary mapping return period to loss amount.
        """
        losses, probs = self.compute_exceedance_probabilities()
        
        results = {}
        for rp in return_periods:
            target_prob = 1 / rp
            # Find loss closest to this exceedance probability
            idx = np.argmin(np.abs(probs - target_prob))
            results[rp] = float(losses[idx])
        
        return results
    
    def plot(
        self,
        figsize: tuple[float, float] = (10, 7),
        show_return_periods: bool = True,
        return_periods: Sequence[float] = (10, 50, 100, 200),
        log_x: bool = True,
        log_y: bool = True,
        save_path: str | None = None,
        show: bool = True,
    ) -> Figure:
        """
        Generate the exceedance probability curve plot.
        
        Creates a professional-quality visualization showing:
        - EP curve (main plot)
        - Key return period markers
        - Clear axis labels and formatting
        
        Args:
            figsize: Figure size in inches.
            show_return_periods: Whether to mark key return periods.
            return_periods: Return periods to mark on plot.
            log_x: Use log scale for loss axis.
            log_y: Use log scale for probability axis.
            save_path: Path to save figure (optional).
            show: Whether to display the plot.
        
        Returns:
            Matplotlib Figure object.
        """
        losses, probs = self.compute_exceedance_probabilities()
        
        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        
        # Main EP curve
        ax.plot(
            losses,
            probs,
            linewidth=2.5,
            color="#1f77b4",
            label="Exceedance Probability",
            zorder=5,
        )
        
        # Fill under curve for visual emphasis
        ax.fill_between(
            losses,
            probs,
            alpha=0.15,
            color="#1f77b4",
            zorder=4,
        )
        
        # Add return period markers
        if show_return_periods:
            rp_losses = self.get_return_period_losses(return_periods)
            
            for rp, loss in rp_losses.items():
                prob = 1 / rp
                
                # Vertical line to EP curve
                ax.axhline(
                    y=prob,
                    color="#d62728",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.6,
                    zorder=3,
                )
                
                # Marker point
                ax.scatter(
                    [loss],
                    [prob],
                    s=80,
                    color="#d62728",
                    zorder=6,
                    edgecolors="white",
                    linewidths=1.5,
                )
                
                # Label
                ax.annotate(
                    f"1-in-{int(rp)}\n${loss:,.0f}M",
                    xy=(loss, prob),
                    xytext=(15, 10),
                    textcoords="offset points",
                    fontsize=9,
                    color="#333333",
                    fontweight="medium",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="#cccccc",
                        alpha=0.9,
                    ),
                    zorder=7,
                )
        
        # Axis configuration
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        
        # Labels and title
        ax.set_xlabel("Annual Aggregate Loss ($M)", fontsize=12, fontweight="medium")
        ax.set_ylabel("Annual Exceedance Probability", fontsize=12, fontweight="medium")
        ax.set_title(self.title, fontsize=14, fontweight="bold", pad=15)
        
        # Grid styling
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(left=max(1, np.min(losses[losses > 0]) * 0.5))
        ax.set_ylim(bottom=0.5 / self.n_years, top=1.0)
        
        # Right-side return period axis
        ax2 = ax.twinx()
        ax2.set_yscale("log" if log_y else "linear")
        ax2.set_ylim(ax.get_ylim())
        
        # Set return period ticks
        rp_ticks = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
        rp_tick_positions = [1 / rp for rp in rp_ticks if 1 / rp >= ax.get_ylim()[0]]
        rp_tick_labels = [f"1-in-{rp}" for rp in rp_ticks if 1 / rp >= ax.get_ylim()[0]]
        
        ax2.set_yticks(rp_tick_positions)
        ax2.set_yticklabels(rp_tick_labels, fontsize=9)
        ax2.set_ylabel("Return Period", fontsize=12, fontweight="medium")
        
        # Add statistics annotation
        el = np.mean(self.losses)
        var_99 = np.percentile(self.losses, 99)
        stats_text = f"Expected Loss: ${el:,.1f}M\nVaR 99%: ${var_99:,.1f}M\nSimulated Years: {self.n_years:,}"
        
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f8f9fa",
                edgecolor="#dee2e6",
                alpha=0.95,
            ),
            zorder=8,
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def plot_comparison(
        self,
        other_ylt: pd.DataFrame,
        other_label: str = "Alternative",
        base_label: str = "Base Case",
        figsize: tuple[float, float] = (10, 7),
        save_path: str | None = None,
        show: bool = True,
    ) -> Figure:
        """
        Plot comparison of two exceedance curves.
        
        Useful for comparing scenarios (e.g., with/without dark scenario).
        
        Args:
            other_ylt: Alternative Year Loss Table for comparison.
            other_label: Label for alternative curve.
            base_label: Label for base curve.
            figsize: Figure size.
            save_path: Path to save figure.
            show: Whether to display.
        
        Returns:
            Matplotlib Figure object.
        """
        # Compute base curve
        base_losses, base_probs = self.compute_exceedance_probabilities()
        
        # Compute other curve
        other_losses = np.sort(other_ylt["loss"].values)[::-1]
        other_ranks = np.arange(1, len(other_losses) + 1)
        other_probs = other_ranks / (len(other_losses) + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        
        # Plot both curves
        ax.plot(base_losses, base_probs, linewidth=2.5, color="#1f77b4", label=base_label)
        ax.plot(other_losses, other_probs, linewidth=2.5, color="#d62728", label=other_label, linestyle="--")
        
        # Styling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Annual Aggregate Loss ($M)", fontsize=12)
        ax.set_ylabel("Annual Exceedance Probability", fontsize=12)
        ax.set_title("Exceedance Curve Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def return_period_table(
        self,
        return_periods: Sequence[float] = (5, 10, 25, 50, 100, 200, 250, 500, 1000),
    ) -> pd.DataFrame:
        """
        Generate a return period table.
        
        Standard output format for catastrophe model results.
        
        Args:
            return_periods: Return periods to include.
        
        Returns:
            DataFrame with return periods and corresponding losses.
        """
        rp_losses = self.get_return_period_losses(return_periods)
        
        data = []
        for rp in return_periods:
            if rp in rp_losses:
                exceedance_prob = 1 / rp * 100  # As percentage
                data.append({
                    "Return Period (Years)": int(rp),
                    "Exceedance Probability (%)": f"{exceedance_prob:.2f}%",
                    "Loss ($M)": f"{rp_losses[rp]:,.2f}",
                })
        
        return pd.DataFrame(data)
