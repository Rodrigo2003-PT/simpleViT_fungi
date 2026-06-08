from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from stratified_sampling import (
    StratifiedSample,
    SpatialMetricsSummary,
)

ArrayLike1D = Union[np.ndarray, Sequence[float]]

def _as_1d_float(x: ArrayLike1D, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like. Got shape {arr.shape}.")
    return arr

def _as_2d_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    return arr

def _assert_square_patches(vec: np.ndarray, num_patches_side: int, name: str) -> None:
    expected = num_patches_side * num_patches_side
    if vec.size != expected:
        raise ValueError(
            f"{name} has length {vec.size}, but num_patches_side^2 = {expected}. "
            f"Check patch/grid parameters."
        )


class SpatialAttentionVisualizer:
    """
    Spatial map visualizer with stratified sampling support.
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "figure.facecolor": "white",
                "savefig.dpi": self.dpi,
            }
        )

    # ---------- Generic plotting primitive ----------

    def plot_patch_scalar_map(
        self,
        values: ArrayLike1D,
        num_patches_side: int,
        title: str,
        save_name: str,
        cbar_label: str,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        report_scalars: bool = True, 
    ) -> str:
        """
        Plot a patch-level scalar map as a num_patches_side × num_patches_side heatmap.
        """
        values = _as_1d_float(values, "values")
        _assert_square_patches(values, num_patches_side, "values")

        grid = values.reshape(num_patches_side, num_patches_side)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)

        # Compute and report scalar statistics
        if report_scalars:
            l1_norm = float(np.sum(np.abs(values)))
            l2_norm = float(np.linalg.norm(values))
            max_val = float(np.max(np.abs(values)))
            
            title_with_stats = (
                f"{title}\n"
                f"L1={l1_norm:.3f} | L2={l2_norm:.3f} | max={max_val:.3f}"
            )
        else:
            title_with_stats = title

        ax.set_title(title_with_stats, fontweight="bold", pad=12)
        ax.set_xlabel("Patch column", fontweight="bold")
        ax.set_ylabel("Patch row", fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)

    # ---------- Rollout and attribution ----------

    def plot_rollout_relevance_heatmap(
        self,
        rollout_relevance: Union[np.ndarray, ArrayLike1D],
        num_patches_side: int,
        sample_idx: int = 0,
        title: str = "Attention-flow rollout relevance (mean pooling)",
        save_name: str = "rollout_relevance.png",
        show: bool = False,
    ) -> str:
        arr = np.asarray(rollout_relevance, dtype=np.float32)
        if arr.ndim == 2:
            if sample_idx < 0 or sample_idx >= arr.shape[0]:
                raise IndexError(f"sample_idx {sample_idx} out of range")
            vec = arr[sample_idx]
            title_use = f"{title} | sample {sample_idx}"
        elif arr.ndim == 1:
            vec = arr
            title_use = title
        else:
            raise ValueError(f"rollout_relevance must be 1D or 2D. Got {arr.shape}")

        return self.plot_patch_scalar_map(
            values=vec,
            num_patches_side=num_patches_side,
            title=title_use,
            save_name=save_name,
            cbar_label="Attention-flow relevance (a.u.)",
            cmap="viridis",
            show=show,
        )
    
    def plot_decision_attribution_heatmap(
        self,
        attribution: Union[np.ndarray, ArrayLike1D],
        num_patches_side: int,
        sample_idx: int = 0,
        title: str = "Decision-linked attribution (class-conditional)",
        save_name: str = "decision_attribution.png",
        show: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        report_scalars: bool = True,
    ) -> str:
        """
        Plot Integrated Gradients attribution heatmap.
        
        Args:
            attribution: Attribution vector(s)
            num_patches_side: Grid size
            sample_idx: Which sample to plot (if 2D input)
            title: Plot title
            save_name: Output filename
            show: Display figure
            vmin: Minimum colorbar value (None = auto)
            vmax: Maximum colorbar value (None = auto)
            report_scalars: Include L1/L2/max in title
            
        Returns:
            Path to saved figure
        """
        arr = np.asarray(attribution, dtype=np.float32)
        if arr.ndim == 2:
            if sample_idx < 0 or sample_idx >= arr.shape[0]:
                raise IndexError(f"sample_idx {sample_idx} out of range")
            vec = arr[sample_idx]
            title_use = f"{title} | sample {sample_idx}"
        elif arr.ndim == 1:
            vec = arr
            title_use = title
        else:
            raise ValueError(f"attribution must be 1D or 2D. Got {arr.shape}")

        return self.plot_patch_scalar_map(
            values=vec,
            num_patches_side=num_patches_side,
            title=title_use,
            save_name=save_name,
            cbar_label="Attribution magnitude (a.u.)",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            show=show,
            report_scalars=report_scalars,
        )
    
    def plot_ig_comparison_panel(
        self,
        attributions_dict: Dict[str, np.ndarray],
        num_patches_side: int,
        sample_indices: Dict[str, int],
        title: str = "Integrated Gradients Comparison",
        save_name: str = "ig_comparison_panel.png",
        show: bool = False,
    ) -> str:
        """
        Plot IG attribution maps with SHARED colorbar scale for valid comparison.
        
        Args:
            attributions_dict: {"TP": array, "FP": array, ...} - one vector per group
            num_patches_side: Grid size
            sample_indices: {"TP": 123, "FP": 456, ...} - which sample for each group
            title: Overall title
            save_name: Output filename
            show: Display figure
            
        Returns:
            Path to saved figure
        """
        n_groups = len(attributions_dict)
        if n_groups == 0:
            raise ValueError("attributions_dict is empty")
        
        # Compute global vmax across ALL samples
        all_values = []
        for group_name, attr_array in attributions_dict.items():
            idx = sample_indices[group_name]
            if attr_array.ndim == 2:
                vec = attr_array[idx]
            else:
                vec = attr_array
            all_values.append(np.abs(vec))
        
        global_vmax = np.max(np.concatenate([v.flatten() for v in all_values]))
        
        # Create subplots
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5.5))
        if n_groups == 1:
            axes = [axes]
        
        for ax, (group_name, attr_array) in zip(axes, attributions_dict.items()):
            idx = sample_indices[group_name]
            
            if attr_array.ndim == 2:
                vec = attr_array[idx]
            else:
                vec = attr_array
            
            # Compute statistics
            l1_norm = float(np.sum(np.abs(vec)))
            l2_norm = float(np.linalg.norm(vec))
            max_val = float(np.max(np.abs(vec)))
            
            # Plot with shared vmax
            grid = vec.reshape(num_patches_side, num_patches_side)
            im = ax.imshow(grid, cmap="magma", interpolation="nearest", 
                        vmin=0, vmax=global_vmax)
            
            ax.set_title(
                f"{group_name} (sample {idx})\n"
                f"L1={l1_norm:.3f} | L2={l2_norm:.3f} | max={max_val:.3f}",
                fontweight="bold", fontsize=10
            )
            ax.set_xlabel("Patch column")
            ax.set_ylabel("Patch row")
            
            # Colorbar per subplot (but all have same range)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attribution", fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f" Saved: {save_path.name}")
        return str(save_path)

class SlideConditionedVisualizer:
    """
    Slide-conditioned error visualization (unchanged from original).
    """

    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "figure.facecolor": "white",
                "savefig.dpi": self.dpi,
            }
        )

    def plot_slide_conditioned_errors(
        self,
        projection: np.ndarray,
        slide_ids: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str],
        method_name: str = "t-SNE",
        save_name: str = "slide_conditioned_errors.png",
        show: bool = False,
        max_slides_per_error: int = 5,
    ) -> str:
        proj = np.asarray(projection, dtype=np.float32)
        if proj.ndim != 2 or proj.shape[1] != 2:
            raise ValueError(f"projection must be (N, 2). Got {proj.shape}")

        slide_ids = np.asarray(slide_ids)
        labels = np.asarray(labels).astype(int)
        predictions = np.asarray(predictions).astype(int)

        tp = (labels == 1) & (predictions == 1)
        tn = (labels == 0) & (predictions == 0)
        fp = (labels == 0) & (predictions == 1)
        fn = (labels == 1) & (predictions == 0)

        fp_slides = np.unique(slide_ids[fp])[:max_slides_per_error]
        fn_slides = np.unique(slide_ids[fn])[:max_slides_per_error]

        n_fp = len(fp_slides)
        n_fn = len(fn_slides)

        if n_fp == 0 and n_fn == 0:
            print("No FP/FN errors to visualize")
            return ""

        n_rows = max(n_fp, n_fn)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, sid in enumerate(fp_slides):
            ax = axes[i, 0]
            slide_mask = slide_ids == sid
            fp_in_slide = slide_mask & fp
            tn_in_slide = slide_mask & tn

            ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, alpha=0.08)
            if np.any(tn_in_slide):
                ax.scatter(proj[tn_in_slide, 0], proj[tn_in_slide, 1],
                          c="blue", s=90, alpha=0.75, marker="o",
                          edgecolors="black", linewidth=0.8,
                          label=f"TN (n={int(np.sum(tn_in_slide))})")
            if np.any(fp_in_slide):
                ax.scatter(proj[fp_in_slide, 0], proj[fp_in_slide, 1],
                          c="red", s=140, alpha=0.90, marker="X",
                          edgecolors="black", linewidth=1.2,
                          label=f"FP (n={int(np.sum(fp_in_slide))})")

            ax.set_title(f"Slide {sid} (GT={class_names[0]})", fontweight="bold")
            ax.set_xlabel(f"{method_name} dim 1")
            ax.set_ylabel(f"{method_name} dim 2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        for i in range(len(fp_slides), n_rows):
            axes[i, 0].axis("off")

        for i, sid in enumerate(fn_slides):
            ax = axes[i, 1]
            slide_mask = slide_ids == sid
            fn_in_slide = slide_mask & fn
            tp_in_slide = slide_mask & tp

            ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, alpha=0.08)
            if np.any(tp_in_slide):
                ax.scatter(proj[tp_in_slide, 0], proj[tp_in_slide, 1],
                          c="green", s=90, alpha=0.75, marker="o",
                          edgecolors="black", linewidth=0.8,
                          label=f"TP (n={int(np.sum(tp_in_slide))})")
            if np.any(fn_in_slide):
                ax.scatter(proj[fn_in_slide, 0], proj[fn_in_slide, 1],
                          c="orange", s=140, alpha=0.90, marker="X",
                          edgecolors="black", linewidth=1.2,
                          label=f"FN (n={int(np.sum(fn_in_slide))})")

            ax.set_title(f"Slide {sid} (GT={class_names[1]})", fontweight="bold")
            ax.set_xlabel(f"{method_name} dim 1")
            ax.set_ylabel(f"{method_name} dim 2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        for i in range(len(fn_slides), n_rows):
            axes[i, 1].axis("off")

        fig.suptitle(
            f"Slide-conditioned errors in {method_name} space",
            fontsize=15, fontweight="bold", y=0.995
        )

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)

    def plot_slide_error_summary(
        self,
        slide_ids: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_name: str = "slide_error_summary.png",
        show: bool = False,
        statistical_classification: Optional[Dict] = None,
    ) -> str:
        slide_ids = np.asarray(slide_ids)
        labels = np.asarray(labels).astype(int)
        predictions = np.asarray(predictions).astype(int)

        unique_slides = np.unique(slide_ids)
        slide_error_rates = []
        slide_sizes = []
        slide_gt = []

        for sid in unique_slides:
            mask = slide_ids == sid
            slide_sizes.append(int(np.sum(mask)))
            slide_gt.append(int(labels[mask][0]))
            slide_error_rates.append(float(np.mean(labels[mask] != predictions[mask])))

        slide_error_rates = np.asarray(slide_error_rates, dtype=np.float32)
        slide_sizes = np.asarray(slide_sizes, dtype=np.int32)
        slide_gt = np.asarray(slide_gt, dtype=np.int32)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        ax1 = axes[0]
        ax1.hist(slide_error_rates, bins=20, color="slategray", edgecolor="black", alpha=0.75)
        ax1.axvline(slide_error_rates.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {slide_error_rates.mean():.3f}")
        ax1.set_title("Slide-level error rate distribution", fontweight="bold")
        ax1.set_xlabel("Error rate")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        ax2 = axes[1]
        for cls, color in [(0, "royalblue"), (1, "seagreen")]:
            m = slide_gt == cls
            if np.any(m):
                ax2.hist(slide_error_rates[m], bins=15, alpha=0.55, color=color,
                        edgecolor="black", label=f"Class {cls} (n={int(np.sum(m))})")
        ax2.set_title("Error rate by GT class", fontweight="bold")
        ax2.set_xlabel("Error rate")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.25)
        ax2.legend()

        ax3 = axes[2]
        colors = np.where(slide_gt == 0, "royalblue", "seagreen")
        ax3.scatter(slide_sizes, slide_error_rates, c=colors, s=90, alpha=0.7,
                   edgecolors="black", linewidth=0.4)
        ax3.set_title("Slide size vs error rate", fontweight="bold")
        ax3.set_xlabel("Slide size (patches)")
        ax3.set_ylabel("Error rate")
        ax3.grid(True, alpha=0.25)

        if statistical_classification is not None:
            cls = statistical_classification.get("classification", "N/A")
            conf = statistical_classification.get("confidence", None)
            parts = [f"Statistical: {cls}"]
            if conf is not None:
                parts.append(f"conf={conf:.3f}")
            fig.suptitle(" | ".join(parts), fontsize=12, fontweight="bold", y=1.02)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return str(save_path)

def plot_metric_distributions_by_group(
    metric_dict: Dict[str, np.ndarray],
    metric_name: str,
    save_path: str,
    ylabel: str = "Metric Value",
    title: str = "Metric Distribution by Group",
    dpi: int = 300,
    show: bool = False,
) -> str:
    """
    
    Scientific Goal: Show statistically significant differences in attention 
    metrics (Gini, Entropy) between True Positives and False Positives.
    
    Args:
        metric_dict: Dictionary mapping group names to metric arrays
                     Example: {"TP": array([...]), "FP": array([...])}
        metric_name: Name of metric (for title)
        save_path: Output file path
        ylabel: Y-axis label
        title: Plot title
        dpi: Resolution
        show: Display figure
        
    Returns:
        Path to saved figure
        
    References:
    - Tukey (1977): Exploratory Data Analysis
    - McGill et al. (1978): Variations of Box Plots
    """
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors for each group
    color_map = {
        "TP": "#2ecc71",  # Green
        "TN": "#3498db",  # Blue
        "FP": "#e74c3c",  # Red
        "FN": "#f39c12",  # Orange
    }
    
    # Prepare data for boxplot
    data = []
    labels = []
    colors = []
    
    # Order: TN, FP, TP, FN (negative class, then positive class)
    order = ["TN", "FP", "TP", "FN"]
    
    for group_name in order:
        if group_name in metric_dict and len(metric_dict[group_name]) > 0:
            data.append(metric_dict[group_name])
            n = len(metric_dict[group_name])
            mean_val = np.mean(metric_dict[group_name])
            labels.append(f"{group_name}\n(n={n})\nμ={mean_val:.3f}")
            colors.append(color_map[group_name])
    
    if len(data) == 0:
        print(f"Warning: No data to plot for {metric_name}")
        return ""
    
    # Create boxplot
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(color="darkred", linewidth=2, linestyle="--"),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    
    # Color boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add scatter overlay (jittered points for transparency)
    for i, (group_data, color) in enumerate(zip(data, colors), start=1):
        # Subsample if too many points
        if len(group_data) > 200:
            plot_data = np.random.choice(group_data, size=200, replace=False)
        else:
            plot_data = group_data
        
        # Jitter x positions
        x_jitter = np.random.normal(i, 0.04, size=len(plot_data))
        ax.scatter(
            x_jitter,
            plot_data,
            alpha=0.3,
            s=20,
            color=color,
            edgecolors="none",
        )
    
    # Statistical annotation (if comparing TP vs FP)
    if "TP" in metric_dict and "FP" in metric_dict:
        tp_data = metric_dict["TP"]
        fp_data = metric_dict["FP"]
        
        if len(tp_data) >= 2 and len(fp_data) >= 2:
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(tp_data, fp_data, equal_var=False)
            
            # Cohen's d effect size
            mean_tp = np.mean(tp_data)
            mean_fp = np.mean(fp_data)
            pooled_std = np.sqrt((np.var(tp_data, ddof=1) + np.var(fp_data, ddof=1)) / 2)
            cohens_d = (mean_tp - mean_fp) / pooled_std if pooled_std > 0 else 0
            
            # Annotation
            y_max = max(np.max(tp_data), np.max(fp_data))
            y_annot = y_max * 1.1
            
            sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            annot_text = f"TP vs FP: p={p_val:.4f} {sig_text}\nCohen's d={cohens_d:.3f}"
            
            ax.text(
                0.98, 0.98,
                annot_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
    
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Add legend for mean/median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
        Line2D([0], [0], color="darkred", linewidth=2, linestyle="--", label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"  Saved: {Path(save_path).name}")
    return save_path