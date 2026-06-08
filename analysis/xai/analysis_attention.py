"""
analysis_attention.py

Implements stratified spatial attention analysis following:
- Cochran (1977): Sampling Techniques
- Neyman (1934): On the Two Different Aspects of the Representative Method
- Thompson (2012): Sampling, 3rd Edition
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import get_config
from embeddings import EmbeddingAnalyzer
from extractor import SimpleViTEmbeddingExtractor
from helper import create_slide_id_provenance_map, extract_slide_id_from_filename, validate_slide_grouping
from stratified_sampling import (
    StratifiedAttentionSampler,
    SpatialMetricsAggregator,
    SpatialMetricsSummary,
    AttentionDistributionalMetrics,
    AttributionMagnitudeMetrics,
    StrataDefinition,
    compute_effect_size_cohens_d,
    compare_groups_patch_level,
)
from training_utils import NPYImageFolder, TransformSubset, create_transforms, get_device_info
from visualization import SlideConditionedVisualizer, SpatialAttentionVisualizer, plot_metric_distributions_by_group


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    preprocessed_root: Path
    preprocessed_dir: Path
    output_root: Path
    analysis_root: Path

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ProjectPaths":
        project_root = Path(args.project_root).expanduser().resolve()
        preprocessed_root = Path(args.preprocessed_root).expanduser().resolve()
        preprocessed_dir = preprocessed_root / args.dataset_subdir
        output_root = Path(args.output_root).expanduser().resolve()
        analysis_root = output_root / args.experiment / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        return ProjectPaths(
            project_root=project_root,
            preprocessed_root=preprocessed_root,
            preprocessed_dir=preprocessed_dir,
            output_root=output_root,
            analysis_root=analysis_root,
        )


# -----------------------------
# Split loading
# -----------------------------
def _normalize_split_entries(split_data: Dict[str, Any], split_name: str) -> List[str]:
    k1 = f"{split_name}_base_paths"
    if k1 in split_data:
        return list(split_data[k1])
    raise KeyError(f"Split file does not contain keys for split '{split_name}'")


def load_split_with_conversion(split_path: str, dataset: NPYImageFolder, split_name: str) -> List[int]:
    with open(split_path, "r") as f:
        split_data = json.load(f)

    entries = _normalize_split_entries(split_data, split_name)

    basename_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    rel_to_idx: Dict[str, int] = {}
    relstem_to_idx: Dict[str, int] = {}

    root = Path(dataset.root).resolve()

    for idx, (filepath, _) in enumerate(dataset.samples):
        p = Path(filepath)
        basename_to_idx[p.name] = idx
        stem_to_idx[p.stem] = idx

        try:
            rel = p.resolve().relative_to(root).as_posix()
        except Exception:
            rel = p.as_posix()

        rel_to_idx[rel] = idx
        relstem_to_idx[Path(rel).with_suffix("").as_posix()] = idx

    indices: List[int] = []
    missing: List[str] = []

    for e in entries:
        if isinstance(e, int):
            indices.append(int(e))
            continue

        s = str(e)
        p = Path(s)

        candidates = [
            p.name,
            p.stem,
            s,
            p.with_suffix(".npy").name,
            p.with_suffix(".npy").as_posix(),
            Path(s).with_suffix("").as_posix(),
        ]

        found = None
        for c in candidates:
            if c in basename_to_idx:
                found = basename_to_idx[c]
                break
            if c in stem_to_idx:
                found = stem_to_idx[c]
                break
            if c in rel_to_idx:
                found = rel_to_idx[c]
                break
            if c in relstem_to_idx:
                found = relstem_to_idx[c]
                break

        if found is None:
            missing.append(s)
        else:
            indices.append(int(found))

    if missing:
        warnings.warn(f"{len(missing)}/{len(entries)} missing. First: {missing[0]}")

    return indices


# -----------------------------
# Model load
# -----------------------------
def _infer_num_classes_from_state_dict(state: Dict[str, torch.Tensor]) -> int:
    if "linear_head.weight" in state:
        return int(state["linear_head.weight"].shape[0])
    for k, v in state.items():
        if k.endswith(".weight") and v.ndim == 2 and "head" in k:
            return int(v.shape[0])
    raise ValueError("Could not infer num_classes from state_dict")


def load_model_and_checkpoint(
    model_path: str,
    cli_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)

    ckpt_config = checkpoint.get("config", None)
    if ckpt_config is None:
        warnings.warn("Checkpoint missing 'config'. Using CLI config.")
        ckpt_config = dict(cli_config)

    for k in ["image_size", "patch_size", "channels"]:
        if k in cli_config and k in ckpt_config and cli_config[k] != ckpt_config[k]:
            warnings.warn(f"Config mismatch '{k}': CLI={cli_config[k]} vs CKPT={ckpt_config[k]}")

    state = checkpoint.get("model_state_dict", None)
    if state is None:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    class_to_idx = checkpoint.get("class_to_idx", None)
    if class_to_idx is not None:
        num_classes = len(class_to_idx)
    else:
        num_classes = _infer_num_classes_from_state_dict(state)

    model_cfg = ckpt_config.get("model_config", ckpt_config.get("model_cfg", None))
    if model_cfg is None:
        model_cfg = cli_config.get("model_config", {})
        warnings.warn("No model_config in checkpoint; using CLI.")

    from vit_pytorch import SimpleViT

    model = SimpleViT(
        image_size=int(ckpt_config["image_size"]),
        patch_size=int(ckpt_config["patch_size"]),
        num_classes=int(num_classes),
        **dict(model_cfg),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint, ckpt_config


# -----------------------------
# Data loader
# -----------------------------
def create_dataloader_with_provenance(
    dataset: NPYImageFolder,
    indices: Sequence[int],
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
    batch_size: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray]:
    mean = checkpoint.get("mean", None)
    std = checkpoint.get("std", None)
    if mean is None or std is None:
        warnings.warn("Checkpoint missing mean/std")

    transform = create_transforms(mean, std, config, train=False)
    subset = TransformSubset(dataset, list(indices), transform)

    bs = int(batch_size if batch_size is not None else config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 2))

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dataset_indices = np.asarray(indices, dtype=np.int64)
    print(f"DataLoader: {len(indices)} samples | batch_size={bs}")
    return loader, dataset_indices


def reconstruct_slide_id_provenance(
    dataset: NPYImageFolder,
    dataset_indices: np.ndarray,
    validate: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    slide_ids: List[str] = []
    filepaths: List[str] = []
    labels: List[int] = []

    for idx in dataset_indices:
        filepath, label = dataset.samples[int(idx)]
        filename = Path(filepath).name
        slide_id = extract_slide_id_from_filename(filename, validate=validate)

        slide_ids.append(slide_id)
        filepaths.append(filepath)
        labels.append(int(label))

    slide_ids_arr = np.asarray(slide_ids, dtype=object)
    labels_arr = np.asarray(labels, dtype=int)

    if validate:
        ok, err = validate_slide_grouping(slide_ids_arr, labels_arr)
        if not ok:
            raise ValueError(f"Slide validation failed: {err}")

    provenance_map = create_slide_id_provenance_map(
        file_paths=filepaths,
        slide_ids=slide_ids_arr,
        labels=labels_arr,
    )
    print(f"Slide provenance: {len(np.unique(slide_ids_arr))} unique slides")
    return slide_ids_arr, provenance_map

def run_rollout_population_analysis(
    embeddings_dict: Dict[str, Any],
    slide_ids: np.ndarray,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    paths: ProjectPaths,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Rollout-based population analysis with stratified representatives.
    
    References:
    - Abnar & Zuidema (2020): Quantifying Attention Flow in Transformers
    - Chefer et al. (2021): Transformer Interpretability Beyond Attention
    - Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks
    """
    
    if "attention_storage_paths" not in embeddings_dict:
        print(" No attention data. Re-run with --extract-attention.")
        return {}
    
    print("\n" + "="*70)
    print("ROLLOUT POPULATION ANALYSIS")
    print("="*70)
    
    # Extract predictions and labels
    y_true = np.asarray(embeddings_dict["labels"], dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)
    probabilities = embeddings_dict.get("probabilities", None)
    
    # Define error groups
    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    
    n_tp, n_tn, n_fp, n_fn = np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)
    
    print(f"\nCohort Composition:")
    print(f"  TP: {n_tp:4d} | TN: {n_tn:4d}")
    print(f"  FP: {n_fp:4d} | FN: {n_fn:4d}")
    print(f"  Total: {len(y_true)}")
    
    # =========================================================================
    # STEP 1: Compute Attention Rollout for Entire Cohort
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Computing Attention Rollout (Full Cohort)")
    print(f"{'='*70}")
    
    attention_paths = list(embeddings_dict["attention_storage_paths"])
    per_layer_attention = []
    
    for layer_idx, attn_path in enumerate(attention_paths):
        print(f"  Loading layer {layer_idx}...")
        attn_mmap = np.load(attn_path, mmap_mode="r")
        per_layer_attention.append(attn_mmap)
    
    extractor = SimpleViTEmbeddingExtractor(model=model, device=device, verbose=False)
    
    print(f"\n  Computing rollout (discard_ratio={args.rollout_discard_ratio})...")
    rollout = extractor.compute_attention_rollout_mean_pooling(
        per_layer_attention,
        discard_ratio=float(args.rollout_discard_ratio),
    )
    
    print(f" Rollout shape: {rollout.shape} (N, P)")
    
    # =========================================================================
    # STEP 2: Compute Distributional Metrics (Gini & Entropy)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Computing Distributional Metrics")
    print(f"{'='*70}")
    
    metrics = AttentionDistributionalMetrics.compute_rollout_metrics_batch(rollout)
    gini_all = metrics["gini_coefficients"]
    entropy_all = metrics["spatial_entropies"]
    
    print(f"  ✓ Gini coefficients: {gini_all.shape}")
    print(f"  ✓ Spatial entropies: {entropy_all.shape}")
    
    # Compute group statistics
    aggregator = SpatialMetricsAggregator()
    
    gini_tp = aggregator.compute_summary(gini_all[tp]) if n_tp > 0 else None
    gini_tn = aggregator.compute_summary(gini_all[tn]) if n_tn > 0 else None
    gini_fp = aggregator.compute_summary(gini_all[fp]) if n_fp > 0 else None
    gini_fn = aggregator.compute_summary(gini_all[fn]) if n_fn > 0 else None
    
    entropy_tp = aggregator.compute_summary(entropy_all[tp]) if n_tp > 0 else None
    entropy_tn = aggregator.compute_summary(entropy_all[tn]) if n_tn > 0 else None
    entropy_fp = aggregator.compute_summary(entropy_all[fp]) if n_fp > 0 else None
    entropy_fn = aggregator.compute_summary(entropy_all[fn]) if n_fn > 0 else None
    
    # Print summaries
    print(f"\n  Gini Coefficient (Focality Proxy):")
    if gini_tp: print(f"    TP: {gini_tp.mean:.4f} ± {gini_tp.std:.4f} (95% CI: [{gini_tp.ci_lower:.4f}, {gini_tp.ci_upper:.4f}])")
    if gini_fp: print(f"    FP: {gini_fp.mean:.4f} ± {gini_fp.std:.4f} (95% CI: [{gini_fp.ci_lower:.4f}, {gini_fp.ci_upper:.4f}])")
    if gini_tn: print(f"    TN: {gini_tn.mean:.4f} ± {gini_tn.std:.4f} (95% CI: [{gini_tn.ci_lower:.4f}, {gini_tn.ci_upper:.4f}])")
    if gini_fn: print(f"    FN: {gini_fn.mean:.4f} ± {gini_fn.std:.4f} (95% CI: [{gini_fn.ci_lower:.4f}, {gini_fn.ci_upper:.4f}])")
    
    print(f"\n  Spatial Entropy (Diffuseness Proxy):")
    if entropy_tp: print(f"    TP: {entropy_tp.mean:.4f} ± {entropy_tp.std:.4f} (95% CI: [{entropy_tp.ci_lower:.4f}, {entropy_tp.ci_upper:.4f}])")
    if entropy_fp: print(f"    FP: {entropy_fp.mean:.4f} ± {entropy_fp.std:.4f} (95% CI: [{entropy_fp.ci_lower:.4f}, {entropy_fp.ci_upper:.4f}])")
    if entropy_tn: print(f"    TN: {entropy_tn.mean:.4f} ± {entropy_tn.std:.4f} (95% CI: [{entropy_tn.ci_lower:.4f}, {entropy_tn.ci_upper:.4f}])")
    if entropy_fn: print(f"    FN: {entropy_fn.mean:.4f} ± {entropy_fn.std:.4f} (95% CI: [{entropy_fn.ci_lower:.4f}, {entropy_fn.ci_upper:.4f}])")
    
    # Effect sizes (TP vs FP comparison - key hypothesis test)
    if gini_tp and gini_fp:
        gini_effect = compute_effect_size_cohens_d(gini_all[tp], gini_all[fp])
        print(f"\n  Cohen's d (TP vs FP Gini): {gini_effect:.3f}")
        if abs(gini_effect) > 0.8:
            print(f"    → LARGE effect: FPs show {'more diffuse' if gini_effect > 0 else 'more focal'} attention")
    
    if entropy_tp and entropy_fp:
        entropy_effect = compute_effect_size_cohens_d(entropy_all[tp], entropy_all[fp])
        print(f"  Cohen's d (TP vs FP Entropy): {entropy_effect:.3f}")
        if abs(entropy_effect) > 0.8:
            print(f"    → LARGE effect: FPs show {'higher' if entropy_effect < 0 else 'lower'} entropy")
    
    # Statistical test (Welch's t-test)
    if gini_tp and gini_fp:
        stat_test = aggregator.compare_strata(gini_all[tp], gini_all[fp])
        print(f"\n  Welch's t-test (TP vs FP Gini): p = {stat_test['p_value']:.4f}")
        if stat_test['significant']:
            print(f"    → SIGNIFICANT difference (p < 0.05)")
    
    # =========================================================================
    # STEP 3: Save Population Statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Saving Population Statistics")
    print(f"{'='*70}")
    
    results = {
        "experiment": args.experiment,
        "split": args.split,
        "n_samples": int(len(y_true)),
        "n_tp": int(n_tp),
        "n_tn": int(n_tn),
        "n_fp": int(n_fp),
        "n_fn": int(n_fn),
        "rollout_config": {
            "discard_ratio": float(args.rollout_discard_ratio),
            "n_layers": len(attention_paths),
        },
        "gini_statistics": {
            "tp": gini_tp.to_dict() if gini_tp else None,
            "tn": gini_tn.to_dict() if gini_tn else None,
            "fp": gini_fp.to_dict() if gini_fp else None,
            "fn": gini_fn.to_dict() if gini_fn else None,
        },
        "entropy_statistics": {
            "tp": entropy_tp.to_dict() if entropy_tp else None,
            "tn": entropy_tn.to_dict() if entropy_tn else None,
            "fp": entropy_fp.to_dict() if entropy_fp else None,
            "fn": entropy_fn.to_dict() if entropy_fn else None,
        },
        "effect_sizes": {
            "gini_tp_vs_fp": float(gini_effect) if (gini_tp and gini_fp) else None,
            "entropy_tp_vs_fp": float(entropy_effect) if (entropy_tp and entropy_fp) else None,
        },
        "statistical_tests": {
            "gini_tp_vs_fp": stat_test if (gini_tp and gini_fp) else None,
        }
    }
    
    stats_path = paths.analysis_root / f"rollout_stats_{args.split}.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved: {stats_path}")
    
    # =========================================================================
    # STEP 4: Generate Distributional Boxplots (Requirement 2.1)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Generating Distributional Boxplots")
    print(f"{'='*70}")
    
    # Gini boxplot
    gini_dict = {}
    if n_tp > 0: gini_dict["TP"] = gini_all[tp]
    if n_tn > 0: gini_dict["TN"] = gini_all[tn]
    if n_fp > 0: gini_dict["FP"] = gini_all[fp]
    if n_fn > 0: gini_dict["FN"] = gini_all[fn]
    
    if len(gini_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=gini_dict,
            metric_name="Gini Coefficient (Focality)",
            save_path=str(paths.analysis_root / f"rollout_gini_distribution_{args.split}.png"),
            ylabel="Gini Coefficient",
            title=f"Attention Focality Distribution | {args.split.upper()}",
        )
        print(f"  ✓ Saved: rollout_gini_distribution_{args.split}.png")
    
    # Entropy boxplot
    entropy_dict = {}
    if n_tp > 0: entropy_dict["TP"] = entropy_all[tp]
    if n_tn > 0: entropy_dict["TN"] = entropy_all[tn]
    if n_fp > 0: entropy_dict["FP"] = entropy_all[fp]
    if n_fn > 0: entropy_dict["FN"] = entropy_all[fn]
    
    if len(entropy_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=entropy_dict,
            metric_name="Spatial Entropy (Diffuseness)",
            save_path=str(paths.analysis_root / f"rollout_entropy_distribution_{args.split}.png"),
            ylabel="Shannon Entropy (nats)",
            title=f"Attention Diffuseness Distribution | {args.split.upper()}",
        )
        print(f"  ✓ Saved: rollout_entropy_distribution_{args.split}.png")
    
    # =========================================================================
    # STEP 5: Select Stratified Representatives (Requirement 3.3)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Selecting Stratified Representatives")
    print(f"{'='*70}")
    
    image_size = int(config["image_size"])
    patch_size = int(config["patch_size"])
    num_patches_side = image_size // patch_size
    
    representatives = {}
    
    # For each group, select top 3 closest to median Gini
    sampler = StratifiedAttentionSampler(random_seed=42)
    
    for group_name, group_mask, n_samples in [
        ("TP", tp, n_tp),
        ("TN", tn, n_tn),
        ("FP", fp, n_fp),
        ("FN", fn, n_fn),
    ]:
        if n_samples < 1:
            continue
        
        print(f"\n  {group_name} (n={n_samples}):")
        
        # Create single stratum for this group
        stratum = StrataDefinition(
            stratum_id=group_name,
            mask=group_mask,
            n_samples=int(n_samples),
            target_samples=min(3, int(n_samples)),  # Top 3 or all if fewer
        )
        
        # Select representatives closest to median Gini
        sample = sampler.select_representative_samples(
            strata=[stratum],
            metric_values=gini_all,
            n_per_stratum=min(3, int(n_samples)),
            selection_criterion="median",
        )
        
        representatives[group_name] = sample.sample_indices
        
        for rank, idx in enumerate(sample.sample_indices):
            print(f"    Rep {rank+1}: sample {idx:4d} | Gini={gini_all[idx]:.4f} | Entropy={entropy_all[idx]:.4f}")
    
    # =========================================================================
    # STEP 6: Visualize Representatives (Rollout + IG)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Visualizing Representatives (Rollout + Integrated Gradients)")
    print(f"{'='*70}")
    
    viz = SpatialAttentionVisualizer(output_dir=str(paths.analysis_root), dpi=300)
    
    for group_name, indices in representatives.items():
        for rank, idx in enumerate(indices):
            # Plot rollout heatmap
            save_name = f"rep_{group_name}_{idx:04d}_rollout_{args.split}.png"
            viz.plot_rollout_relevance_heatmap(
                rollout_relevance=rollout[idx],
                num_patches_side=num_patches_side,
                title=f"Attention Rollout | {group_name} Representative {rank+1} (sample {idx})",
                save_name=save_name,
                show=False,
            )
            print(f"    ✓ {save_name}")
    
    # =========================================================================
    # STEP 7: Compute Integrated Gradients for Representatives
    # =========================================================================
    if args.compute_integrated_gradients:
        
        ig_attribution_full = extractor.compute_integrated_gradients(
            dataloader=dataloader,
            target_class=args.attribution_target_class,
            steps=int(args.ig_steps),
            aggregate_method=str(args.attribution_aggregate_method),
            baseline="zeros",
        )

        ig_results = run_ig_population_analysis(
            ig_attribution=ig_attribution_full,
            embeddings_dict=embeddings_dict,
            paths=paths,
            args=args,
        )
        
        all_rep_indices = np.concatenate([representatives[g] for g in representatives.keys()])
        ig_attribution_reps = ig_attribution_full[all_rep_indices]

        ig_idx = 0
        for group_name, indices in representatives.items():
            for rank, orig_idx in enumerate(indices):
                save_name = f"rep_{group_name}_{orig_idx:04d}_ig_{args.split}.png"
                viz.plot_decision_attribution_heatmap(
                    attribution=ig_attribution_reps[ig_idx],
                    num_patches_side=num_patches_side,
                    title=f"Integrated Gradients | {group_name} Rep {rank+1} (sample {orig_idx})",
                    save_name=save_name,
                    show=False,
                )
                print(f" {save_name}")
                ig_idx += 1
        
        image_size = int(config["image_size"])
        patch_size = int(config["patch_size"])
        num_patches_side = image_size // patch_size

        panel_attributions = {}
        panel_indices = {}

        for groupname in ["TP", "TN", "FP", "FN"]:
            if groupname not in representatives:
                continue
            if len(representatives[groupname]) < 1:
                continue
            exemplar_idx = int(representatives[groupname][0])
            panel_attributions[groupname] = ig_attribution_full[exemplar_idx]
            panel_indices[groupname] = exemplar_idx

        if len(panel_attributions) >= 2:
            viz.plot_ig_comparison_panel(
                attributions_dict=panel_attributions,
                num_patches_side=num_patches_side,
                sample_indices=panel_indices,
                title=f"IG Comparison Shared Scale (Rollout reps) {args.split.upper()}",
                save_name=f"ig_comparison_shared_scale_{args.split}.png",
                show=False
            )
        
        return results

def run_ig_population_analysis(
    ig_attribution: np.ndarray,
    embeddings_dict: Dict[str, Any],
    paths: ProjectPaths,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Integrated Gradients population analysis
    
    Args:
        ig_attribution: (N, P) attribution maps
        embeddings_dict: Embedding extraction results
        slide_ids: (N,) slide identifiers
        paths: Project paths
        args: CLI arguments
        config: Configuration
        
    Returns:
        Dictionary with results
    """
    
    # Extract predictions and labels
    y_true = np.asarray(embeddings_dict["labels"], dtype=int)
    y_pred = np.asarray(embeddings_dict["predictions"], dtype=int)
    
    # Define error groups
    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    
    n_tp, n_tn, n_fp, n_fn = np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)
    
    print(f"\nCohort Composition:")
    print(f"  TP: {n_tp:4d} | TN: {n_tn:4d}")
    print(f"  FP: {n_fp:4d} | FN: {n_fn:4d}")
    
    # =========================================================================
    # STEP 1: Compute Patch-Level Metrics
    # =========================================================================
    
    metrics = AttributionMagnitudeMetrics.compute_attribution_metrics_batch(
        ig_attribution
    )
    
    l1_all = metrics["l1_magnitudes"]
    l2_all = metrics["l2_magnitudes"]
    max_all = metrics["max_magnitudes"]
    gini_all = metrics["gini_coefficients"]
    entropy_all = metrics["spatial_entropies"]
    
    # =========================================================================
    # STEP 2: Patch-Level Group Statistics
    # =========================================================================
    
    aggregator = SpatialMetricsAggregator()

    # L1
    l1_tp = aggregator.compute_summary(l1_all[tp]) if n_tp > 0 else None
    l1_tn = aggregator.compute_summary(l1_all[tn]) if n_tn > 0 else None
    l1_fp = aggregator.compute_summary(l1_all[fp]) if n_fp > 0 else None
    l1_fn = aggregator.compute_summary(l1_all[fn]) if n_fn > 0 else None

    # L2
    l2_tp = aggregator.compute_summary(l2_all[tp]) if n_tp > 0 else None
    l2_tn = aggregator.compute_summary(l2_all[tn]) if n_tn > 0 else None
    l2_fp = aggregator.compute_summary(l2_all[fp]) if n_fp > 0 else None
    l2_fn = aggregator.compute_summary(l2_all[fn]) if n_fn > 0 else None

    # Max
    max_tp = aggregator.compute_summary(max_all[tp]) if n_tp > 0 else None
    max_tn = aggregator.compute_summary(max_all[tn]) if n_tn > 0 else None
    max_fp = aggregator.compute_summary(max_all[fp]) if n_fp > 0 else None
    max_fn = aggregator.compute_summary(max_all[fn]) if n_fn > 0 else None

    # IG-Gini
    gini_tp = aggregator.compute_summary(gini_all[tp]) if n_tp > 0 else None
    gini_tn = aggregator.compute_summary(gini_all[tn]) if n_tn > 0 else None
    gini_fp = aggregator.compute_summary(gini_all[fp]) if n_fp > 0 else None
    gini_fn = aggregator.compute_summary(gini_all[fn]) if n_fn > 0 else None

    # IG-Entropy
    entropy_tp = aggregator.compute_summary(entropy_all[tp]) if n_tp > 0 else None
    entropy_tn = aggregator.compute_summary(entropy_all[tn]) if n_tn > 0 else None
    entropy_fp = aggregator.compute_summary(entropy_all[fp]) if n_fp > 0 else None
    entropy_fn = aggregator.compute_summary(entropy_all[fn]) if n_fn > 0 else None


    # =========================================================================
    # STEP 3: Patch-Level Comparison
    # =========================================================================

    comparisons = {}
    effect_sizes = {}

    # Comparison A: FP vs TN on IG L1 magnitude
    if n_fp >= 2 and n_tn >= 2:
        comparisons["fp_vs_tn_l1"] = compare_groups_patch_level(l1_all[fp], l1_all[tn])
        effect_sizes["fp_vs_tn_l1_cohens_d"] = comparisons["fp_vs_tn_l1"]["cohens_d"]

    # Comparison B: FN vs TP on IG L1 magnitude
    if n_fn >= 2 and n_tp >= 2:
        comparisons["fn_vs_tp_l1"] = compare_groups_patch_level(l1_all[fn], l1_all[tp])
        effect_sizes["fn_vs_tp_l1_cohens_d"] = comparisons["fn_vs_tp_l1"]["cohens_d"]


    # =========================================================================
    # STEP 4: Distributional Boxplots
    # =========================================================================

    # L1 magnitude boxplot
    l1_dict = {}
    if n_tp > 0: l1_dict["TP"] = l1_all[tp]
    if n_tn > 0: l1_dict["TN"] = l1_all[tn]
    if n_fp > 0: l1_dict["FP"] = l1_all[fp]
    if n_fn > 0: l1_dict["FN"] = l1_all[fn]

    if len(l1_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=l1_dict,
            metric_name="IG L1 Magnitude",
            save_path=str(paths.analysis_root / f"ig_l1_distribution_{args.split}.png"),
            ylabel="L1 Magnitude",
            title=f"IG Evidence Magnitude Distribution | {args.split.upper()}",
        )

    # Gini concentration boxplot
    gini_dict = {}
    if n_tp > 0: gini_dict["TP"] = gini_all[tp]
    if n_tn > 0: gini_dict["TN"] = gini_all[tn]
    if n_fp > 0: gini_dict["FP"] = gini_all[fp]
    if n_fn > 0: gini_dict["FN"] = gini_all[fn]

    if len(gini_dict) >= 2:
        plot_metric_distributions_by_group(
            metric_dict=gini_dict,
            metric_name="IG Gini Coefficient",
            save_path=str(paths.analysis_root / f"ig_gini_distribution_{args.split}.png"),
            ylabel="Gini Coefficient",
            title=f"IG Evidence Focality Distribution | {args.split.upper()}",
        )

    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    results = {
        "experiment": args.experiment,
        "split": args.split,
        "n_samples": int(len(y_true)),
        "n_tp": int(n_tp),
        "n_tn": int(n_tn),
        "n_fp": int(n_fp),
        "n_fn": int(n_fn),
        "ig_config": {
            "target_class": args.attribution_target_class,
            "steps": args.ig_steps,
            "aggregate_method": args.attribution_aggregate_method,
        },
        "patch_level_statistics": {
            "magnitude": {
                "l1": {
                    "tp": l1_tp.to_dict() if l1_tp else None,
                    "tn": l1_tn.to_dict() if l1_tn else None,
                    "fp": l1_fp.to_dict() if l1_fp else None,
                    "fn": l1_fn.to_dict() if l1_fn else None,
                },
                "l2": {
                    "tp": l2_tp.to_dict() if l2_tp else None,
                    "tn": l2_tn.to_dict() if l2_tn else None,
                    "fp": l2_fp.to_dict() if l2_fp else None,
                    "fn": l2_fn.to_dict() if l2_fn else None,
                },
                "max": {
                    "tp": max_tp.to_dict() if max_tp else None,
                    "tn": max_tn.to_dict() if max_tn else None,
                    "fp": max_fp.to_dict() if max_fp else None,
                    "fn": max_fn.to_dict() if max_fn else None,
                },
            },
            "concentration": {
                "gini": {
                    "tp": gini_tp.to_dict() if gini_tp else None,
                    "tn": gini_tn.to_dict() if gini_tn else None,
                    "fp": gini_fp.to_dict() if gini_fp else None,
                    "fn": gini_fn.to_dict() if gini_fn else None,
                },
                "entropy": {
                    "tp": entropy_tp.to_dict() if entropy_tp else None,
                    "tn": entropy_tn.to_dict() if entropy_tn else None,
                    "fp": entropy_fp.to_dict() if entropy_fp else None,
                    "fn": entropy_fn.to_dict() if entropy_fn else None,
                },
            },
        },

        "comparisons": comparisons,
        "effect_sizes": effect_sizes,
    }

    stats_path = paths.analysis_root / f"ig_stats_{args.split}.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Saved: {stats_path}")

    print(f"\n{'='*70}")
    print("IG Population Analysis Complete")
    print(f"{'='*70}\n")

    return results

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embedding + attention analysis")

    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--preprocessed-root", type=str, default="/content/data/preprocessed")
    p.add_argument("--dataset-subdir", type=str, default="fungi")
    p.add_argument("--output-root", type=str, default="/content/drive/MyDrive/colab/analysis/output")
    p.add_argument("--split-file", type=str, 
                default="/content/drive/MyDrive/colab/analysis/preprocessed/split_indices.json")

    p.add_argument("--experiment", type=str, default="standard")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force-recompute", action="store_true")

    p.add_argument("--extract-attention", action="store_true")
    p.add_argument("--attention-layers", type=int, nargs="+", default=None)

    p.add_argument("--run-embedding-analysis", action="store_true")

    p.add_argument("--compute-attention-rollout", action="store_true")
    p.add_argument("--rollout-discard-ratio", type=float, default=0.9)

    p.add_argument("--compute-integrated-gradients", action="store_true")
    p.add_argument("--attribution-target-class", type=int, default=None)
    p.add_argument("--attribution-aggregate-method", type=str, default="l2norm")
    p.add_argument("--ig-steps", type=int, default=50)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = ProjectPaths.from_args(args)

    cli_config = get_config(args.experiment)
    device = get_device_info()

    dataset = NPYImageFolder(str(paths.preprocessed_dir))
    class_names = getattr(dataset, "classes", ["class0", "class1"])
    print(f"Dataset: {len(dataset)} samples | classes={class_names}")

    indices = load_split_with_conversion(args.split_file, dataset, args.split)

    model_path = paths.output_root / args.experiment / f"final_model_{args.experiment}.pth"
    model, checkpoint, ckpt_config = load_model_and_checkpoint(str(model_path), cli_config, device)

    config = ckpt_config

    loader, dataset_indices = create_dataloader_with_provenance(
        dataset=dataset,
        indices=indices,
        checkpoint=checkpoint,
        config=config,
        batch_size=args.batch_size,
    )

    embeddings_path = paths.analysis_root / f"embeddings_{args.split}_{args.experiment}.npz"

    if embeddings_path.exists() and not args.force_recompute:
        embeddings_dict = SimpleViTEmbeddingExtractor.load_embeddings(str(embeddings_path), verbose=True)
    else:
        extractor = SimpleViTEmbeddingExtractor(model=model, device=device, verbose=True)
        embeddings_dict = extractor.extract_embeddings(
            dataloader=loader,
            dataset_indices=dataset_indices,
            extract_patch_tokens=False,
            extract_transformer_tokens=False,
            extract_attention=bool(args.extract_attention),
            attention_layers=args.attention_layers,
            save_spatial_attention=True,
            enforce_forward_equivalence=True,
        )
        extractor.save_embeddings(embeddings_dict, str(embeddings_path), compress=False)

    y_true = np.asarray(embeddings_dict.get("labels"), dtype=int)
    y_pred = np.asarray(embeddings_dict.get("predictions"), dtype=int)
    embeddings_dict["labels"] = y_true
    embeddings_dict["predictions"] = y_pred

    if "mean" not in embeddings_dict:
        embeddings_dict["mean"] = checkpoint.get("mean")
    if "std" not in embeddings_dict:
        embeddings_dict["std"] = checkpoint.get("std")

    slide_ids, provenance_map = reconstruct_slide_id_provenance(
    dataset, embeddings_dict["indices"], validate=True
    )
    embeddings_dict["slide_ids"] = slide_ids

    # ========== EMBEDDING ANALYSIS ==========
    if args.run_embedding_analysis:
        analyzer = EmbeddingAnalyzer(output_dir=str(paths.analysis_root), dpi=300, random_seed=42)
        pooled = np.asarray(embeddings_dict["pooled_embeddings"], dtype=np.float32)
      
        slide_viz = SlideConditionedVisualizer(output_dir=str(paths.analysis_root), dpi=300)

        umap_proj = analyzer.compute_umap_projection(
            pooled, 
            n_neighbors=15, 
            min_dist=0.1
        )
        umap_metrics = analyzer.analyze_embedding_quality(
            embeddings=pooled,
            projection=umap_proj,
            labels=y_true,
            predictions=y_pred,
            slide_ids=slide_ids,
            method_name="UMAP",
        )
        umap_metrics_path = paths.analysis_root / f"umap_quality_metrics_{args.split}.json"
        with open(umap_metrics_path, "w") as f:
            json.dump(umap_metrics, f, indent=2)
        print(f"Saved UMAP metrics: {umap_metrics_path}")

        analyzer.plot_embedding_projection(
            projection=umap_proj,
            labels=y_true,
            predictions=y_pred,
            slide_ids=slide_ids,
            class_names=class_names,
            method_name="UMAP",
            save_name=f"umap_comprehensive_{args.split}.png",
            plot_errors_only=False,
            show=False,
        )
        slide_viz.plot_slide_conditioned_errors(
            projection=umap_proj,
            slide_ids=slide_ids,
            labels=y_true,
            predictions=y_pred,
            class_names=class_names,
            method_name="UMAP",
            save_name=f"umap_slide_conditioned_{args.split}.png",
            show=False,
        )

    # ========== ROLLOUT POPULATION ANALYSIS ==========
    if args.compute_attention_rollout:
        run_rollout_population_analysis(
            embeddings_dict=embeddings_dict,
            slide_ids=slide_ids,
            model=model,
            dataloader=loader,
            paths=paths,
            args=args,
            config=config,
            device=device,
        )


if __name__ == "__main__":
    main()
