"""
extractor.py
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm


# -----------------------------
# Low-level attention capture
# -----------------------------
class SpatialAttentionCapture:
    """
    Forward hook capturing attention weights with full spatial resolution.

    Stores:
    - Full attention tensors: [B, heads, P, P]
    - Token importance (attention received): [B, P] = column mass of head-avg attention
    - Head-averaged attention: [B, P, P]
    """

    def __init__(
        self,
        attention_module: nn.Module,
        layer_idx: int,
        output_dir: str,
        batch_counter: List[int],
        save_spatial: bool = True,
    ):
        self.attention_module = attention_module
        self.layer_idx = layer_idx
        self.output_dir = output_dir
        self.batch_counter = batch_counter
        self.save_spatial = save_spatial

        self.layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        os.makedirs(self.layer_dir, exist_ok=True)

        # Required attributes for lucidrains/vit-pytorch SimpleViT Attention module
        required = ["heads", "scale", "attend", "to_qkv", "norm"]
        missing = [k for k in required if not hasattr(attention_module, k)]
        if missing:
            raise AttributeError(
                f"Attention module at layer {layer_idx} missing required attributes: {missing}. "
                "This extractor assumes the lucidrains/vit-pytorch SimpleViT attention API."
            )

        self.heads = attention_module.heads
        self.scale = attention_module.scale
        self.attend = attention_module.attend

    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        x = input[0].detach()

        with torch.no_grad():
            x_norm = module.norm(x)
            qkv = module.to_qkv(x_norm).chunk(3, dim=-1)

            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
                qkv,
            )

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)  # [B, heads, P, P]

            batch_idx = int(self.batch_counter[0])

            if self.save_spatial:
                save_path = os.path.join(self.layer_dir, f"batch_{batch_idx:04d}.npy")
                np.save(save_path, attn.detach().cpu().numpy().astype(np.float32))

            attn_head_avg = attn.mean(dim=1)  # [B, P, P]
            token_importance = attn_head_avg.sum(dim=1)  # [B, P] (sum over queries -> attention received)

            stats_path = os.path.join(self.layer_dir, f"batch_{batch_idx:04d}_stats.npz")
            np.savez_compressed(
                stats_path,
                token_importance=token_importance.cpu().numpy().astype(np.float32),
                attn_head_avg=attn_head_avg.cpu().numpy().astype(np.float32),
            )

            del attn, dots, q, k, v, x_norm, qkv, attn_head_avg, token_importance
            torch.cuda.empty_cache()

    def load_all_batches(self) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        batch_files = sorted([f for f in os.listdir(self.layer_dir) if f.endswith(".npy")])
        if not batch_files:
            warnings.warn(f"No batch files in {self.layer_dir}")
            return None, {}

        first_batch = np.load(os.path.join(self.layer_dir, batch_files[0]), mmap_mode="r")
        _, num_heads, num_patches, _ = first_batch.shape

        total_samples = 0
        for f in batch_files:
            total_samples += int(np.load(os.path.join(self.layer_dir, f), mmap_mode="r").shape[0])

        full_attn_path = os.path.join(self.layer_dir, "layer_attention.npy")
        attn_mmap = np.lib.format.open_memmap(
            full_attn_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_samples, num_heads, num_patches, num_patches),
        )
        token_importance_full = np.zeros((total_samples, num_patches), dtype=np.float32)

        offset = 0
        for batch_file in tqdm(batch_files, desc=f"Consolidating Layer {self.layer_idx}"):
            batch_path = os.path.join(self.layer_dir, batch_file)
            batch_data = np.load(batch_path)
            batch_size = batch_data.shape[0]

            attn_mmap[offset : offset + batch_size] = batch_data

            stats_file = batch_file.replace(".npy", "_stats.npz")
            stats_path = os.path.join(self.layer_dir, stats_file)

            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                token_importance_full[offset : offset + batch_size] = stats["token_importance"]

            offset += batch_size

            os.remove(batch_path)
            if os.path.exists(stats_path):
                os.remove(stats_path)
            
            del batch_data
            gc.collect()

        stats_path = os.path.join(self.layer_dir, "layer_statistics.npz")
        np.savez_compressed(
            stats_path,
            token_importance=token_importance_full,
        )

        del attn_mmap, token_importance_full
        gc.collect()

        attention_full = np.load(full_attn_path, mmap_mode="r")
        stats_npz = np.load(stats_path)
        
        statistics = {
            "token_importance": stats_npz["token_importance"],
        }
        return attention_full, statistics


# -----------------------------
# Model forward alignment
# -----------------------------
@dataclass(frozen=True)
class ForwardDecomposition:
    patch_tokens_pre_pos: torch.Tensor  # [B, P, D]
    patch_tokens_post_pos: torch.Tensor  # [B, P, D]
    transformer_tokens: torch.Tensor  # [B, P, D]
    pooled: torch.Tensor  # [B, D]
    logits: torch.Tensor  # [B, C]


class SimpleViTEmbeddingExtractor:
    """
    Extractor for the lucidrains/vit-pytorch SimpleViT (mean-pooled, sincos pos emb).

    Outputs are *definitions* (not ambiguous names):
    - patch_tokens_pre_pos: tokens from to_patch_embedding (before pos add)
    - transformer_tokens: output tokens after transformer encoder
    - pooled_embeddings: mean(token) pooled vector (input to linear_head)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        verbose: bool = True,
        temp_dir: Optional[str] = None,
        forward_check_atol: float = 1e-5,
        forward_check_rtol: float = 1e-5,
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.forward_check_atol = float(forward_check_atol)
        self.forward_check_rtol = float(forward_check_rtol)

        self.model.eval()

        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="vit_attention_spatial_")
        else:
            self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        if self.verbose:
            print(f"Spatial attention storage: {self.temp_dir}")

        self._attention_captures: List[SpatialAttentionCapture] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._batch_counter = [0]

        self._verify_architecture()

    # ---- architecture / API checks ----
    def _verify_architecture(self) -> None:
        required_top = ["to_patch_embedding", "transformer", "linear_head", "pos_embedding"]
        missing = [k for k in required_top if not hasattr(self.model, k)]
        if missing:
            raise AttributeError(
                f"Model is missing required attributes: {missing}. "
                "This extractor assumes lucidrains/vit-pytorch SimpleViT-like API."
            )

        if not hasattr(self.model.transformer, "layers"):
            raise AttributeError("model.transformer.layers not found; cannot register attention hooks.")

        # Validate that each layer has Attention at index 0 (as in lucidrains Transformer)
        for li, layer in enumerate(self.model.transformer.layers):
            if not isinstance(layer, (list, tuple, nn.ModuleList)) or len(layer) < 1:
                raise AttributeError(f"Unexpected transformer.layers[{li}] structure: {type(layer)}")
            attn = layer[0]
            required_attn = ["to_qkv", "attend", "heads", "scale", "norm"]
            if any(not hasattr(attn, k) for k in required_attn):
                raise AttributeError(
                    f"Transformer layer {li} attention module lacks required API fields. "
                    f"Missing: {[k for k in required_attn if not hasattr(attn, k)]}"
                )

    # ---- forward decomposition ----
    def _decompose_forward(self, X: torch.Tensor) -> ForwardDecomposition:
        """
        Recompute the model forward using the explicit components, matching the SimpleViT definition:
            patch = to_patch_embedding(X)
            patch += pos_embedding
            tokens = transformer(patch)
            pooled = mean(tokens)
            logits = linear_head(pooled)
        """
        patch_pre = self.model.to_patch_embedding(X)  # [B, P, D]
        pos = self.model.pos_embedding.to(device=X.device, dtype=patch_pre.dtype)
        patch_post = patch_pre + pos
        tokens = self.model.transformer(patch_post)
        pooled = tokens.mean(dim=1)
        logits = self.model.linear_head(pooled)
        return ForwardDecomposition(
            patch_tokens_pre_pos=patch_pre,
            patch_tokens_post_pos=patch_post,
            transformer_tokens=tokens,
            pooled=pooled,
            logits=logits,
        )

    def _assert_forward_equivalence(self, logits_direct: torch.Tensor, logits_decomposed: torch.Tensor) -> None:
        if logits_direct.shape != logits_decomposed.shape:
            raise RuntimeError(
                f"Logit shape mismatch: direct={tuple(logits_direct.shape)} vs decomposed={tuple(logits_decomposed.shape)}"
            )

        ok = torch.allclose(
            logits_direct,
            logits_decomposed,
            atol=self.forward_check_atol,
            rtol=self.forward_check_rtol,
        )
        if not ok:
            max_abs = (logits_direct - logits_decomposed).abs().max().item()
            raise RuntimeError(
                "Forward-pass alignment check FAILED: model(X) logits differ from decomposed logits. "
                f"max|Δ|={max_abs:.6e}. This invalidates embedding/attribution interpretability."
            )

    # ---- attention hooks ----
    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_captures = []

    def _register_attention_hooks(self, layer_indices: Optional[List[int]], save_spatial: bool) -> None:
        self._remove_hooks()

        if layer_indices is None:
            layer_indices = list(range(len(self.model.transformer.layers)))

        for li in layer_indices:
            if li < 0 or li >= len(self.model.transformer.layers):
                warnings.warn(f"Layer {li} out of range; skipping.")
                continue
            attn_module = self.model.transformer.layers[li][0]
            capture = SpatialAttentionCapture(
                attention_module=attn_module,
                layer_idx=li,
                output_dir=self.temp_dir,
                batch_counter=self._batch_counter,
                save_spatial=save_spatial,
            )
            self._attention_captures.append(capture)
            self._hooks.append(attn_module.register_forward_hook(capture))

        if self.verbose and layer_indices:
            print(f"Registered attention hooks for layers: {layer_indices}")

    # ---- main extraction ----
    def extract_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_indices: np.ndarray,
        extract_patch_tokens: bool = False,
        extract_transformer_tokens: bool = False,
        extract_attention: bool = False,
        attention_layers: Optional[List[int]] = None,
        max_batches: Optional[int] = None,
        save_spatial_attention: bool = True,
        enforce_forward_equivalence: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary with:
        - pooled_embeddings: (N, D) [float32]
        - predictions: (N,) [int]
        - probabilities: (N, C) [float32]
        - labels: (N,) [int]
        - indices: (N,) dataset indices
        - patch_tokens_pre_pos: (N, P, D) [float32] if requested
        - transformer_tokens: (N, P, D) [float32] if requested
        - attention_storage_paths: list[str] if attention extracted
        - token_importance: list[np.ndarray] if attention extracted (loaded stats arrays)
        - attn_head_avg: list[np.ndarray] if attention extracted (loaded stats arrays)
        """
        expected_samples = len(dataloader.dataset)
        if len(dataset_indices) != expected_samples:
            raise ValueError(
                f"dataset_indices length ({len(dataset_indices)}) != DataLoader dataset size ({expected_samples})"
            )

        if extract_attention:
            self._register_attention_hooks(attention_layers, save_spatial_attention)
        else:
            self._remove_hooks()

        pooled_list: List[np.ndarray] = []
        patch_list: List[np.ndarray] = []
        tokens_list: List[np.ndarray] = []

        preds_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        iterator = tqdm(dataloader, total=total, desc="Extracting") if self.verbose else dataloader

        n_seen = 0
        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                self._batch_counter[0] = int(batch_idx)
                X = X.to(self.device)

                # Direct forward (the actual deployed path)
                logits_direct = self.model(X)
                probs = torch.softmax(logits_direct, dim=1)
                preds = torch.argmax(logits_direct, dim=1)

                # Decomposed forward (for aligned embeddings + internal features)
                decomp = self._decompose_forward(X)

                if enforce_forward_equivalence:
                    self._assert_forward_equivalence(logits_direct, decomp.logits)

                pooled_list.append(decomp.pooled.detach().cpu().numpy().astype(np.float32))

                if extract_patch_tokens:
                    patch_list.append(decomp.patch_tokens_pre_pos.detach().cpu().numpy().astype(np.float32))

                if extract_transformer_tokens:
                    tokens_list.append(decomp.transformer_tokens.detach().cpu().numpy().astype(np.float32))

                preds_list.append(preds.detach().cpu().numpy().astype(np.int64))
                probs_list.append(probs.detach().cpu().numpy().astype(np.float32))
                labels_list.append(y.detach().cpu().numpy().astype(np.int64))

                n_seen += int(X.size(0))
                del X, logits_direct, probs, preds, decomp
                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    gc.collect()

        if n_seen != expected_samples:
            warnings.warn(f"Processed {n_seen} samples, expected {expected_samples} (check max_batches).")

        results: Dict[str, Any] = {
            "pooled_embeddings": np.concatenate(pooled_list, axis=0),
            "predictions": np.concatenate(preds_list, axis=0),
            "probabilities": np.concatenate(probs_list, axis=0),
            "labels": np.concatenate(labels_list, axis=0),
            "indices": dataset_indices.copy(),
        }

        if extract_patch_tokens:
            results["patch_tokens_pre_pos"] = np.concatenate(patch_list, axis=0)

        if extract_transformer_tokens:
            results["transformer_tokens"] = np.concatenate(tokens_list, axis=0)

        if extract_attention and self._attention_captures:
            if self.verbose:
                print("\nConsolidating attention tensors/statistics...")

            attention_paths: List[str] = []
            token_importance_per_layer: List[np.ndarray] = []
            attn_head_avg_per_layer: List[np.ndarray] = []

            for capture in self._attention_captures:
                layer_attn, layer_stats = capture.load_all_batches()
                if layer_attn is None:
                    continue
                attention_paths.append(os.path.join(capture.layer_dir, "layer_attention.npy"))
                token_importance_per_layer.append(layer_stats["token_importance"])
                
                del layer_attn
                gc.collect()

            if attention_paths:
                results["attention_storage_paths"] = attention_paths
                results["token_importance"] = token_importance_per_layer
                #results["attn_head_avg"] = attn_head_avg_per_layer

        self._remove_hooks()
        return results

    # ---- attention rollout (mean pooling) ----
    def compute_attention_rollout_mean_pooling(
        self,
        attention_weights: List[np.ndarray],
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Attention-flow rollout for mean-pooled architectures (Abnar & Zuidema style).

        attention_weights: list of arrays, each with shape (N, heads, P, P)
        Returns: (N, P) global relevance (sum over query positions).
        """
        if not attention_weights:
            raise ValueError("No attention weights provided.")

        n_samples = attention_weights[0].shape[0]
        n_patches = attention_weights[0].shape[2]

        rollout = np.eye(n_patches, dtype=np.float32)[None, :, :].repeat(n_samples, axis=0)

        for layer_attn in attention_weights:
            attn_head_avg = layer_attn.mean(axis=1).astype(np.float32)  # (N, P, P)
            attn_adjusted = 0.5 * attn_head_avg + 0.5 * np.eye(n_patches, dtype=np.float32)

            if discard_ratio > 0:
                thr = np.percentile(attn_adjusted, discard_ratio * 100.0, axis=-1, keepdims=True)
                attn_adjusted = np.where(attn_adjusted < thr, 0.0, attn_adjusted)

            attn_adjusted = attn_adjusted / (attn_adjusted.sum(axis=-1, keepdims=True) + 1e-8)
            rollout = np.matmul(attn_adjusted, rollout)

        global_relevance = rollout.sum(axis=1)  # sum over queries
        return global_relevance.astype(np.float32)
    
    # ---- decision-linked attribution (integrated gradients) ----
    def compute_integrated_gradients(
        self,
        dataloader: torch.utils.data.DataLoader,
        target_class: Optional[int] = None,
        steps: int = 50,
        aggregate_method: str = "l2norm",
        baseline: str = "zeros",
        max_batches: Optional[int] = None,
    ) -> np.ndarray:
        """
        Integrated gradients on pixel space, propagated to patch tokens.

        baseline:
        - "zeros": baseline image = 0 (after normalization, corresponds to mean intensity if normalized appropriately)
        - "noise": small gaussian noise around 0 (stochastic baseline; less standard)
        """
        if steps < 2:
            raise ValueError("steps must be >= 2")
        if aggregate_method not in {"l2norm", "meanabs", "sumabs"}:
            raise ValueError("aggregate_method must be one of: l2norm, meanabs, sumabs")
        if baseline not in {"zeros", "noise"}:
            raise ValueError("baseline must be one of: zeros, noise")

        self.model.eval()
        self.model.requires_grad_(True)

        all_attr: List[np.ndarray] = []
        total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
        iterator = tqdm(dataloader, total=total, desc="Integrated gradients") if self.verbose else dataloader

        with torch.enable_grad():
            for bi, (X, _) in enumerate(iterator):
                if max_batches is not None and bi >= max_batches:
                    break

                X = X.to(self.device)

                if baseline == "zeros":
                    X0 = torch.zeros_like(X)
                else:
                    X0 = torch.randn_like(X) * 0.01

                # Determine target labels (predicted class on the *actual* input)
                with torch.no_grad():
                    logits = self.model(X)
                    preds = logits.argmax(dim=1)

                ig_accum: Optional[torch.Tensor] = None
                for s in range(steps):
                    alpha = float(s) / float(steps - 1)
                    Xs = X0 + alpha * (X - X0)
                    Xs.requires_grad_(True)

                    patch_pre = self.model.to_patch_embedding(Xs)
                    pos = self.model.pos_embedding.to(device=Xs.device, dtype=patch_pre.dtype)
                    patch = patch_pre + pos
                    patch.retain_grad()

                    tokens = self.model.transformer(patch)
                    pooled = tokens.mean(dim=1)
                    logits_s = self.model.linear_head(pooled)

                    if target_class is None:
                        selected = logits_s.gather(1, preds[:, None]).sum()
                    else:
                        selected = logits_s[:, int(target_class)].sum()

                    self.model.zero_grad(set_to_none=True)
                    if patch.grad is not None:
                        patch.grad.zero_()

                    selected.backward()
                    grads = patch.grad  # (B, P, D)

                    if grads is None:
                        raise RuntimeError("IG: gradient is None; check autograd graph.")

                    if ig_accum is None:
                        ig_accum = grads.detach().clone()
                    else:
                        ig_accum = ig_accum + grads.detach()

                    del Xs, patch_pre, patch, tokens, pooled, logits_s, selected, grads
                    torch.cuda.empty_cache()

                # Average gradient along the path
                ig_avg = ig_accum / float(steps)

                # Path difference in patch-token space (baseline-to-input)
                with torch.no_grad():
                    patch_in = self.model.to_patch_embedding(X) + self.model.pos_embedding.to(device=X.device, dtype=torch.float32)
                    patch_base = self.model.to_patch_embedding(X0) + self.model.pos_embedding.to(device=X.device, dtype=torch.float32)
                    delta = (patch_in - patch_base).to(device=ig_avg.device, dtype=ig_avg.dtype)

                ig = ig_avg * delta  # (B, P, D)

                if aggregate_method == "l2norm":
                    attr = ig.norm(dim=-1)
                elif aggregate_method == "meanabs":
                    attr = ig.abs().mean(dim=-1)
                else:
                    attr = ig.abs().sum(dim=-1)

                all_attr.append(attr.detach().cpu().numpy().astype(np.float32))

                del X0, logits, preds, ig_accum, ig_avg, patch_in, patch_base, delta, ig, attr
                torch.cuda.empty_cache()
                if bi % 5 == 0:
                    gc.collect()

        self.model.requires_grad_(False)
        return np.concatenate(all_attr, axis=0).astype(np.float32)

    # ---- persistence ----
    def save_embeddings(
        self,
        embeddings_dict: Dict[str, Any],
        output_path: str,
        compress: bool = True,
        include_statistics: bool = True,
    ) -> None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if include_statistics:
            stats = self.compute_embedding_statistics(embeddings_dict)
            stats_path = out_path.with_suffix(".statistics.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            if self.verbose:
                print(f"Saved embedding stats: {stats_path}")

        # Avoid saving memmaps directly; store paths instead
        save_dict: Dict[str, Any] = {}
        for k, v in embeddings_dict.items():
            if k in {"attention_weights"}:
                continue
            save_dict[k] = v

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(str(out_path), **save_dict)

        if self.verbose:
            print(f"Saved embeddings: {out_path}")

    @staticmethod
    def load_embeddings(embeddings_path: str, verbose: bool = True) -> Dict[str, Any]:
        p = Path(embeddings_path)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings file not found: {p}")

        data = np.load(str(p), allow_pickle=True)
        d: Dict[str, Any] = {k: data[k] for k in data.files}

        # Convert numpy string arrays to python list if needed
        if "attention_storage_paths" in d and isinstance(d["attention_storage_paths"], np.ndarray):
            d["attention_storage_paths"] = d["attention_storage_paths"].tolist()

        if verbose:
            keys = ", ".join(sorted(d.keys()))
            print(f"Loaded embeddings: {p.name} | keys=[{keys}]")

        return d

    def compute_embedding_statistics(self, embeddings_dict: Dict[str, Any]) -> Dict[str, Any]:
        pooled = np.asarray(embeddings_dict["pooled_embeddings"], dtype=np.float32)
        labels = np.asarray(embeddings_dict["labels"], dtype=np.int64)

        norms = np.linalg.norm(pooled, axis=1)
        unique_classes = np.unique(labels)

        class_centroids = []
        intraclass_vars = []
        for c in unique_classes:
            m = labels == c
            ec = pooled[m]
            if ec.shape[0] == 0:
                continue
            centroid = ec.mean(axis=0)
            class_centroids.append(centroid)
            dists = np.linalg.norm(ec - centroid[None, :], axis=1)
            intraclass_vars.append(float(np.mean(dists)))

        inter_dists = []
        if len(class_centroids) >= 2:
            C = np.stack(class_centroids, axis=0)
            for i in range(C.shape[0]):
                for j in range(i + 1, C.shape[0]):
                    inter_dists.append(float(np.linalg.norm(C[i] - C[j])))

        stats = {
            "embedding_dim": int(pooled.shape[1]),
            "embedding_norm_mean": float(norms.mean()),
            "embedding_norm_std": float(norms.std()),
            "n_samples": int(pooled.shape[0]),
            "n_classes": int(len(unique_classes)),
            "intraclass_distance_mean": float(np.mean(intraclass_vars)) if intraclass_vars else None,
            "interclass_distance_mean": float(np.mean(inter_dists)) if inter_dists else None,
        }
        return stats

    def cleanup(self) -> None:
        # Safe cleanup (tempdir is user-provided sometimes)
        try:
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Cleanup failed: {e}")
