from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StrataDefinition:
    """Definition of a sampling stratum."""
    stratum_id: str
    mask: np.ndarray  # Boolean mask identifying samples in this stratum
    n_samples: int
    target_samples: int  # Number of samples to draw from this stratum
    
    
@dataclass  
class StratifiedSample:
    """A stratified sample with provenance."""
    sample_indices: np.ndarray  # Indices into original dataset
    stratum_ids: List[str]  # Stratum ID for each sample
    weights: np.ndarray  # Sampling weights (for weighted statistics)
    

@dataclass
class SpatialMetricsSummary:
    """Summary statistics for spatial attention metrics."""
    mean: float
    std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_95_lower": float(self.ci_lower),
            "ci_95_upper": float(self.ci_upper),
            "median": float(self.median),
            "q25": float(self.q25),
            "q75": float(self.q75),
            "n_samples": int(self.n_samples),
        }


class StratifiedAttentionSampler:
    """
    Stratified sampling for spatial attention analysis.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def select_representative_samples(
        self,
        strata: List[StrataDefinition],
        metric_values: np.ndarray,
        n_per_stratum: int = 1,
        selection_criterion: str = "median",
    ) -> StratifiedSample:
        """
        Select representative samples closest to stratum summary statistic.
        
        Args:
            strata: Defined strata
            metric_values: Metric values for all samples (e.g., entropy)
            n_per_stratum: Number of representatives per stratum
            selection_criterion: "median", "mean", or "max"
            
        Returns:
            Selected representative samples
        """
        all_indices = []
        all_strata_ids = []
        
        for s in strata:
            available_idx = np.where(s.mask)[0]
            stratum_values = metric_values[available_idx]
            
            if selection_criterion == "median":
                target_val = np.median(stratum_values)
            elif selection_criterion == "mean":
                target_val = np.mean(stratum_values)
            elif selection_criterion == "max":
                target_val = np.max(stratum_values)
            else:
                raise ValueError(f"Unknown criterion: {selection_criterion}")
            
            # Find samples closest to target
            distances = np.abs(stratum_values - target_val)
            closest_local_idx = np.argsort(distances)[:n_per_stratum]
            selected_idx = available_idx[closest_local_idx]
            
            all_indices.extend(selected_idx.tolist())
            all_strata_ids.extend([s.stratum_id] * len(selected_idx))
        
        # Unit weights (these are representatives, not sampled for inference)
        weights = np.ones(len(all_indices), dtype=np.float32)
        
        return StratifiedSample(
            sample_indices=np.array(all_indices, dtype=np.int64),
            stratum_ids=all_strata_ids,
            weights=weights,
        )


class SpatialMetricsAggregator:
    """
    Compute weighted statistics for stratified samples.
    """
    
    @staticmethod
    def compute_summary(
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
    ) -> SpatialMetricsSummary:
        """
        Compute summary statistics with confidence intervals.
        
        Args:
            values: Metric values (N,)
            weights: Optional sampling weights (N,)
            confidence_level: CI confidence level (default: 0.95)
            
        Returns:
            Summary statistics
        """
        values = np.asarray(values, dtype=np.float32)
        n = len(values)
        
        if n == 0:
            return SpatialMetricsSummary(
                mean=np.nan, std=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                median=np.nan, q25=np.nan, q75=np.nan, n_samples=0,
            )
        
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
        
        # Weighted mean and variance
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean)**2, weights=weights)
        std = np.sqrt(variance)
        
        # Confidence interval (t-distribution for small samples)
        if n >= 2:
            se = std / np.sqrt(n)
            alpha = 1 - confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean
        
        # Percentiles (unweighted - more robust for small samples)
        median = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        
        return SpatialMetricsSummary(
            mean=float(mean),
            std=float(std),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            median=median,
            q25=q25,
            q75=q75,
            n_samples=n,
        )
    
    @staticmethod
    def compare_strata(
        values_a: np.ndarray,
        values_b: np.ndarray,
        weights_a: Optional[np.ndarray] = None,
        weights_b: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compare two strata using Welch's t-test.
        
        Returns:
            Dictionary with test statistics and p-value
        """
        values_a = np.asarray(values_a, dtype=np.float32)
        values_b = np.asarray(values_b, dtype=np.float32)
        
        if len(values_a) < 2 or len(values_b) < 2:
            return {
                "test": "welch_t_test",
                "statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "note": "Insufficient samples for test",
            }
        
        # Use weighted means if weights provided
        if weights_a is not None:
            mean_a = np.average(values_a, weights=weights_a)
            var_a = np.average((values_a - mean_a)**2, weights=weights_a)
        else:
            mean_a = np.mean(values_a)
            var_a = np.var(values_a, ddof=1)
        
        if weights_b is not None:
            mean_b = np.average(values_b, weights=weights_b)
            var_b = np.average((values_b - mean_b)**2, weights=weights_b)
        else:
            mean_b = np.mean(values_b)
            var_b = np.var(values_b, ddof=1)
        
        # Welch's t-test
        n_a = len(values_a)
        n_b = len(values_b)
        
        se_diff = np.sqrt(var_a/n_a + var_b/n_b)
        if se_diff == 0:
            t_stat = np.nan
            p_val = np.nan
        else:
            t_stat = (mean_a - mean_b) / se_diff
            # Welch-Satterthwaite degrees of freedom
            df = (var_a/n_a + var_b/n_b)**2 / (
                (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
            )
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
        
        return {
            "test": "welch_t_test",
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            "df": float(df) if not np.isnan(p_val) else np.nan,
        }

class AttentionDistributionalMetrics:
    """
    Compute distributional metrics for attention rollout vectors.
    
    References:
    - Gini (1912): Variability and Mutability
    - Shannon (1948): A Mathematical Theory of Communication
    - Lorenz (1905): Methods of Measuring Concentration of Wealth
    """
    
    @staticmethod
    def compute_gini_coefficient(attention_vector: np.ndarray) -> float:
        """
        Compute Gini Coefficient for attention distribution.
        
        High Gini (→1.0) = Focal attention (model looks at specific patches)
        Low Gini (→0.0) = Diffuse attention (model looks everywhere equally)
        
        Args:
            attention_vector: (P,) attention weights (must sum to ≈1.0)
            
        Returns:
            Gini coefficient in [0, 1]
            
        Mathematical Definition:
            G = (Σᵢ Σⱼ |aᵢ - aⱼ|) / (2n Σᵢ aᵢ)
            
        Interpretation:
            - G ≈ 0: Uniform distribution (texture bias)
            - G ≈ 1: Peaked distribution (focal attention on cell/object)
        """
        attention_vector = np.asarray(attention_vector, dtype=np.float64).flatten()
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Remove zeros (padding/masked patches)
        attention_vector = attention_vector[attention_vector > 0]
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Sort ascending
        sorted_attention = np.sort(attention_vector)
        n = len(sorted_attention)
        
        # Compute Gini using Lorenz curve area
        # G = 1 - 2*B where B is area under Lorenz curve
        cumsum = np.cumsum(sorted_attention)
        
        # Normalize cumulative sum
        total = cumsum[-1]
        if total == 0:
            return np.nan
            
        # Area under Lorenz curve (using trapezoidal rule)
        # Lorenz curve: (i/n, cumsum[i]/total)
        heights = cumsum / total
        # Area = sum of trapezoids
        lorenz_area = np.sum((heights[:-1] + heights[1:]) / 2.0) / n
        
        gini = 1.0 - 2.0 * lorenz_area
        
        return float(np.clip(gini, 0.0, 1.0))
    
    @staticmethod
    def compute_spatial_entropy(attention_vector: np.ndarray) -> float:
        """
        Compute Shannon Entropy for attention distribution.
        
        High Entropy = Diffuse/uniform attention (texture bias indicator)
        Low Entropy = Concentrated attention (focal attention)
        
        Args:
            attention_vector: (P,) attention weights (normalized to sum ≈1.0)
            
        Returns:
            Entropy in nats (natural logarithm base)
            
        Mathematical Definition:
            H = -Σᵢ pᵢ log(pᵢ)
            
        Interpretation:
            - H ≈ log(P): Maximum entropy (uniform distribution)
            - H ≈ 0: Minimum entropy (delta function)
        """
        attention_vector = np.asarray(attention_vector, dtype=np.float64).flatten()
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Ensure proper probability distribution
        attention_vector = attention_vector + 1e-10  # Numerical stability
        attention_vector = attention_vector / attention_vector.sum()
        
        # Remove near-zero probabilities (contribute nothing to entropy)
        attention_vector = attention_vector[attention_vector > 1e-10]
        
        if len(attention_vector) == 0:
            return np.nan
        
        # Shannon entropy (natural logarithm)
        entropy = -np.sum(attention_vector * np.log(attention_vector))
        
        return float(entropy)
    
    @staticmethod
    def compute_rollout_metrics_batch(
        rollout_vectors: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Gini and Entropy for batch of rollout vectors.
        
        Args:
            rollout_vectors: (N, P) attention rollout for N samples
            
        Returns:
            Dictionary with:
                - gini_coefficients: (N,) array
                - spatial_entropies: (N,) array
        """
        n_samples = rollout_vectors.shape[0]
        
        gini_values = np.zeros(n_samples, dtype=np.float32)
        entropy_values = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            gini_values[i] = AttentionDistributionalMetrics.compute_gini_coefficient(
                rollout_vectors[i]
            )
            entropy_values[i] = AttentionDistributionalMetrics.compute_spatial_entropy(
                rollout_vectors[i]
            )
        
        return {
            "gini_coefficients": gini_values,
            "spatial_entropies": entropy_values,
        }

class AttributionMagnitudeMetrics:
    """
    Quantitative descriptors for Integrated Gradients attribution vectors.
    """
    
    @staticmethod
    def compute_l1_magnitude(attribution: np.ndarray) -> float:
        """
        L1 norm: Sum of absolute attribution values.
        
        Args:
            attribution: (P,) attribution vector per patch
            
        Returns:
            L1 magnitude (non-negative scalar)
        """
        attribution = np.asarray(attribution, dtype=np.float64).flatten()
        return float(np.sum(np.abs(attribution)))
    
    @staticmethod
    def compute_l2_magnitude(attribution: np.ndarray) -> float:
        """
        L2 norm: Euclidean magnitude of attribution vector.
        
        Args:
            attribution: (P,) attribution vector per patch
            
        Returns:
            L2 magnitude (non-negative scalar)
        """
        attribution = np.asarray(attribution, dtype=np.float64).flatten()
        return float(np.linalg.norm(attribution))
    
    @staticmethod
    def compute_max_magnitude(attribution: np.ndarray) -> float:
        """
        Maximum absolute attribution value.
        
        Args:
            attribution: (P,) attribution vector per patch
            
        Returns:
            Maximum |attribution|
        """
        attribution = np.asarray(attribution, dtype=np.float64).flatten()
        return float(np.max(np.abs(attribution)))
    
    @staticmethod
    def compute_attribution_concentration(attribution: np.ndarray) -> Dict[str, float]:
        """
        Compute Gini and Entropy on *normalized* |attribution| distribution.
        
        Args:
            attribution: (P,) attribution vector per patch
            
        Returns:
            Dictionary with:
                - gini: Gini coefficient [0, 1]
                - entropy: Shannon entropy (nats)
        """
        attribution = np.asarray(attribution, dtype=np.float64).flatten()
        
        # Normalize to probability distribution (using absolute values)
        abs_attr = np.abs(attribution)
        
        if abs_attr.sum() == 0:
            return {"gini": np.nan, "entropy": np.nan}
        
        # Create pseudo-probability distribution
        prob_dist = abs_attr / abs_attr.sum()

        gini = AttentionDistributionalMetrics.compute_gini_coefficient(prob_dist)
        entropy = AttentionDistributionalMetrics.compute_spatial_entropy(prob_dist)
        
        return {
            "gini": float(gini),
            "entropy": float(entropy),
        }
    
    @staticmethod
    def compute_attribution_metrics_batch(
        attribution_vectors: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all attribution metrics for batch.
        
        Args:
            attribution_vectors: (N, P) attribution maps for N samples
            
        Returns:
            Dictionary with arrays:
                - l1_magnitudes: (N,)
                - l2_magnitudes: (N,)
                - max_magnitudes: (N,)
                - gini_coefficients: (N,)
                - spatial_entropies: (N,)
        """
        n_samples = attribution_vectors.shape[0]
        
        l1_mags = np.zeros(n_samples, dtype=np.float32)
        l2_mags = np.zeros(n_samples, dtype=np.float32)
        max_mags = np.zeros(n_samples, dtype=np.float32)
        gini_values = np.zeros(n_samples, dtype=np.float32)
        entropy_values = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            l1_mags[i] = AttributionMagnitudeMetrics.compute_l1_magnitude(
                attribution_vectors[i]
            )
            l2_mags[i] = AttributionMagnitudeMetrics.compute_l2_magnitude(
                attribution_vectors[i]
            )
            max_mags[i] = AttributionMagnitudeMetrics.compute_max_magnitude(
                attribution_vectors[i]
            )
            concentration = AttributionMagnitudeMetrics.compute_attribution_concentration(
                attribution_vectors[i]
            )
            gini_values[i] = concentration["gini"]
            entropy_values[i] = concentration["entropy"]
        
        return {
            "l1_magnitudes": l1_mags,
            "l2_magnitudes": l2_mags,
            "max_magnitudes": max_mags,
            "gini_coefficients": gini_values,
            "spatial_entropies": entropy_values,
        }

def compute_effect_size_cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for two groups.
    
    |d| < 0.2: Negligible
    |d| ≈ 0.5: Medium
    |d| ≈ 0.8: Large
    |d| > 1.2: Very large
    
    Args:
        group_a: First group values
        group_b: Second group values
        
    Returns:
        Cohen's d
    """
    group_a = np.asarray(group_a, dtype=np.float32)
    group_b = np.asarray(group_b, dtype=np.float32)
    
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    
    n_a = len(group_a)
    n_b = len(group_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return np.nan
    
    cohens_d = (mean_a - mean_b) / pooled_std
    
    return float(cohens_d)

def compare_groups_patch_level(values_a: np.ndarray, values_b: np.ndarray, confidencelevel: float = 0.95) -> Dict:
    """
    Patch-level comparison: mean difference, CI for diff, Welch t-test p-value, and Cohen's d.
    Returns descriptive inference (assumes independent samples).
    """
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)

    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test": "welch_ttest",
            "note": "Insufficient samples for test",
            "n_a": int(len(values_a)),
            "n_b": int(len(values_b)),
            "mean_diff": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
        }

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    n_a = len(values_a)
    n_b = len(values_b)

    mean_diff = mean_a - mean_b
    se_diff = np.sqrt(var_a / n_a + var_b / n_b)

    # Welch-Satterthwaite df
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a**2) / (n_a**2 * (n_a - 1)) + (var_b**2) / (n_b**2 * (n_b - 1))
    df = df_num / df_den if df_den > 0 else np.nan

    alpha = 1.0 - confidencelevel
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df=df) if not np.isnan(df) else np.nan
    ci_lower = mean_diff - tcrit * se_diff if not np.isnan(tcrit) else np.nan
    ci_upper = mean_diff + tcrit * se_diff if not np.isnan(tcrit) else np.nan

    # Welch p-value via existing aggregator utility
    ttest = SpatialMetricsAggregator.compare_strata(values_a, values_b)

    d = compute_effect_size_cohens_d(values_a, values_b)

    return {
        "test": "welch_ttest",
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": float(mean_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "df": float(ttest.get("df", np.nan)),
        "statistic": float(ttest.get("statistic", np.nan)),
        "p_value": float(ttest.get("p_value", np.nan)),
        "cohens_d": float(d),
    }
