"""
Author: Rodrigo Sá
Date: 2025
"""

import os
import multiprocessing
import shutil
from pathlib import Path
import random
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from timeit import default_timer as timer
from typing import Optional, Dict, List, Tuple, Union
from collections import defaultdict
import warnings

from canonical_splits import extract_slide_id_from_base
try:
    from basicpy import BaSiC
except ImportError as e:
    raise ImportError(
        "BaSiCPy is required for this preprocessing pipeline. "
        "Install it with: pip install basicpy"
    ) from e

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "tifffile is required for this pipeline. "
        "Install it with: pip install tifffile"
    ) from e


SUPPORTED_EXTENSIONS = {'.tif', '.tiff'}


class MicroscopyPreprocessor:
    """
    Two modes:
    'baseline': Global linear normalization only
    'optimized': Adds BaSiCPy illumination correction
    """

    def __init__(self, method: str = 'baseline',
                 basic_model: Optional[BaSiC] = None,
                 bit_depth: int = 12):
        """
        Initialize preprocessor.

        Args:
            method: 'baseline' or 'optimized'
            basic_model: Pre-fitted BaSiCPy model
            bit_depth: Original bit depth
        """
        if method not in ['baseline', 'optimized']:
            raise ValueError(
                f"method must be 'baseline' or 'optimized', got '{method}'"
            )

        self.method = method
        self.basic_model = basic_model
        self.bit_depth = bit_depth
        self.bit_depth_max = (2 ** bit_depth) - 1

        if method == 'optimized':
            if basic_model is None:
                raise ValueError(
                    "BaSiCPy model required for 'optimized' method. "
                    "Provide a fitted BaSiC instance."
                )

    def _global_linear_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Global linear normalization preserving photometric consistency.

        Args:
            image: Raw image
        Returns:
            Normalized float32 image in [0,1]
        """
        normalized = image.astype(np.float32) / self.bit_depth_max

        if normalized.max() > 1.0 or normalized.min() < 0.0:
            warnings.warn(
                f"Normalized image has values outside [0,1]: "
                f"[{normalized.min():.4f}, {normalized.max():.4f}]. "
                f"Check bit_depth parameter (current: {self.bit_depth}). "
                f"Clipping to valid range."
            )
            normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess single microscopy image with photometric scale preservation.

        Args:
            image: Input image

        Returns:
            Preprocessed float32 image [0-1] with preserved photometric scale
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        if len(image.shape) != 2:
            raise ValueError("Expected single-channel (grayscale) image, got shape "
                            f"{image.shape}")
        gray = image.copy()

        # Global linear normalization (preserves photometric scale)
        img_normalized = self._global_linear_normalize(gray)

        if self.method == 'optimized' and self.basic_model is not None:
            img_3d = img_normalized[np.newaxis, :, :]
            img_corrected = self.basic_model.transform(img_3d)
            img_corrected = img_corrected[0]
            img_corrected = np.clip(img_corrected, 0.0, 1.0)
        else:
            img_corrected = img_normalized

        return img_corrected.astype(np.float32)

def load_12bit_tif(image_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load TIF image with metadata extraction.

    Args:
        image_path: Path to TIF file

    Returns:
        Tuple of (image_array, metadata_dict)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with tifffile.TiffFile(image_path) as tif:
        image = tif.asarray()
        container_bit_depth = tif.pages[0].bitspersample

        metadata = {
            "container_bit_depth": container_bit_depth,
            "dtype": str(image.dtype),
            "shape": image.shape,
            "photometric": tif.pages[0].photometric,
            "effective_bit_depth": None,
        }

    return image, metadata

def infer_effective_bit_depth(
    img_array: np.ndarray,
    config_bit_depth: int,
    container_bit_depth: int,
) -> int:
    
    bit_depth_max = (2 ** config_bit_depth) - 1
    max_val = int(img_array.max())

    if container_bit_depth == config_bit_depth:
        return config_bit_depth

    if container_bit_depth > config_bit_depth and max_val <= bit_depth_max:
        return config_bit_depth

    warnings.warn(
        f"Bit-depth mismatch for image: config={config_bit_depth}-bit, "
        f"container={container_bit_depth}-bit, max_pixel={max_val}. "
        "Please verify bit-depth configuration."
    )
    return config_bit_depth

def get_image_paths(dataset_dir: Union[str, Path]) -> List[Path]:
    dataset_path = Path(dataset_dir)
    exts = {e.lower() for e in SUPPORTED_EXTENSIONS}
    paths = []
    for p in dataset_path.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)

def stratified_sample_paths_by_slide(
    image_paths: List[Path],
    n_samples: int,
    random_seed: int = 42,
) -> List[Path]:
    rng = random.Random(random_seed)

    # Group image paths by slide identifier
    slide_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for p in image_paths:
        slide_id = extract_slide_id_from_base(p)
        slide_to_paths[slide_id].append(p)

    slide_ids = list(slide_to_paths.keys())
    n_slides = len(slide_ids)
    if n_slides == 0:
        raise RuntimeError("No images available for BaSiC fitting.")

    base_per_slide = max(n_samples // n_slides, 1)

    sampled: List[Path] = []

    for sid in slide_ids:
        paths = slide_to_paths[sid]
        k = min(base_per_slide, len(paths))
        sampled.extend(rng.sample(paths, k))

    if len(sampled) < n_samples:
        remaining = n_samples - len(sampled)
        remaining_pool = [p for p in image_paths if p not in sampled]
        if remaining_pool:
            extra_k = min(remaining, len(remaining_pool))
            sampled.extend(rng.sample(remaining_pool, extra_k))

    if len(sampled) > n_samples:
        sampled = rng.sample(sampled, n_samples)

    return sampled

def fit_basicpy_from_paths(
    image_paths: List[Path],
    output_path: str,
    n_samples: int = 100,
    bit_depth: int = 12,
    random_seed: int = 42
) -> BaSiC:
    """
    Fit BaSiCPy model on specified image paths.

    Args:
        image_paths: List of Path objects to TRAINING images only
        output_path: Path to save fitted model
        n_samples: Number of images to sample for fitting (100-200 recommended)
        bit_depth: Bit depth of images
        random_seed: Random seed for reproducibility

    Returns:
        Fitted BaSiCPy model

    Raises:
        RuntimeError: If insufficient valid images found
    """

    if n_samples < 100:
        warnings.warn(
            f"Using only {n_samples} images for BaSiC fitting. "
            f"Peng et al. (2017) recommend 100-200 images for robust estimates."
        )

    if len(image_paths) < n_samples:
        warnings.warn(
            f"Only {len(image_paths)} training images available for BaSiC fitting, "
            f"less than requested {n_samples}. Using all available training images."
        )
        n_samples = len(image_paths)

    if len(image_paths) < 10:
        raise RuntimeError(
            f"Insufficient images for BaSiC fitting. Found {len(image_paths)} images, "
            f"but need at least 10 for stable flatfield/darkfield estimation."
        )

    np.random.seed(random_seed)
    sampled_paths = stratified_sample_paths_by_slide(
        image_paths=image_paths,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    slide_counts: Dict[str, int] = defaultdict(int)
    species_counts: Dict[str, int] = defaultdict(int)

    for p in sampled_paths:
        sid = extract_slide_id_from_base(p)
        slide_counts[sid] += 1
        stem_parts = p.stem.split("_")
        if len(stem_parts) >= 2:
            species = stem_parts[1]
        else:
            species = p.parent.name
        species_counts[species] += 1

    for sp, count in sorted(species_counts.items()):
        sp_slides = {sid for sid in slide_counts if f"_{sp}_" in sid}
        print(f"   Species {sp}: {count} images from {len(sp_slides)} slides")

    images = []
    target_size = None

    for img_path in tqdm(sampled_paths, desc="  Progress", ncols=80):
        try:
            if img_path.suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError(f"Non-TIF image encountered during BaSiC fitting: {img_path}")
            
            img_array, metadata = load_12bit_tif(str(img_path))

            if img_array.ndim != 2:
                raise ValueError(
                    f"Expected single-channel TIF, got shape {img_array.shape} "
                    f"for {img_path.name}"
                )

            container_bit_depth = metadata["container_bit_depth"]
            effective_bit_depth = infer_effective_bit_depth(
                img_array=img_array,
                config_bit_depth=bit_depth,
                container_bit_depth=container_bit_depth,
            )
            metadata["effective_bit_depth"] = effective_bit_depth

            bit_depth_max = (2 ** effective_bit_depth) - 1
            img_norm = img_array.astype(np.float32) / bit_depth_max

            if target_size is None:
                target_size = img_norm.shape
            else:
                if img_norm.shape != target_size:
                    img_norm = cv2.resize(
                        img_norm, (target_size[1], target_size[0]),
                        interpolation=cv2.INTER_LANCZOS4
                    )

            images.append(img_norm)

        except Exception as e:
            warnings.warn(f"Failed to load {img_path.name}: {e}")
            continue

    if len(images) < 10:
        raise RuntimeError(
            f"Only loaded {len(images)} images successfully. "
            "BaSiC requires at least 10 images (preferably 100-200) for stable fitting."
        )

    print(f"\n  Successfully loaded: {len(images)} TRAINING images ({target_size[0]}×{target_size[1]} px)")

    images_stack = np.stack(images, axis=0)

    print(f"  Fitting BaSiCPy model")
    basic_model = BaSiC(
        get_darkfield=True,
        smoothness_flatfield=1.0,
        smoothness_darkfield=1.0
    )
    basic_model.fit(images_stack)

    empty_mask = calculate_empty_field_mask(images_stack)
    bgpixels_before = images_stack[:, empty_mask]
    bgstd_before = float(bgpixels_before.std())

    images_corrected = basic_model.transform(images_stack)
    images_corrected = np.clip(images_corrected, 0.0, 1.0)
    bgpixels_after = images_corrected[:, empty_mask]
    bgstd_after = float(bgpixels_after.std())
    if bgstd_before > 0:
        reduction_ratio = 1.0 - (bgstd_after / bgstd_before)
    else:
        reduction_ratio = float("nan")

    print(f"Background std before BaSiC (empty-field): {bgstd_before:.6f}")
    print(f"Background std after  BaSiC (empty-field): {bgstd_after:.6f}")
    if not np.isnan(reduction_ratio):
        print(f"Background std reduction (empty-field): {reduction_ratio * 100:.1f}%")
    else:
        print("Background std reduction (empty-field): NA (bgstd_before == 0)")

    gamma_prime, mean_diff_before, mean_diff_after = calculate_correction_score(
        images_stack, images_corrected
    )
    print(f"Correction score Γ'(I_corr): {gamma_prime:.4f}")
    print(f"Mean abs. diff before: {mean_diff_before:.6f}")
    print(f"Mean abs. diff after : {mean_diff_after:.6f}")

    if gamma_prime > 1.05:
        raise RuntimeError(
            f"BaSiC correction is degrading image quality (Γ'={gamma_prime:.4f} > 1.05). "
            "This implies the correction model is overfitting to artifacts or background estimation failed. "
            "Preprocessing aborted to protect data integrity."
        )
    if gamma_prime < 0.95:
        print(f"Γ'(I_corr) indicates uniformity improvement ≈ {(1.0 - gamma_prime) * 100:.1f}%")
    elif gamma_prime > 1.0:
        warnings.warn(
            f"Minimal/Negative correction effect (Γ'={gamma_prime:.4f}). "
            "Consider using 'baseline' method (normalization only) if this persists."
        )
    else:
        print("Γ'(I_corr) ≈ 1: Global correction effect is small (dataset already uniform).")

    print(f"\n  BaSiCPy Fitting Complete:")
    print(f"    Flatfield shape: {basic_model.flatfield.shape}")
    print(f"    Flatfield range: [{basic_model.flatfield.min():.4f}, "
          f"{basic_model.flatfield.max():.4f}]")
    print(f"    Flatfield mean: {basic_model.flatfield.mean():.4f}")

    if basic_model.darkfield is not None:
        print(f"    Darkfield shape: {basic_model.darkfield.shape}")
        print(f"    Darkfield range: [{basic_model.darkfield.min():.4f}, "
              f"{basic_model.darkfield.max():.4f}]")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        flatfield=basic_model.flatfield,
        darkfield=basic_model.darkfield,
        baseline_drift=getattr(basic_model, 'baseline', None),
        bit_depth_max=bit_depth_max,
        n_samples=len(images),
        image_shape=target_size,
        method='BaSiCPy',
        random_seed=random_seed,
        bgstdbefore=bgstd_before,
        bgstdafter=bgstd_after,
        bgreductionratio=reduction_ratio,
        gamma_prime=gamma_prime,
        meandiff_before=mean_diff_before,
        meandiff_after=mean_diff_after,
    )
    print(f"  BaSiCPy model saved to: {output_path}")

    return basic_model

def calculate_empty_field_mask(
    images_stack: np.ndarray,
    threshold_percentile: float = 90.0,
    min_coverage: float = 0.05,
) -> np.ndarray:
    """
    Identify empty-field regions (cell-free background) in brightfield images.

    Brightfield assumption: cells appear dark, background appears bright.
    The mask is derived from the temporal median image thresholded at a high
    intensity percentile, retaining only the brightest (background) pixels.

    Args:
        images_stack: Normalised image stack of shape (N, H, W) in [0, 1].
        threshold_percentile: Intensity percentile used to define the background
            threshold on the median image (default: 90th percentile).
        min_coverage: Minimum fraction of pixels that must be classified as
            background; warns if coverage falls below this value (default: 0.05).

    Returns:
        Boolean mask of shape (H, W); True indicates a background pixel.
    """
    # images_stack: (N, H, W), normalised to [0, 1]
    median_image = np.median(images_stack, axis=0)  # (H, W)
    threshold = np.percentile(median_image, threshold_percentile)
    empty_mask = median_image > threshold

    coverage = empty_mask.sum() / empty_mask.size
    if coverage < min_coverage:
        warnings.warn(
            f"Empty-field mask covers only {coverage * 100:.1f}% of pixels. "
            "Dataset may be too dense; the uniformity metric may be unreliable."
        )
    return empty_mask

def calculate_correction_score(
    images_before: np.ndarray,
    images_after: np.ndarray,
    max_pairs_per_image: int = 10,
) -> Tuple[float, float, float]:
    """
    Compute the correction score Γ'(I_corr) as defined in Peng et al. (2017).

    Interpretation:
        Γ' < 1  →  correction improves inter-image consistency (desired)
        Γ' ≈ 1  →  negligible correction effect
        Γ' > 1  →  correction degrades consistency (abort threshold: Γ' > 1.05)

    The score is computed as the ratio of mean absolute pairwise pixel differences
    after correction to the same quantity before correction, using a windowed
    pair-sampling strategy for computational tractability.

    Args:
        images_before: Normalised image stack before correction, shape (N, H, W).
        images_after: Corrected image stack, shape (N, H, W).
        max_pairs_per_image: Maximum number of forward-adjacent pairs per image
            (default: 10). Controls the O(N²) cost of exhaustive pairing.

    Returns:
        Tuple of (gamma_prime, mean_diff_before, mean_diff_after).
    """
    n = images_before.shape[0]
    diff_before = []
    diff_after = []

    for i in range(n):
        j_max = min(i + max_pairs_per_image + 1, n)
        for j in range(i + 1, j_max):
            db = np.abs(images_before[i] - images_before[j]).mean()
            da = np.abs(images_after[i] - images_after[j]).mean()
            diff_before.append(db)
            diff_after.append(da)

    if not diff_before:
        return 1.0, 0.0, 0.0

    mean_before = float(np.mean(diff_before))
    mean_after = float(np.mean(diff_after))
    gamma_prime = mean_after / mean_before if mean_before > 0 else 1.0
    return gamma_prime, mean_before, mean_after


def apply_preprocessing(image: np.ndarray, method: str,
                       basic_model: Optional[BaSiC] = None,
                       bit_depth: int = 12) -> np.ndarray:
    """
    Apply preprocessing pipeline to single image.

    Args:
        image: Input image
        method: 'baseline' or 'optimized'
        basic_model: Pre-fitted BaSiCPy model
        bit_depth: Original bit depth

    Returns:
        Preprocessed float32 image [0-1] with preserved photometric scale
    """
    preprocessor = MicroscopyPreprocessor(
        method=method,
        basic_model=basic_model,
        bit_depth=bit_depth
    )
    return preprocessor.preprocess(image)

def resize_and_preprocess_image(args: Tuple) -> Dict:

    img_path, target_path, image_size, method, basic_model, bit_depth = args

    try:
        # Load image
        if img_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Non-TIF image encountered: {img_path}")
        
        img_array, metadata = load_12bit_tif(str(img_path))

        container_bit_depth = metadata["container_bit_depth"]
        effective_bit_depth = infer_effective_bit_depth(
            img_array=img_array,
            config_bit_depth=bit_depth,
            container_bit_depth=container_bit_depth,
        )
        metadata["effective_bit_depth"] = effective_bit_depth
        
        if img_array.ndim != 2:
            raise ValueError(
                f"Expected single-channel TIF, got shape {img_array.shape} "
                f"for {img_path.name}"
            )

        preprocessed_float32 = apply_preprocessing(
            img_array, method, basic_model, effective_bit_depth
        )
        
        img_resized_float32 = cv2.resize(
            preprocessed_float32,
            (image_size, image_size),
            interpolation=cv2.INTER_LANCZOS4
        )

        if img_resized_float32.min() < -0.01 or img_resized_float32.max() > 1.01:
            warnings.warn(
                f"Output image {img_path.name} has values outside [0,1]: "
                f"[{img_resized_float32.min():.4f}, {img_resized_float32.max():.4f}]. "
                f"Clipping to valid range."
            )
            img_resized_float32 = np.clip(img_resized_float32, 0.0, 1.0)

        # Save as NPY (preserves float32 precision)
        target_path = target_path.with_suffix('.npy')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target_path), img_resized_float32)

        return {
            'success': True,
            'source': str(img_path),
            'target': str(target_path),
            'metadata': metadata,
            'format': f'NPY float32 [0-1] with {method} BaSiCPy'
        }

    except Exception as e:
        return {
            'success': False,
            'source': str(img_path),
            'error': str(e)
        }

def calculate_optimal_workers(cpu_count: int, min_workers: int = 1) -> int:
    """Calculate optimal number of parallel workers."""
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return max(cpu_count - 1, min_workers)
    else:
        return max(cpu_count - 2, min_workers)

def preprocess_dataset(
    source_dir: str,
    target_dir: str,
    config: Dict,
    basic_model: Optional[BaSiC] = None,
    num_workers: Optional[int] = None,
    force_reprocess: bool = False,
    chunk_size: int = 500,
) -> bool:
    """
    Args:
        source_dir: Path to source dataset (TIF images).
        target_dir: Path to output directory.
        config: Configuration dictionary with:
            - preprocessing['method']: 'baseline' or 'optimized'
            - bit_depth: Original bit depth
            - image_size: Target size for resizing
        basic_model: Pre-fitted BaSiCPy model.
        num_workers: Number of parallel workers.
        force_reprocess: If True, reprocess even if target .npy exists.
        chunk_size: Number of images per processing chunk.

    Returns:
        True if successful, False otherwise.
    """
    cpu_count = multiprocessing.cpu_count()
    if num_workers is None:
        num_workers = calculate_optimal_workers(cpu_count)

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    preprocessing_config = config["preprocessing"]
    method = preprocessing_config["method"]
    bit_depth = config.get("bit_depth", 12)
    image_size = config["image_size"]

    # Collect all source images
    image_paths = get_image_paths(source_path)
    total_images = len(image_paths)
    if total_images == 0:
        print(f" ERROR: No images found in {source_dir}")
        return False

    # Prepare processing tasks
    tasks: List[Tuple] = []
    for img_path in image_paths:
        relative_path = img_path.relative_to(source_path)
        target_img_path = target_path / relative_path.with_suffix(".npy")
        if not target_img_path.exists() or force_reprocess:
            tasks.append(
                (img_path, target_img_path, image_size, method, basic_model, bit_depth)
            )

    if not tasks:
        print(" All images already preprocessed!")
        return True

    # Process images
    processed = 0
    failed = 0
    failed_images = []
    start_time = timer()

    if num_workers == 1:
        print(" Using sequential processing")
        with tqdm(total=len(tasks), desc=" Progress", unit="img", ncols=100) as pbar:
            for task in tasks:
                result = resize_and_preprocess_image(task)
                if result["success"]:
                    processed += 1
                else:
                    failed += 1
                    failed_images.append(
                        {
                            "source": result["source"],
                            "error": result.get("error", "Unknown"),
                        }
                    )
                pbar.update(1)
    else:
        print(f" Using parallel processing ({num_workers} workers)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(tasks), desc=" Progress", unit="img", ncols=100) as pbar:
                for chunk_start in range(0, len(tasks), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(tasks))
                    chunk_tasks = tasks[chunk_start:chunk_end]
                    futures = {
                        executor.submit(resize_and_preprocess_image, task): task
                        for task in chunk_tasks
                    }

                    for future in as_completed(futures):
                        result = future.result()
                        if result["success"]:
                            processed += 1
                        else:
                            failed += 1
                            failed_images.append(
                                {
                                    "source": result["source"],
                                    "error": result.get("error", "Unknown"),
                                }
                            )
                        pbar.update(1)

    end_time = timer()
    elapsed = end_time - start_time

    target_images = list(target_path.rglob("*.npy"))
    if target_images:
        total_size_mb = sum(f.stat().st_size for f in target_images) / (1024 * 1024)
    else:
        total_size_mb = 0.0

    success = failed == 0

    print(f" Processed: {processed}/{total_images} images")
    if failed > 0:
        print(f" Failed: {failed} images")
    print(f" Time: {elapsed:.2f}s ({len(tasks)/elapsed:.1f} images/sec)")
    print(f" Output size: {total_size_mb:.1f} MB")
    if method == "optimized":
        print(" Correction: BaSiCPy")
    print(f" Status: {'COMPLETE' if success else 'PARTIAL'}")

    return success

def zip_preprocessed_data(source_dir: str, zip_path: str) -> bool:
    """
    Create a ZIP archive of preprocessed data.
    """
    if not os.path.exists(source_dir):
        print(f" ERROR: Source not found: {source_dir}")
        return False

    zip_path = os.path.abspath(zip_path)
    zip_dir = os.path.dirname(zip_path)
    os.makedirs(zip_dir, exist_ok=True)

    zip_base_name = os.path.splitext(os.path.basename(zip_path))[0]
    local_zip_base = os.path.join(zip_dir, zip_base_name)

    root_dir = os.path.dirname(source_dir)
    base_dir = os.path.basename(source_dir)

    try:
        archive_path = shutil.make_archive(
            base_name=local_zip_base,
            format="zip",
            root_dir=root_dir,
            base_dir=base_dir,
        )
        zip_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f" Location: {archive_path}")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
        return False
