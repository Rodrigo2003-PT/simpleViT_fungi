"""
Usage
-----
    python data/canonical_splits.py \\
        --tiff-dir /path/to/raw_tiff_dataset \\
        --output   split_indices.json \\
        --test-size 0.15 \\
        --seed      42

Author: Rodrigo Sá
Date:   2025
"""

import json
import argparse
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from collections import Counter, defaultdict


def get_base_paths_from_directory(dataset_dir: Path,
                                  extensions: Set[str]) -> List[Path]:
    """
    Extract base paths from dataset directory.

    Args:
        dataset_dir: Path to dataset root
        extensions: Set of valid extensions

    Returns:
        List of relative base paths (e.g., 'Candida albicans/20250923_alb_K_1')
    """
    base_paths: List[Path] = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        for img_path in class_dir.rglob('*'):
            if img_path.suffix.lower() in extensions:
                rel_path = img_path.relative_to(dataset_dir)
                base_path = rel_path.with_suffix('')
                base_paths.append(base_path)

    return sorted(base_paths)


def get_tiff_samples(tiff_dir: Path) -> List[Path]:
    """
    Enumerate all TIFF images in a dataset directory and return their
    extension-stripped relative base paths.

    The dataset directory is expected to follow the standard layout::

        tiff_dir/
        ├── Candida albicans/
        │   ├── 20250101_alb_A_patch_0001.tif
        │   └── ...
        └── Candida glabrata/
            ├── 20250101_gla_B_patch_0001.tif
            └── ...

    Args:
        tiff_dir: Root directory of the TIFF dataset.

    Returns:
        Sorted list of relative base paths with extensions stripped
        (e.g. ``Path('Candida albicans/20250101_alb_A_patch_0001')``).

    Raises:
        ValueError: If no TIFF images are found in ``tiff_dir``.
    """
    tiff_extensions = {'.tif', '.tiff', '.TIF', '.TIFF'}
    base_paths = get_base_paths_from_directory(tiff_dir, tiff_extensions)

    print(f" TIFF images found: {len(base_paths)}")

    if len(base_paths) == 0:
        raise ValueError(
            f"No TIFF images found in '{tiff_dir}'. "
            "Check that the directory contains class subdirectories with "
            ".tif / .tiff files."
        )

    return base_paths

def extract_slide_id_from_base(p: Path) -> str:
    stem = p.name
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    date, species_code, letter = parts[0], parts[1], parts[2]
    return f"{date}_{species_code}_{letter}"

def analyze_slide_distribution(slide_to_indices: Dict[str, List[int]],
                               slide_to_class: Dict[str, str]) -> None:
    """
    Performs rigorous descriptive statistics at the slide level.

    Reports:
    - Number of unique slides per class.
    - Intra-slide sampling variance (min/max images per slide).
    - List of slide identifiers for audit purposes.
    """
    print("\n" + "=" * 70)
    print("SLIDE-LEVEL GRANULARITY ANALYSIS")
    print("=" * 70)

    # Group data by class
    class_stats = defaultdict(list)
    for sid, indices in slide_to_indices.items():
        cls = slide_to_class[sid]
        n_images = len(indices)
        class_stats[cls].append((sid, n_images))

    # Calculate and print statistics
    for cls in sorted(class_stats.keys()):
        slides_data = class_stats[cls]  # List of tuples (slide_id, n_images)
        n_slides = len(slides_data)

        # Extract counts for stats
        counts = [n for _, n in slides_data]
        min_img = np.min(counts)
        max_img = np.max(counts)
        mean_img = np.mean(counts)
        std_img = np.std(counts)

        print(f"\nCLASS: {cls}")
        print(f"  • Unique Slides (N): {n_slides}")
        print(f"  • Images per Slide: Min={min_img}, Max={max_img}, "
              f"Mean={mean_img:.1f} ± {std_img:.1f}")

        print(f"  • Slide ID List (Count):")
        for sid, count in sorted(slides_data, key=lambda x: x[0]):
            print(f"      - {sid} (n={count})")

    print("=" * 70 + "\n")


def create_stratified_group_split(
        tiff_bases: List[Path],
        test_size: float,
        random_seed: int,
) -> Dict:
    """
    Create a constrained slide-stratified train/test split.

    The search runs for up to ``MAX_ATTEMPTS`` random candidate splits and
    retains the one whose test-set class ratio deviates least from the
    global ratio.  A split is accepted immediately when the deviation falls
    below ``TOLERANCE`` (2 %).

    Args:
        tiff_bases: Sorted list of relative base paths for all TIFF samples
            (extension-stripped, as returned by ``get_tiff_samples``).
        test_size: Target fraction of slides to assign to the test partition.
        random_seed: Seed for the randomised search.

    Returns:
        Dictionary containing train/test base paths, class distribution,
        drift metrics, and provenance metadata — ready for JSON serialisation.

    Raises:
        ValueError: If ``test_size`` is not in (0, 1) or ``tiff_bases`` is empty.
    """
    # Internal alias — all logic below operates on this list unchanged
    common_bases = tiff_bases
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(common_bases) == 0:
        raise ValueError("tiff_bases is empty; nothing to split")

    labels = [str(p.parent) for p in common_bases]
    unique_classes = sorted(set(labels))

    slide_ids = [extract_slide_id_from_base(p) for p in common_bases]

    slide_to_indices: Dict[str, List[int]] = defaultdict(list)
    slide_to_class: Dict[str, str] = {}

    for idx, (sid, cls) in enumerate(zip(slide_ids, labels)):
        slide_to_indices[sid].append(idx)
        if sid in slide_to_class and slide_to_class[sid] != cls:
            raise ValueError(f"Mixed-label slide detected: {sid}")
        slide_to_class[sid] = cls

    class_to_slides: Dict[str, List[str]] = defaultdict(list)
    for sid, cls in slide_to_class.items():
        class_to_slides[cls].append(sid)

    slide_to_count = {sid: len(idxs) for sid, idxs in slide_to_indices.items()}
    total_images_global = len(common_bases)
    target_cls = unique_classes[0]
    target_count_global = labels.count(target_cls)
    global_ratio = target_count_global / total_images_global

    print(f"Target Global Image Ratio ({Path(target_cls).name}): {global_ratio:.1%}")

    best_split_slides = None
    min_drift = float('inf')
    best_test_ratio = 0.0

    search_rng = np.random.RandomState(random_seed)
    MAX_ATTEMPTS = 2000
    TOLERANCE = 0.02

    print(f"Searching for split with Image Drift < {TOLERANCE:.1%} (Max {MAX_ATTEMPTS} attempts)...")

    for i in range(MAX_ATTEMPTS):
        attempt_rng = np.random.RandomState(search_rng.randint(0, 1000000))
        current_test_slides = set()

        for cls, slides in class_to_slides.items():
            n_slides = len(slides)
            if n_slides == 0: continue

            n_test = max(1, int(round(test_size * n_slides)))
            if n_test >= n_slides: n_test = n_slides - 1

            chosen = attempt_rng.choice(slides, size=n_test, replace=False)
            current_test_slides.update(chosen)

        if not current_test_slides: continue

        test_total = sum(slide_to_count[s] for s in current_test_slides)
        test_target = sum(slide_to_count[s] for s in current_test_slides
                          if slide_to_class[s] == target_cls)

        if test_total == 0: continue

        test_ratio = test_target / test_total
        drift = abs(test_ratio - global_ratio)

        if drift < min_drift:
            min_drift = drift
            best_split_slides = current_test_slides
            best_test_ratio = test_ratio

        if drift <= TOLERANCE:
            print(f"  [Success] Attempt {i + 1}: Drift {drift:.2%} (Test Ratio: {test_ratio:.1%})")
            break

    if min_drift > TOLERANCE:
        print(f"  [Warning] Could not meet tolerance. Best drift: {min_drift:.2%}")

    test_idx = sorted(
        idx for sid, idxs in slide_to_indices.items()
        if sid in best_split_slides
        for idx in idxs
    )
    train_idx = sorted(set(range(len(common_bases))) - set(test_idx))

    train_bases = [common_bases[i] for i in train_idx]
    test_bases = [common_bases[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)

    split_data: Dict = {
        "train_base_paths": [p.as_posix() for p in train_bases],
        "test_base_paths": [p.as_posix() for p in test_bases],
        "classes": unique_classes,
        "random_seed": random_seed,
        "test_size": test_size,
        "total_samples": len(common_bases),
        "train_count": len(train_bases),
        "test_count": len(test_bases),
        "class_distribution": {
            "train": dict(zip(unique_train, map(int, counts_train))),
            "test": dict(zip(unique_test, map(int, counts_test))),
        },
        "sampling_method": "Constrained Stratified Group Sampling",
        "constraints": {
            "slide_level": f"Stratified (approx {test_size:.0%})",
            "image_level": f"Global Ratio Match (Tolerance {TOLERANCE:.1%})"
        },
        "drift_metrics": {
            "global_ratio": float(global_ratio),
            "test_ratio": float(best_test_ratio),
            "drift_achieved": float(min_drift)
        }
    }
    return split_data


def _resolve_dataset_root(path: Path) -> Tuple[Path, Optional[Path]]:
    """
    Resolve the dataset root from a directory path or a ``.zip`` archive.

    If ``path`` is a ZIP file it is extracted to a temporary directory.
    If the archive contains a single top-level directory that directory is
    used as the dataset root; otherwise the extraction root itself is used.
    The caller is responsible for removing the temporary directory via the
    returned ``temp_dir`` handle.

    Args:
        path: Path to the TIFF dataset directory or a ``.zip`` archive.

    Returns:
        ``(dataset_root, temp_dir)`` — ``temp_dir`` is ``None`` when the
        input is already a directory.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is neither a directory nor a ``.zip`` file.
    """
    if path.suffix.lower() == ".zip":
        if not path.exists():
            raise FileNotFoundError(f"TIFF ZIP file not found: {path}")

        temp_dir = Path(tempfile.mkdtemp(prefix="canonical_tiff_"))
        print(f"\n[TIFF] Input is a ZIP archive: {path}")
        print(f"[TIFF] Extracting to temporary directory: {temp_dir}")

        with zipfile.ZipFile(str(path), "r") as zip_ref:
            zip_ref.extractall(str(temp_dir))

        candidates = [d for d in temp_dir.iterdir() if d.is_dir()]
        if len(candidates) == 1:
            dataset_root = candidates[0]
            print(f"[TIFF] Using single top-level directory as dataset root: {dataset_root}")
        else:
            dataset_root = temp_dir
            print(f"[TIFF] Using extraction root as dataset root: {dataset_root}")

        return dataset_root, temp_dir

    # Directory input
    if not path.exists():
        raise FileNotFoundError(f"TIFF directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"TIFF path must be a directory or .zip, got: {path}")

    print(f"\n[TIFF] Using existing directory as dataset root: {path}")
    return path, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a canonical slide-stratified train/test split for a "
            "TIFF brightfield microscopy dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiff-dir",
        type=Path,
        required=True,
        help="Path to TIFF dataset directory or ZIP archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("split_indices.json"),
        help="Output path for the canonical split JSON file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion of slides assigned to the test partition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the constrained search.",
    )

    args = parser.parse_args()

    tiff_root, tiff_temp = _resolve_dataset_root(args.tiff_dir)

    try:
        # Enumerate TIFF samples
        tiff_bases = get_tiff_samples(tiff_root)

        # Create slide-stratified split
        split_data = create_stratified_group_split(
            tiff_bases, args.test_size, args.seed
        )

        # Persist canonical split
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(split_data, f, indent=4)

        print(f"\n Split saved to: {args.output}")
        print(f" Total samples:  {split_data['total_samples']}")
        print(f" Train:          {split_data['train_count']}")
        print(f" Test:           {split_data['test_count']}")
        print(f" Drift achieved: {split_data['drift_metrics']['drift_achieved']:.2%}")

    finally:
        if tiff_temp is not None and tiff_temp.exists():
            print(f"[TIFF] Removing temporary directory: {tiff_temp}")
            shutil.rmtree(tiff_temp, ignore_errors=True)

if __name__ == "__main__":
    main()