"""
Usage
-----
    python data/processing_helper.py \\
      --input      ./fungi_raw.zip \\
      --output     ./fungi_preprocessed \\
      --method     optimized \\
      --image-size 384 \\
      --bit-depth  12 \\
      --split-json split_indices.json \\
      --basic-samples 100 \\
      --create-zip

Author: Rodrigo Sá
Date:   2025
"""

import sys
import json
import argparse
import shutil
import zipfile
import tempfile

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from preprocessing import (
    preprocess_dataset,
    fit_basicpy_from_paths,
    get_image_paths,
    zip_preprocessed_data,
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess microscopy images for deep learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input dataset directory or ZIP file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output preprocessed dataset directory'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['baseline', 'optimized'],
        default='optimized',
        help='Preprocessing method: baseline (normalization only) or optimized (+ BaSiCPy)'
    )    
    parser.add_argument(
        '--image-size', '-s',
        type=int,
        default=384,
        help='Target image size (square)'
    )
    parser.add_argument(
        '--bit-depth', '-b',
        type=int,
        choices=[8, 12, 16],
        default=12,
        help='Original image bit depth'
    )
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.15,
        help='Proportion of data for test set (0.0-1.0)'
    )
    parser.add_argument(
        '--seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--basic-samples', '-n',
        type=int,
        default=100,
        help='(100-200 recommended)'
    )
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=None,
        help='(default: auto-detect)'
    )
    parser.add_argument(
        '--create-zip',
        action='store_true',
        help='ZIP archive of preprocessed data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output exists'
    )
    parser.add_argument(
        "--split-json",
        type=str,
        required=True,
        help=(
            "Path to split_indices.json produced by canonical_splits.py. "
            "BaSiCPy (optimized method) is fitted exclusively on the training "
            "images identified by this file."
        ),
    )
    
    return parser.parse_args()


def validate_dataset_structure(dataset_dir: Path) -> Tuple[List[str], bool]:
    """
    Validate dataset has required class subdirectories.
    
    Returns:
        Tuple of (class_names, is_valid)
    """
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return [], False
    
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(class_dirs) < 2:
        print(f"ERROR: Expected at least 2 class subdirectories, found {len(class_dirs)}")
        return [], False
    
    class_names = []
    for class_dir in sorted(class_dirs):
        image_paths = get_image_paths(str(class_dir))
        if len(image_paths) == 0:
            print(f"WARNING: No images found in {class_dir.name}")
        else:
            class_names.append(class_dir.name)
    
    if len(class_names) < 2:
        print(f"ERROR: Need at least 2 classes with images")
        return [], False
    
    return class_names, True


def fit_basicpy_model(
    dataset_dir: Path,
    train_base_paths: List[Path],
    output_dir: Path,
    n_samples: int,
    bit_depth: int,
    random_seed: int
) -> object:
    """
    Args:
        dataset_dir: Path to raw dataset
        train_base_paths: Base paths for training images
        output_dir: Directory to save fitted model
        n_samples: Number of samples to use for fitting
        bit_depth: Original bit depth
        random_seed: Random seed
        
    Returns:
        Fitted BaSiCPy model
    """
    
    train_image_paths = [dataset_dir / p.with_suffix('.tif') for p in train_base_paths]
 
    existing_paths = [p for p in train_image_paths if p.exists()]
    if len(existing_paths) < len(train_image_paths):
        missing = len(train_image_paths) - len(existing_paths)
        print(f"  Warning: {missing} .tif files not found")
        
        train_image_paths = []
        for base_path in train_base_paths:
            found = False
            for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                full_path = dataset_dir / base_path.with_suffix(ext)
                if full_path.exists():
                    train_image_paths.append(full_path)
                    found = True
                    break
            if not found:
                print(f"    WARNING: Could not find image for {base_path}")
    
    if len(train_image_paths) < 10:
        raise RuntimeError(
            f"Only found {len(train_image_paths)} training images. "
            "Need at least 10 for stable BaSiCPy fitting."
        )
    
    print(f"  Found {len(train_image_paths)} training images")
    
    model_path = output_dir / "basicpy_model.npz"
    basicpy_model = fit_basicpy_from_paths(
        image_paths=train_image_paths,
        output_path=str(model_path),
        n_samples=min(n_samples, len(train_image_paths)),
        bit_depth=bit_depth,
        random_seed=random_seed
    )

    basic_data = np.load(model_path)
    bg_std_before = float(basic_data.get('bg_std_before', np.nan))
    bg_std_after = float(basic_data.get('bg_std_after', np.nan))
    bg_reduction_ratio = float(basic_data.get('bg_reduction_ratio', np.nan))
    n_samples_used = int(basic_data.get('n_samples', len(train_image_paths)))
    image_shape = tuple(basic_data.get('image_shape', (np.nan, np.nan)))
    gamma_prime = float(basic_data.get("gamma_prime", np.nan))
    mean_diff_before = float(basic_data.get("meandiff_before", np.nan))
    mean_diff_after = float(basic_data.get("meandiff_after", np.nan))

    if not np.isnan(bg_std_before) and not np.isnan(bg_std_after):
        print(f"BaSiC background std (empty-field): {bg_std_before:.6f} -> {bg_std_after:.6f}")
    if not np.isnan(bg_reduction_ratio):
        print(
            f"BaSiC background std reduction (empty-field): "
            f"{bg_reduction_ratio * 100:.1f}% (n_images_used={n_samples_used}, "
            f"image_shape={image_shape})"
        )
    if not np.isnan(gamma_prime):
        print(
            f"BaSiC correction score Γ'(I_corr): {gamma_prime:.4f} "
            f"(mean diff before={mean_diff_before:.6f}, after={mean_diff_after:.6f})"
        )

    return basicpy_model

def preprocess_images(
    input_dir: Path,
    output_dir: Path,
    method: str,
    image_size: int,
    bit_depth: int,
    basicpy_model: Optional[object],
    num_workers: Optional[int],
    force: bool
) -> bool:
    """
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory
        method: 'baseline' or 'optimized'
        image_size: Target image size
        bit_depth: Original bit depth
        basicpy_model: Pre-fitted BaSiCPy model
        force: Force reprocessing
        
    Returns:
        True if successful
    """
    config = {
        'preprocessing': {
            'method': method,
            'basic_params': {'n_samples_for_fitting': 100}
        },
        'image_size': image_size,
        'bit_depth': bit_depth
    }
    
    success = preprocess_dataset(
        source_dir=str(input_dir),
        target_dir=str(output_dir),
        config=config,
        basic_model=basicpy_model,
        force_reprocess=force,
    )
    
    return success

def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    temp_extracted_dir: Optional[Path] = None

    if input_path.suffix.lower() == ".zip":
        print(f" Input is a ZIP archive: {input_path}")
        temp_extracted_dir = Path(
            tempfile.mkdtemp(prefix="raw_dataset_")
        )
        print(f" Extracting ZIP to directory: {temp_extracted_dir}")

        with zipfile.ZipFile(str(input_path), "r") as zip_ref:
            zip_ref.extractall(str(temp_extracted_dir))

        candidates = [d for d in temp_extracted_dir.iterdir() if d.is_dir()]
        if len(candidates) == 1:
            dataset_dir = candidates[0]
            print(f" Using single top-level directory as dataset root: {dataset_dir}")
        else:
            dataset_dir = temp_extracted_dir
            print(f" Using extraction root as dataset root: {dataset_dir}")
    else:
        dataset_dir = input_path

    class_names, is_valid = validate_dataset_structure(dataset_dir)
    if not is_valid:
        if temp_extracted_dir is not None:
            shutil.rmtree(temp_extracted_dir, ignore_errors=True)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    split_path = Path(args.split_json)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split indices file not found: {split_path}\n"
            "Run data/canonical_splits.py first to generate it."
        )
    with open(split_path, "r") as f:
        split_data = json.load(f)
    train_base_paths = [Path(p) for p in split_data["train_base_paths"]]
    test_base_paths  = [Path(p) for p in split_data["test_base_paths"]]
    print(f" Train images: {len(train_base_paths)}")
    print(f" Test images:  {len(test_base_paths)}")

    basicpy_model = None
    if args.method == "optimized":
        basicpy_model = fit_basicpy_model(
            dataset_dir=dataset_dir,
            train_base_paths=train_base_paths,
            output_dir=output_path,
            n_samples=args.basic_samples,
            bit_depth=args.bit_depth,
            random_seed=args.seed,
        )

    preprocessed_dir = output_path / "preprocessed"
    success = preprocess_images(
        input_dir=dataset_dir,
        output_dir=preprocessed_dir,
        method=args.method,
        image_size=args.image_size,
        bit_depth=args.bit_depth,
        basicpy_model=basicpy_model,
        num_workers=args.num_workers,
        force=args.force,
    )

    if not success:
        print("\nERROR: Preprocessing failed!")
        if temp_extracted_dir is not None:
            shutil.rmtree(temp_extracted_dir, ignore_errors=True)
        sys.exit(1)
    
    metadata = {
        "method": args.method,      
        "bit_depth": args.bit_depth,
        "image_size": args.image_size,
        "n_channels": 1,
        "basic_samples": args.basic_samples,
        "baSiC_used": args.method == "optimized",
    }

    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_path}")


    if args.create_zip:
        zip_path = output_path.parent / f"{output_path.name}.zip"
        success = zip_preprocessed_data(
            source_dir=str(preprocessed_dir),
            zip_path=str(zip_path),
        )
        if success:
            print(f"Archive created: {zip_path}")
            shutil.rmtree(preprocessed_dir, ignore_errors=True)

    if temp_extracted_dir is not None:
        print(f"Removing temporary extracted dataset: {temp_extracted_dir}")
        shutil.rmtree(temp_extracted_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
