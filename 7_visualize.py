import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SegMamba segmentation results")
    parser.add_argument('--test_data_dir', type=str, required=True, 
                        help='Directory or file pattern containing original test NIfTI images')
    parser.add_argument('--pred_dir', type=str, default='./prediction_results/segmamba',
                        help='Directory containing predicted segmentation masks')
    parser.add_argument('--case_name', type=str, default='case_0',
                        help='Case name to visualize (e.g., case_0)')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Slice index to visualize (if None, uses middle slice)')
    parser.add_argument('--modality_idx', type=int, default=0,
                        help='Modality index to visualize (0-3 for T1, T1ce, T2, FLAIR)')
    return parser.parse_args()

def load_nifti(file_path):
    """Load a NIfTI file and return the data as a NumPy array."""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data

def visualize_slice(image, masks, slice_idx, modality_idx, case_name):
    """Visualize a slice of the image and its segmentation masks."""
    # Ensure image and masks are NumPy arrays
    image = np.asarray(image)
    masks = np.asarray(masks)

    # Get the middle slice if slice_idx is not specified
    if slice_idx is None:
        slice_idx = image.shape[-1] // 2

    # Extract the specified slice and modality
    img_slice = image[modality_idx, :, :, slice_idx]  # Shape: [H, W]
    mask_tc = masks[0, :, :, slice_idx]  # Tumor Core
    mask_wt = masks[1, :, :, slice_idx]  # Whole Tumor
    mask_et = masks[2, :, :, slice_idx]  # Enhancing Tumor

    # Normalize image for visualization
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot the original image slice
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'Image (Modality {modality_idx}) - Slice {slice_idx}')
    axes[0].axis('off')

    # Plot the Tumor Core (TC) mask
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(mask_tc, cmap='Reds', alpha=0.5)
    axes[1].set_title('Tumor Core (TC)')
    axes[1].axis('off')

    # Plot the Whole Tumor (WT) mask
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(mask_wt, cmap='Greens', alpha=0.5)
    axes[2].set_title('Whole Tumor (WT)')
    axes[2].axis('off')

    # Plot the Enhancing Tumor (ET) mask
    axes[3].imshow(img_slice, cmap='gray')
    axes[3].imshow(mask_et, cmap='Blues', alpha=0.5)
    axes[3].set_title('Enhancing Tumor (ET)')
    axes[3].axis('off')

    plt.suptitle(f'Segmentation Results for {case_name}')
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()

    # Load test images
    if os.path.isdir(args.test_data_dir):
        # Assume test_data_dir contains NIfTI files named like case_0.nii.gz
        test_file_pattern = os.path.join(args.test_data_dir, f"{args.case_name}.nii*")
        test_files = glob.glob(test_file_pattern)
        if not test_files:
            raise FileNotFoundError(f"No test image found for {args.case_name} in {args.test_data_dir}")
        test_image_path = test_files[0]
    else:
        # Assume test_data_dir is a file pattern
        test_files = glob.glob(args.test_data_dir)
        test_file = next((f for f in test_files if args.case_name in f), None)
        if test_file is None:
            raise FileNotFoundError(f"No test image found for {args.case_name} matching {args.test_data_dir}")
        test_image_path = test_file

    # Load predicted segmentation mask
    pred_file = os.path.join(args.pred_dir, f"{args.case_name}.nii.gz")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predicted mask not found at {pred_file}")

    # Load the data
    print(f"Loading test image from {test_image_path}")
    image = load_nifti(test_image_path)  # Shape: [C, H, W, D], C=4
    print(f"Loading predicted mask from {pred_file}")
    masks = load_nifti(pred_file)  # Shape: [C, H, W, D], C=3 (TC, WT, ET)

    # Visualize the specified slice
    visualize_slice(image, masks, args.slice_idx, args.modality_idx, args.case_name)

if __name__ == "__main__":
    main()