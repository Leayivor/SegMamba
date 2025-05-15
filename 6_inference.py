import numpy as np
import torch
import os
import argparse
from monai.inferers import SlidingWindowInferer
from light_training.prediction import Predictor
from light_training.dataloading.dataset import get_test_loader_from_test, MedicalDataset
import glob
from monai.utils import set_determinism

set_determinism(123)  # For reproducibility

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with SegMamba model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the model checkpoint. If not provided, will use the default path.')
    parser.add_argument('--output_dir', type=str, default='./prediction_results/segmamba',
                        help='Directory to save prediction results')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run inference on')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for sliding window inference')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 128, 128], 
                        help='Patch size for sliding window inference')
    parser.add_argument('--overlap', type=float, default=0.5, 
                        help='Overlap ratio for sliding window inference')
    parser.add_argument('--mirror_axes', nargs='+', type=int, default=[0, 1, 2], 
                        help='Axes for test time augmentation via mirroring')
    
    return parser.parse_args()

def define_model_segmamba(model_path, patch_size, batch_size, overlap, mirror_axes):
    """Define the SegMamba model and predictor"""
    from model_segmamba.segmamba import SegMamba
    
    model = SegMamba(in_chans=4,
                    out_chans=4,
                    depths=[2, 2, 2, 2],
                    feat_size=[48, 96, 192, 384])
    
    if model_path is None:
        # Use default path if none provided
        model_path = "./logs/segmamba/model/final_model.pt"
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        new_sd = filter_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model.eval()
    
    window_infer = SlidingWindowInferer(roi_size=patch_size,
                                      sw_batch_size=batch_size,
                                      overlap=overlap,
                                      progress=True,
                                      mode="gaussian")

    predictor = Predictor(window_infer=window_infer,
                        mirror_axes=mirror_axes)

    return model, predictor

def filter_state_dict(sd):
    """Filter state dict to remove module prefix if present"""
    if "module" in sd:
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_k = k[7:] if k.startswith("module") else k 
        new_sd[new_k] = v 
    del sd 
    return new_sd

def convert_labels(labels):
    """Convert labels to TC, WT and ET"""
    result = [(labels == 1) | (labels == 3), 
              (labels == 1) | (labels == 3) | (labels == 2), 
              labels == 3]
    
    return torch.cat(result, dim=1).float()

def convert_labels_dim0(labels):
    """Convert labels to TC, WT and ET (dimension 0)"""
    result = [(labels == 1) | (labels == 3), 
              (labels == 1) | (labels == 3) | (labels == 2), 
              labels == 3]
    
    return torch.cat(result, dim=0).float()

def run_inference(args):
    """Run inference on test data"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test data
    if os.path.isdir(args.data_dir):
        # If data_dir is a directory, use get_test_loader_from_test
        test_ds = get_test_loader_from_test(args.data_dir)
    else:
        # If data_dir is a file pattern, use glob and create dataset directly
        test_datalist = glob.glob(args.data_dir)
        if not test_datalist:
            raise ValueError(f"No files found matching pattern: {args.data_dir}")
        print(f"Found {len(test_datalist)} files for inference")
        test_ds = MedicalDataset(test_datalist, test=True)
    
    # Define model and predictor
    model, predictor = define_model_segmamba(
        args.model_path, 
        args.patch_size, 
        args.batch_size, 
        args.overlap,
        args.mirror_axes
    )
    
    # Run inference on each sample
    device = args.device
    print(f"Running inference on {len(test_ds)} samples using device: {device}")
    
    for i in range(len(test_ds)):
        print(f"Processing sample {i+1}/{len(test_ds)}")
        batch = test_ds[i]
        
        # Get input data
        image = batch["data"]
        properties = batch["properties"]
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image[None]
        
        # Convert to torch tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Run inference
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)
        
        # Post-process predictions
        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        
        # Get final segmentation by taking argmax
        model_output = model_output.argmax(dim=0)[None]
        model_output = convert_labels_dim0(model_output)
        
        # Restore to original image space
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        
        # Save prediction to NIfTI file
        case_name = properties.get('name', [f"case_{i}"])[0]
        predictor.save_to_nii(
            model_output, 
            raw_spacing=[1, 1, 1],
            case_name=case_name,
            save_dir=args.output_dir
        )
        
        print(f"Saved prediction for {case_name}")
    
    print(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
