import os
import torch
import numpy as np
import argparse
import mrcfile
import time
import random
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Import from existing libraries
from models.se_transformer_euler_sampling_optimised import SETransformerEulerSampling
from scripts.utils_ablation_fsc import (
    rotate_tensor, shift_tensor, calculate_resolution_fsc, 
    alignment_eval, plot_fsc_curve
)

# angstroms per pixel, dont change or results look wierd
PIXEL_SIZE = 2.276

# -------------------------------
# Utility and setup functions
# -------------------------------

def setup_logger(log_file='fsc_test.log'):
    """Setup logger with file and console handlers for detailed logging."""
    logger = logging.getLogger('fsc_pipeline')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def normalize_data(data, logger=None):
    """Normalize data with safeguards against division by zero or very small values."""
    if logger:
        logger.info(f"Normalizing data with shape {data.shape}")
        logger.info(f"Before normalization: min={data.min().item():.4f}, max={data.max().item():.4f}, "
                   f"mean={data.mean().item():.4f}, std={data.std().item():.4f}")
    
    mean = data.mean()
    std = data.std()
    
    # dont divide by zero or bad things happen, trust me
    if std < 1e-6:
        if logger:
            logger.warning(f"Very small std detected: {std.item():.8f}, using 1.0 instead")
        std = torch.tensor(1.0, device=data.device)
    
    normalized_data = (data - mean) / std
    
    if torch.isnan(normalized_data).any():
        if logger:
            logger.warning("NaNs detected in normalized data, fixing...")
        normalized_data = torch.nan_to_num(normalized_data)
    
    if logger:
        logger.info(f"After normalization: min={normalized_data.min().item():.4f}, max={normalized_data.max().item():.4f}, "
                   f"mean={normalized_data.mean().item():.4f}, std={normalized_data.std().item():.4f}")
    
    return normalized_data, mean, std

# -------------------------------
# Data loading and processing
# -------------------------------

def load_data(mrc_file, coords_file, subtomo_size=32, logger=None):
    """Load data with detailed logging and error checking."""
    if logger:
        logger.info(f"Loading data from {mrc_file} using coordinates from {coords_file}")
    
    start_time = time.time()
    
    try:
        # Load coordinates
        coords = []
        with open(coords_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    if ',' in line:
                        values = [float(x.strip()) for x in line.split(',')]
                    else:
                        values = [float(x) for x in line.split()]
                    coords.append(values)
                except ValueError:
                    continue
        
        coords = np.array(coords).astype(int)  
        
        if logger:
            logger.info(f"Loaded {len(coords)} coordinates")
        
        # Process in batches to manage memory
        batch_size = 50
        num_batches = (len(coords) + batch_size - 1) // batch_size
        subtomograms = []
        
        # Try to open MRC file - somtimes these files be hating on us
        try:
            mrc = mrcfile.mmap(mrc_file, mode='r', permissive=True)
            if mrc.data is None:
                raise ValueError("Memory mapping failed, data is None")
        except Exception as e:
            if logger:
                logger.warning(f"Memory mapping failed: {str(e)}, trying direct read...")
            try:
                mrc = mrcfile.open(mrc_file, mode='r', permissive=True)
                if mrc.data is None:
                    raise ValueError("Direct read failed, data is None")
            except Exception as e2:
                raise IOError(f"Failed to open MRC file: {str(e2)}")
        
        # Get volume shape
        volume_shape = mrc.data.shape
        
        if logger:
            logger.info(f"MRC file shape: {volume_shape}, processing in {num_batches} batches")
        
        # Prepare to extract subvolumes
        half_size = subtomo_size // 2
        
        # Process coordinates in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            
            if logger and batch_idx % 5 == 0:
                logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx+1}-{end_idx}/{len(coords)})")
            
            # Extract subvolumes for this batch
            for coord_idx, coord in enumerate(batch_coords):
                try:
                    x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                    
                    # Calculate boundaries with clamping
                    z_start = max(0, z - half_size)
                    z_end = min(volume_shape[0], z + half_size)
                    y_start = max(0, y - half_size)
                    y_end = min(volume_shape[1], y + half_size)
                    x_start = max(0, x - half_size)
                    x_end = min(volume_shape[2], x + half_size)
                    
                    # Initialize subvolume
                    subvol = np.zeros((subtomo_size, subtomo_size, subtomo_size), dtype=np.float32)
                    
                    # Calculate extraction dimensions
                    z_extract = z_end - z_start
                    y_extract = y_end - y_start
                    x_extract = x_end - x_start
                    
                    # Calculate target positions
                    z_target_start = half_size - (z - z_start)
                    y_target_start = half_size - (y - y_start)
                    x_target_start = half_size - (x - x_start)
                    
                    # Extract data
                    subvol[
                        z_target_start:z_target_start+z_extract,
                        y_target_start:y_target_start+y_extract,
                        x_target_start:x_target_start+x_extract
                    ] = mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Check subvolume statistics for early problem detection
                    vol_mean = np.mean(subvol)
                    vol_std = np.std(subvol)
                    vol_min = np.min(subvol)
                    vol_max = np.max(subvol)
                    
                    # Check for NaN values - those guys are bad news
                    if np.isnan(subvol).any() or np.isinf(subvol).any():
                        if logger:
                            logger.warning(f"NaN/Inf values detected in subvolume at coordinate {coord}")
                        subvol = np.nan_to_num(subvol, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Check for extreme values
                    if np.abs(subvol).max() > 1e6:
                        if logger:
                            logger.warning(f"Extreme values detected in subvolume at coordinate {coord}, clamping")
                        subvol = np.clip(subvol, -1e6, 1e6)
                    
                    # Add to result list
                    subtomograms.append(subvol)
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error extracting subtomogram at coordinate {coord}: {str(e)}")
                    # Create an empty subvolume as substitute
                    subvol = np.zeros((subtomo_size, subtomo_size, subtomo_size), dtype=np.float32)
                    subtomograms.append(subvol)
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        # Close MRC file
        if mrc is not None:
            mrc.close()
            
        # Convert to torch tensor and add channel dimension
        if subtomograms:
            subtomograms_tensor = torch.tensor(np.stack(subtomograms, axis=0), dtype=torch.float32).unsqueeze(1)
            
            # Final check for NaN/Inf values in tensor
            has_nan = torch.isnan(subtomograms_tensor).any().item()
            has_inf = torch.isinf(subtomograms_tensor).any().item()
            
            if has_nan or has_inf:
                if logger:
                    logger.warning(f"Found NaN or Inf values in final tensor: NaN={has_nan}, Inf={has_inf}")
                # Replace NaN/Inf with zeros
                subtomograms_tensor = torch.nan_to_num(subtomograms_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for extreme values
            max_abs_val = subtomograms_tensor.abs().max().item()
            if max_abs_val > 1e6:
                if logger:
                    logger.warning(f"Extreme values detected in final tensor (max abs: {max_abs_val}), clamping")
                subtomograms_tensor = torch.clamp(subtomograms_tensor, -1e6, 1e6)
            
            elapsed_time = time.time() - start_time
            if logger:
                logger.info(f"Extracted {len(subtomograms)} subtomograms with shape {subtomograms_tensor.shape} in {elapsed_time:.2f} seconds")
                logger.info(f"Tensor stats: min={subtomograms_tensor.min().item():.4f}, max={subtomograms_tensor.max().item():.4f}, "
                           f"mean={subtomograms_tensor.mean().item():.4f}, std={subtomograms_tensor.std().item():.4f}")
            
            return subtomograms_tensor
        else:
            raise ValueError(f"No valid subtomograms extracted from {mrc_file}")
            
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        if logger:
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
        
        # Return a small dummy tensor rather than failing completely
        if logger:
            logger.warning("Returning dummy tensor due to data loading error")
        return torch.zeros((10, 1, subtomo_size, subtomo_size, subtomo_size), dtype=torch.float32)

# -------------------------------
# Transformation functions
# -------------------------------

def apply_test_transformations(x, device, max_rotation_angle, translation_range):
    """Apply rotation and translation transformations to input volumes."""
    # Move input to the correct device first
    x = x.to(device)
    B, C, D, H, W = x.shape

    # generate random angles in degrees, then convert to radians cuz math
    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * np.pi / 180.0

    # Apply rotations
    x_rot = x.clone()
    x_rot = rotate_tensor(x_rot, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    # Apply translations
    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)
    x_translated = shift_tensor(x_rot, translations, device=device)

    transformed_input = x_translated
    targets = x

    params = torch.cat([angles_rad, translations], dim=1)

    return transformed_input, targets, params

def create_test_dataset(subtomograms, max_rotation_angle, translation_range, device='cpu', logger=None):
    """Create self-supervised test pairs from subtomograms."""
    if logger:
        logger.info(f"Creating test pairs for {subtomograms.shape[0]} subtomograms...")
        logger.info(f"Input stats: min={subtomograms.min().item():.4f}, max={subtomograms.max().item():.4f}, "
                   f"mean={subtomograms.mean().item():.4f}, std={subtomograms.std().item():.4f}")
    
    # Handle NaNs and extreme values before processing
    if torch.isnan(subtomograms).any() or torch.isinf(subtomograms).any():
        if logger:
            logger.warning("NaN/Inf values found in input, fixing...")
        subtomograms = torch.nan_to_num(subtomograms, nan=0.0, posinf=0.0, neginf=0.0)
    
    if subtomograms.abs().max() > 1e6:
        if logger:
            logger.warning("Extreme values detected, clamping to reasonable range")
        subtomograms = subtomograms.clamp(-1e6, 1e6)
    
    # Process in smaller batches
    batch_size = 50
    num_batches = (subtomograms.shape[0] + batch_size - 1) // batch_size
    
    transformed_inputs_list = []
    targets_list = []
    transformation_params_list = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, subtomograms.shape[0])
        
        batch_subtomograms = subtomograms[start_idx:end_idx]
        
        if logger and batch_idx % 5 == 0:
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx+1}-{end_idx}/{subtomograms.shape[0]})")
        
        try:
            # Keep tensors on CPU until transformation to avoid unnecessary transfers
            transformed_inputs, targets, transformation_params = apply_test_transformations(
                batch_subtomograms, 
                device=device,
                max_rotation_angle=max_rotation_angle, 
                translation_range=translation_range
            )
            
            # Move results back to CPU for storage
            transformed_inputs_list.append(transformed_inputs.cpu())
            targets_list.append(targets.cpu())
            transformation_params_list.append(transformation_params.cpu())
            
        except Exception as e:
            if logger:
                logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
            # Create empty tensors as placeholders
            B = batch_subtomograms.shape[0]
            C, D, H, W = 1, 32, 32, 32
            dummy_input = torch.zeros((B, C, D, H, W))
            dummy_params = torch.zeros((B, 6))
            
            transformed_inputs_list.append(dummy_input)
            targets_list.append(dummy_input.clone())
            transformation_params_list.append(dummy_params)
    
    # Combine the results
    transformed_inputs = torch.cat(transformed_inputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    transformation_params = torch.cat(transformation_params_list, dim=0)
    
    if logger:
        logger.info(f"Test pairs created. Shapes: inputs={transformed_inputs.shape}, targets={targets.shape}")
        logger.info(f"Output stats: min={transformed_inputs.min().item():.4f}, max={transformed_inputs.max().item():.4f}, "
                   f"mean={transformed_inputs.mean().item():.4f}, std={transformed_inputs.std().item():.4f}")
    
    return transformed_inputs, targets, transformation_params

# -------------------------------
# Model testing functions
# -------------------------------

def test_model(model, test_loader, device, trans_mean, trans_std, logger=None):
    """Run model inference with detailed error checking and diagnostics."""
    model.eval()
    predicted_trans = []
    ground_truth_trans = []
    aligned_volumes = []
    target_volumes = []
    
    trans_mean = trans_mean.to(device)
    trans_std = trans_std.to(device)
    
    debug_batch_limit = 10  # Number of batches to show detailed debug info
    zero_volume_count = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets, params) in enumerate(test_loader):
                try:
                    # Sanitize inputs
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                    params = torch.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0).to(device)
                    
                    # Log sample values for first few batches
                    if batch_idx < debug_batch_limit:
                        if logger:
                            logger.info(f"Batch {batch_idx} input: shape={inputs.shape}, min={inputs.min().item():.4f}, "
                                       f"max={inputs.max().item():.4f}, sum={inputs.abs().sum().item():.4f}")
                            logger.info(f"First few values: {inputs[0, 0, 0, 0, :5].tolist()}")
                    
                    # Forward pass with appropriate error handling
                    try:
                        transform_preds, aligned_input, aligned_steps = model(input=inputs, target=targets)
                    except Exception as model_error:
                        if logger:
                            logger.error(f"Error in model forward pass for batch {batch_idx}: {str(model_error)}")
                            import traceback
                            logger.error(traceback.format_exc())
                        continue
                    
                    # Verify model outputs
                    if not transform_preds:
                        if logger:
                            logger.warning(f"Empty transform_preds for batch {batch_idx}")
                        continue
                    
                    # Debug aligned volume
                    vol_sum = aligned_input.abs().sum().item()
                    if batch_idx < debug_batch_limit:
                        if logger:
                            logger.info(f"Batch {batch_idx} aligned: min={aligned_input.min().item():.4f}, "
                                      f"max={aligned_input.max().item():.4f}, sum={vol_sum:.4f}")
                    
                    # Process transformation predictions
                    transformation_pred = transform_preds[-1]
                    if torch.isnan(transformation_pred).any() or torch.isinf(transformation_pred).any():
                        if logger:
                            logger.warning(f"NaN/Inf in predictions for batch {batch_idx}, fixing...")
                        transformation_pred = torch.nan_to_num(transformation_pred, nan=0.0)
                    
                    gt_transformation = params[:, :6]
                    
                    # Unnormalize transformations
                    transformation_pred_unnorm = transformation_pred * trans_std + trans_mean
                    gt_transformation_unnorm = gt_transformation * trans_std + trans_mean
                    
                    predicted_trans.append(transformation_pred_unnorm.cpu())
                    ground_truth_trans.append(gt_transformation_unnorm.cpu())
                    
                    # Handle zero volumes - try to recover from intermediate steps
                    if vol_sum < 1e-6:
                        zero_volume_count += 1
                        if aligned_steps and len(aligned_steps) > 0:
                            # Try the last intermediate step
                            last_step = aligned_steps[-1].clone().detach()
                            last_step_sum = last_step.abs().sum().item()
                            
                            if last_step_sum >= 1e-6:
                                if logger and batch_idx < debug_batch_limit:
                                    logger.info(f"Batch {batch_idx}: Using last aligned step instead of zero volume")
                                aligned_volume_copy = last_step.cpu()
                            else:
                                if logger and batch_idx < debug_batch_limit:
                                    logger.warning(f"Batch {batch_idx}: All steps are zero volumes")
                                aligned_volume_copy = aligned_input.clone().detach().cpu()
                        else:
                            aligned_volume_copy = aligned_input.clone().detach().cpu()
                    else:
                        aligned_volume_copy = aligned_input.clone().detach().cpu()
                    
                    target_volume_copy = targets.clone().detach().cpu()
                    
                    # One more safety check
                    aligned_volume_copy = torch.nan_to_num(aligned_volume_copy, nan=0.0)
                    target_volume_copy = torch.nan_to_num(target_volume_copy, nan=0.0)
                    
                    aligned_volumes.append(aligned_volume_copy)
                    target_volumes.append(target_volume_copy)
                    
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        if logger:
                            logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
                
                except Exception as batch_error:
                    if logger:
                        logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    continue
            
            if logger:
                logger.info(f"Completed testing: {zero_volume_count}/{len(test_loader)} batches produced zero volumes")
    
    except Exception as e:
        if logger:
            logger.error(f"Error in test_model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return predicted_trans, ground_truth_trans, aligned_volumes, target_volumes

# -------------------------------
# Evaluation functions
# -------------------------------

def evaluate_half_map_fsc(all_aligned: torch.Tensor,
                          pixel_size: float = PIXEL_SIZE,
                          fsc_threshold: float = 0.143,
                          output_dir: str = None,
                          logger: logging.Logger = None):
    """
    Evaluate half-map FSC (using hardcoded or passed pixel_size).

    Parameters:
      all_aligned    Tensor, shape [N, C, Z, Y, X], C=1
      pixel_size     Voxel size (Å/voxel), defaults to hardcoded value
      fsc_threshold  FSC threshold, default 0.143 (half-bit)
      output_dir     Directory to save curve plots, if None, don't save
      logger         logging.Logger instance, can be None

    Returns:
      resolution (float), fsc_curve (np.ndarray)
    """
    if logger:
        logger.info(f"[FSC] Using pixel_size = {pixel_size:.3f} Å/voxel, threshold = {fsc_threshold}")

    # 1. Exclude volumes that are almost all zeros
    flat = all_aligned.abs().view(all_aligned.size(0), -1).sum(dim=1)
    valid_mask = (flat >= 1e-6)
    valid_vols = all_aligned[valid_mask]
    n = valid_vols.size(0)
    if logger:
        logger.info(f"[Half-map FSC] Valid volumes {n}/{all_aligned.size(0)}")
    if n < 2:
        if logger:
            logger.error("[Half-map FSC] Not enough valid volumes, skipping FSC calculation")
        return None, None

    # 2. Randomly divide in half and average
    perm = torch.randperm(n, device=valid_vols.device)
    half = n // 2
    idx1, idx2 = perm[:half], perm[half:half*2]
    vol1 = valid_vols[idx1].mean(dim=0).squeeze()  # [Z,Y,X]
    vol2 = valid_vols[idx2].mean(dim=0).squeeze()

    # 3. 3D soft spherical mask (cosine edge smoothing)
    Z, Y, X = vol1.shape
    cz, cy, cx = Z/2, Y/2, X/2
    zs = torch.arange(Z, device=vol1.device).view(-1,1,1).float() - cz
    ys = torch.arange(Y, device=vol1.device).view(1,-1,1).float() - cy
    xs = torch.arange(X, device=vol1.device).view(1,1,-1).float() - cx
    dist = torch.sqrt(zs**2 + ys**2 + xs**2)
    radius = min(cz, cy, cx)
    edge_width = radius * 0.1
    mask = torch.ones_like(dist)
    mask[dist > radius] = 0
    band = (dist > (radius - edge_width)) & (dist <= radius)
    mask[band] = 0.5 * (1 + torch.cos(
        np.pi * (dist[band] - (radius - edge_width)) / edge_width
    ))
    vol1_masked = vol1 * mask
    vol2_masked = vol2 * mask

    # 4. Call calculate_resolution_fsc (remove unnecessary parameters)
    res, curve = calculate_resolution_fsc(
        vol1_masked,
        vol2_masked,
        pixel_size=pixel_size,
        threshold=fsc_threshold,
        device=vol1_masked.device
    )

    # 5. Visualize and save curve
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        plt.plot(curve, label='FSC')
        plt.axhline(y=fsc_threshold, linestyle='--', label=f"thr={fsc_threshold}")
        plt.xlabel('Frequency shell index')
        plt.ylabel('FSC')
        plt.title('Half-map FSC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'half_map_fsc_curve.png'))
        plt.close()

    if logger:
        logger.info(f"[Half-map FSC] Resolution: {res:.2f} Å")
    return res, curve

def evaluate_fsc(aligned_volumes, target_volumes, image_size=32, pixel_size=2.0, 
                fsc_threshold=0.5, output_dir=None, name="test", logger=None):
    """Calculate FSC between aligned volumes and targets."""
    if logger:
        logger.info(f"Starting FSC evaluation for {name}...")
    
    if not aligned_volumes:
        if logger:
            logger.error("No aligned volumes to evaluate")
        return None
    
    try:
        # First, check if all volumes are zero
        zero_count = sum(1 for vol in aligned_volumes if vol.abs().sum() < 1e-6)
        if logger:
            logger.info(f"Zero volumes check: {zero_count}/{len(aligned_volumes)} are zero")
        
        if zero_count == len(aligned_volumes):
            if logger:
                logger.error("All volumes are zero, FSC calculation impossible")
            return None
        
        # Create properly formatted list of volumes (ensure batch size 1)
        valid_aligned = []
        valid_targets = []
        
        for i, (aligned_vol, target_vol) in enumerate(zip(aligned_volumes, target_volumes)):
            # Skip zero volumes
            if aligned_vol.abs().sum() < 1e-6:
                continue
                
            # Handle multi-batch volumes by splitting them
            if aligned_vol.dim() == 5 and aligned_vol.size(0) > 1:  # Has batch dimension > 1
                for j in range(aligned_vol.size(0)):
                    valid_aligned.append(aligned_vol[j:j+1])
                    valid_targets.append(target_vol[j:j+1] if target_vol.size(0) > 1 else target_vol)
            else:
                # Ensure batch dimension
                if aligned_vol.dim() == 4:  # No batch dimension
                    aligned_vol = aligned_vol.unsqueeze(0)
                if target_vol.dim() == 4:  # No batch dimension
                    target_vol = target_vol.unsqueeze(0)
                valid_aligned.append(aligned_vol)
                valid_targets.append(target_vol)
        
        # Check if we have any valid volumes
        if not valid_aligned:
            if logger:
                logger.error("No valid volumes after filtering")
            return None
        
        if logger:
            logger.info(f"Valid volumes for averaging: {len(valid_aligned)}/{len(aligned_volumes)}")
        
        # Concatenate instead of stack to avoid dimension mismatch
        all_aligned = torch.cat(valid_aligned, dim=0)
        all_targets = torch.cat(valid_targets, dim=0)
        
        # Log volume statistics
        if logger:
            logger.info(f"Volume statistics:")
            logger.info(f"  Aligned: shape={all_aligned.shape}, min={all_aligned.min().item():.4f}, "
                       f"max={all_aligned.max().item():.4f}, mean={all_aligned.mean().item():.4f}")
            logger.info(f"  Targets: shape={all_targets.shape}, min={all_targets.min().item():.4f}, "
                       f"max={all_targets.max().item():.4f}, mean={all_targets.mean().item():.4f}")
        
        # Calculate average volumes
        avg_aligned = all_aligned.mean(dim=0, keepdim=True)
        avg_targets = all_targets.mean(dim=0, keepdim=True)
        
        # Save visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mid_slice = avg_aligned.shape[2] // 2
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(avg_aligned[0, 0, mid_slice].numpy(), cmap='gray')
            plt.title("Average Aligned Volume")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(avg_targets[0, 0, mid_slice].numpy(), cmap='gray')
            plt.title("Average Target Volume")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f"{name}_average_slices.png"))
            plt.close()
        
        # Calculate FSC
        try:
            resolution, fsc_curve = evaluate_half_map_fsc(
                all_aligned=all_aligned,
                pixel_size=PIXEL_SIZE,
                fsc_threshold=0.143,
                output_dir=output_dir,
                logger=logger
            )
            
            if logger:
                logger.info(f"FSC Resolution: {resolution:.2f} Å")
            
            # Plot and save results
            if output_dir:
                curve_file = os.path.join(output_dir, f"{name}_fsc_curve.png")
                plot_fsc_curve(fsc_curve, threshold=fsc_threshold, filename=curve_file)
                
                summary_file = os.path.join(output_dir, f"{name}_fsc_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"FSC Evaluation Summary for {name}\n")
                    f.write(f"Pixel Size: {pixel_size} Å\n")
                    f.write(f"FSC Threshold: {fsc_threshold}\n")
                    f.write(f"Resolution: {resolution:.2f} Å\n")
                    f.write(f"Valid volumes: {len(valid_aligned)}/{len(aligned_volumes)}\n")
            
            return {
                'resolution': resolution,
                'fsc_curve': fsc_curve,
                'valid_count': len(valid_aligned),
                'total_count': len(aligned_volumes)
            }
            
        except Exception as fsc_error:
            if logger:
                logger.error(f"Error in FSC calculation: {str(fsc_error)}")
                import traceback
                logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        if logger:
            logger.error(f"Error in FSC evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        return None

# -------------------------------
# Main function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description='FSC evaluation pipeline')
    parser.add_argument('--model_path', type=str, default='./results_real_data/best_finetune.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--data_dir', type=str, default='/shared/u/c_mru/EMPIAR-10045/voxel_spacing_2A',
                        help='Directory containing MRC files and coordinates')
    parser.add_argument('--output_dir', type=str, default='./fsc_evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--subtomo_size', type=int, default=32,
                        help='Size of subtomograms')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for testing')
    parser.add_argument('--rotation_range', type=float, default=15.0,
                        help='Maximum rotation angle in degrees for test transformations')
    parser.add_argument('--translation_range', type=float, default=0.2,
                        help='Maximum translation as a fraction of volume dimensions')
    parser.add_argument('--pixel_size', type=float, default=2.0,
                        help='Pixel size in Angstroms for FSC calculation')
    parser.add_argument('--fsc_threshold', type=float, default=0.143,
                        help='FSC threshold for resolution (0.5 or 0.143)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Setup 
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'fsc_test.log')
    logger = setup_logger(log_file)
    logger.info(f"Starting FSC evaluation with arguments: {args}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    random_seed = 42  # answer to life, universe and everything
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # Test file
    test_file = os.path.join(args.data_dir, 'IS002_291013_010.mrc')
    test_coords = os.path.join(args.data_dir, 'IS002_291013_010.coords')
    
    try:
        logger.info("Step 1: Loading data")
        # Load subtomograms
        subtomograms = load_data(test_file, test_coords, args.subtomo_size, logger)
        
        logger.info("Step 2: Creating test dataset")
        # Create test dataset
        transformed_inputs, targets, params = create_test_dataset(
            subtomograms, 
            max_rotation_angle=args.rotation_range,
            translation_range=args.translation_range,
            device=device,
            logger=logger
        )
        
        logger.info("Step 3: Normalizing data")
        # Normalize data
        norm_inputs, mean_input, std_input = normalize_data(transformed_inputs, logger)
        norm_targets, _, _ = normalize_data(targets, logger)
        
        # Normalize transformation parameters
        params_mean = params.mean(dim=0, keepdim=True)
        params_std = params.std(dim=0, keepdim=True)
        norm_params = (params - params_mean) / params_std
        
        logger.info("Step 4: Creating dataset and dataloader")
        # Create dataset and dataloader
        test_dataset = TensorDataset(norm_inputs, norm_targets, norm_params)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info("Step 5: Loading model")
        # Create model
        model = SETransformerEulerSampling(
            in_channels=1,
            num_transformer_blocks=4,
            num_heads=8,
            ff_hidden_dim=512,
            hidden_dim=120,
            feature_type='vector',
            patch_size=(4, 4, 4)
        )
        
        # Load weights
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded model weights")
        else:
            logger.error("Invalid checkpoint format - missing model_state_dict")
            return
        
        model = model.to(device)
        model.eval()
        
        logger.info("Step 6: Running model inference")
        # Run inference
        predicted_trans, ground_truth_trans, aligned_volumes, target_volumes = test_model(
            model, 
            test_loader, 
            device, 
            params_mean.to(device), 
            params_std.to(device),
            logger
        )
        
        logger.info("Step 7: Calculating alignment metrics")
        # Calculate alignment metrics
        concat_pred = torch.cat(predicted_trans, dim=0)
        concat_gt = torch.cat(ground_truth_trans, dim=0)
        
        logger.info(f"Predictions shape: {concat_pred.shape}, GT shape: {concat_gt.shape}")
        logger.info(f"Predictions: min={concat_pred.min().item():.4f}, max={concat_pred.max().item():.4f}")
        logger.info(f"Ground truth: min={concat_gt.min().item():.4f}, max={concat_gt.max().item():.4f}")
        
        alignment_eval(
            concat_gt, 
            concat_pred, 
            image_size=args.subtomo_size, 
            scale=False, 
            transform_type='euler'
        )
        
        logger.info("Step 8: Evaluating FSC")
        # Calculate FSC
        fsc_result = evaluate_fsc(
            aligned_volumes,
            target_volumes,
            image_size=args.subtomo_size,
            pixel_size=args.pixel_size,
            fsc_threshold=args.fsc_threshold,
            output_dir=args.output_dir,
            name=os.path.basename(test_file).split('.')[0],
            logger=logger
        )
        
        if fsc_result:
            resolution = fsc_result['resolution']
            valid_count = fsc_result['valid_count']
            total_count = fsc_result['total_count']
            
            logger.info(f"FSC Evaluation completed successfully:")
            logger.info(f"  Resolution: {resolution:.2f} Å")
            logger.info(f"  Valid volumes: {valid_count}/{total_count}")
        else:
            logger.warning("FSC evaluation failed or returned no results")
        
        logger.info("FSC evaluation pipeline completed")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()