# Train revised model

import os
import random
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
import traceback
import numpy.fft as NF
from models.se_transformer_euler_sampling_optimised import SETransformerEulerSampling
import mrcfile
from datetime import timedelta
import wandb
import time
from torch.utils.data._utils.collate import default_collate
######## MultiProc ##########
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
######## MultiProc ##########


from scripts.utils_ablation_fsc import (
    rotate_tensor, shift_tensor,
    transformation_loss, cross_correlation_loss,
    alignment_eval, gradient_difference_loss,
    shear_tensor, fsc, calculate_resolution_fsc, 
    evaluate_alignment_with_fsc, plot_fsc_curve
)

import argparse
if dist.is_initialized() and dist.get_rank() != 0:
    os.environ['WANDB_MODE'] = 'disabled'
    
# ===========================
# Hyperparameters and Settings
# ===========================
def debug_print(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)
        
parser = argparse.ArgumentParser(description='Training script for SETransformer model.')

# Hyperparameters
parser.add_argument('--pretrain_batch_size', type=int, default=4, help='Batch size for pretraining.')
parser.add_argument('--finetune_batch_size', type=int, default=4, help='Batch size for fine-tuning.')
parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing.')
parser.add_argument('--patch_size', type=int, nargs=3, default=(4, 4, 4), help='Patch size for the model.')

parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs for pretraining.')
parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for fine-tuning.')

parser.add_argument('--learning_rate_pretrain', type=float, default=1e-5, help='Learning rate for pretraining.')
parser.add_argument('--weight_decay_pretrain', type=float, default=2e-8, help='Weight decay for pretraining.')
parser.add_argument('--learning_rate_finetune', type=float, default=1e-5, help='Learning rate for fine-tuning.')
parser.add_argument('--weight_decay_finetune', type=float, default=2e-8, help='Weight decay for fine-tuning.')
parser.add_argument('--best_model_path_pretrain', type=str, default='best_model_pretrain.pth', help='Path to save the best pre-trained model.')
parser.add_argument('--best_model_path_finetune', type=str, default='best_finetune.pth', help='Path to save the best fine-tuned model.')
parser.add_argument('--run_name', type=str, default='optimised', help='Name of the run for logging.')
parser.add_argument('--data_dir', type=str, default='./data', help='Base directory for dataset')
parser.add_argument('--output_dir', type=str, default='./logs_revised_2', help='Directory for saving outputs')
parser.add_argument('--pretrained_dir', type=str, default='./pretrained', help='Directory for pretrained models')

# Flags to determine architecture variants
parser.add_argument('--use_liere', action='store_true', help='Use LIERE/MARE in model architecture.')
parser.add_argument('--use_polyshift', action='store_true', help='Use polyphase anchoring in model architecture.')
parser.add_argument('--use_cross_attention', action='store_true', help='Use cross attention if true and self attention if false.')
parser.add_argument('--use_v2', action='store_true', help='Use MARE v2 in model architecture.')

# Model parameters
parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels.')
parser.add_argument('--num_transformer_blocks', type=int, default=4, help='Number of transformer blocks.')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
parser.add_argument('--ff_hidden_dim', type=int, default=256, help='Hidden dimension of the feedforward network.')
parser.add_argument('--hidden_dim', type=int, default=60, help='Hidden dimension of the transformer network.')

# Augmentation parameters
# TODO: change back to 10 and 10
parser.add_argument('--num_augmentations_pretrain', type=int, default=10, help='Number of augmentations for pre-training.')
parser.add_argument('--rotation_range_pretrain', type=float, default=90.0, help='Rotation range in degrees for pre-training.')
parser.add_argument('--translation_range_pretrain', type=float, default=0.3, help='Translation range as fraction of D, H, W for pre-training.')
parser.add_argument('--shearing_range_pretrain', type=float, default=0.05, help='Shearing range for pre-training.')
parser.add_argument('--num_augmentations_finetune', type=int, default=10, help='Number of augmentations for fine-tuning.')
parser.add_argument('--rotation_range_finetune', type=float, default=15.0, help='Rotation range in degrees for fine-tuning.')
parser.add_argument('--translation_range_finetune', type=float, default=0.2, help='Translation range as fraction of D, H, W for fine-tuning.')
parser.add_argument('--shearing_range_finetune', type=float, default=0.025, help='Shearing range for fine-tuning.')
parser.add_argument('--rotation_range_test', type=float, default=30.0, help='Rotation range in degrees for test set.')
parser.add_argument('--translation_range_test', type=float, default=0.3, help='Translation range as fraction of D, H, W for test set.')

parser.add_argument('--resume_pretrain', type=str, default=None, 
                   help='Path to resume pretraining from checkpoint.')
parser.add_argument('--resume_finetune', type=str, default=None, 
                   help='Path to resume finetuning from checkpoint.')
parser.add_argument('--force_train', action='store_true', help='Force training even if best model exists')

parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), 
                   help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=2, help='Number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')

parser.add_argument('--wandb_project', type=str, default='setransformer_r9', 
                   help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, 
                   help='WandB entity/team name')

parser.add_argument('--pixel_size', type=float, default=2.0, 
                   help='Pixel size in Angstroms for FSC calculation')
parser.add_argument('--fsc_threshold', type=float, default=0.5, 
                   help='FSC threshold for resolution estimation (0.5 or 0.143)')
parser.add_argument('--fsc_batch_size', type=int, default=10, 
                   help='Batch size for FSC evaluation grouping')
parser.add_argument('--subtomo_size', type=int, default=32, 
                   help='Size of subtomograms to extract from MRC files')
# parser.add_argument('--transform_type', type=str, choices=['r9', 'euler', 'r6'], default='r9',
#                     help='Which rotation representation to use: r9, euler, or r6.')
# parser.add_argument('--use_sampling', action='store_true',
#                         help='Use the sampling-based variant of SETransformer.')

args = parser.parse_args()

args.rotation_range_test = args.rotation_range_finetune
args.translation_range_test = args.translation_range_finetune

PRETRAIN_BATCH_SIZE = args.pretrain_batch_size
FINETUNE_BATCH_SIZE = args.finetune_batch_size
TEST_BATCH_SIZE = args.test_batch_size
PATCH_SIZE = tuple(args.patch_size)

PRETRAIN_EPOCHS = args.pretrain_epochs
FINETUNE_EPOCHS = args.finetune_epochs

LEARNING_RATE_PRETRAIN = args.learning_rate_pretrain
WEIGHT_DECAY_PRETRAIN = args.weight_decay_pretrain
LEARNING_RATE_FINETUNE = args.learning_rate_finetune
WEIGHT_DECAY_FINETUNE = args.weight_decay_finetune

BEST_MODEL_PATH_PRETRAIN = os.path.join(args.output_dir, args.best_model_path_pretrain)
BEST_MODEL_PATH_FINETUNE = os.path.join(args.output_dir, args.best_model_path_finetune)

RUN_NAME = args.run_name

USE_LIERE = args.use_liere
USE_POLYSHIFT = args.use_polyshift
USE_CROSS_ATTENTION = args.use_cross_attention
USE_V2 = args.use_v2

DEVICE = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PATIENCE_PRETRAIN = 3
EARLY_STOPPING_PATIENCE_FINETUNE = 3

if args.local_rank == -1:
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print(f'Not using distributed mode')
        return

    torch.cuda.set_device(args.local_rank)
    args.dist_url = 'env://'
    args.dist_backend = 'gloo'
    dist.init_process_group(backend=args.dist_backend, 
                          init_method=args.dist_url,
                          world_size=args.world_size, 
                          rank=args.rank)
    dist.barrier()

def init_wandb(args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config={
                    **vars(args),
                    'num_gpus': torch.cuda.device_count(),
                    'device_name': torch.cuda.get_device_name(args.local_rank),
                    'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
                }
            )
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
            print("Training will continue without wandb logging")

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
debug_print(f"Random seeds set to {SEED}")  

class DistributedMetricsTracker:
    def __init__(self, device):
        self.device = device
        self.reset()
        
    def reset(self):
        self.running_loss = torch.zeros(1, device=self.device)
        self.running_trans_loss = torch.zeros(1, device=self.device)
        self.count = torch.zeros(1, device=self.device)
        self.step = 0

    def update(self, loss, trans_loss, batch_size):
        if not torch.isfinite(loss):
            debug_print(f"Warning: Loss is {loss}, skipping update")
            return
        self.running_loss += loss.item() * batch_size
        self.running_trans_loss += trans_loss.item() * batch_size
        self.count += batch_size
        self.step += 1 
        
    def synchronize(self):
        try:
            if dist.is_initialized():
                dist.all_reduce(self.running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.running_trans_loss, op=dist.ReduceOp.SUM)
                
                if dist.get_rank() == 0 and self.step % 5 == 0:
                    metrics = {
                        'loss': (self.running_loss / self.count).item(),
                        'trans_loss': (self.running_trans_loss / self.count).item()
                    }
                    if wandb.run is not None:
                        wandb.log(metrics)
        except Exception as e:
            print(f"Rank {dist.get_rank()}: Error in synchronize: {e}")
            raise
        
    def get_metrics(self):
        return {
            'loss': (self.running_loss / self.count).item(),
            'trans_loss': (self.running_trans_loss / self.count).item()
        }
def cleanup():
    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
            if dist.get_rank() == 0:
                print("Distributed process group destroyed")
                wandb.finish()
        except Exception as e:
            print(f"Rank {dist.get_rank()}: Error during cleanup: {e}")
            dist.destroy_process_group()
            if dist.get_rank() == 0:
                print("Distributed process group destroyed")
                wandb.finish()
                
# ===========================
# Data Loading and Preparation
# ===========================
def load_data(mrc_file, coords_file, subtomo_size=32):
    '''Extract subtomograms from MRC file and coordinates file with progress reporting and error handling'''
    import mrcfile
    import numpy as np
    import torch
    import time
    import os
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Starting to load data from {mrc_file} using coordinates from {coords_file}")
    
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
        
        coords = np.array(coords).astype(int)  # Convert to integer coordinates
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loaded {len(coords)} coordinates from {coords_file}")
        
        # Define batch size to reduce memory usage
        batch_size = 50  # Process 50 coordinates at a time
        num_batches = (len(coords) + batch_size - 1) // batch_size
        subtomograms = []
        
        # Try different methods to open the MRC file
        mrc = None
        try:
            # First try memory mapping
            mrc = mrcfile.mmap(mrc_file, mode='r', permissive=True)
            if mrc.data is None:
                raise ValueError("Memory mapping failed, data is None")
        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Memory mapping failed: {str(e)}, trying direct read...")
            try:
                # Try direct read if memory mapping fails
                mrc = mrcfile.open(mrc_file, mode='r', permissive=True)
                if mrc.data is None:
                    raise ValueError("Direct read failed, data is None")
            except Exception as e2:
                raise IOError(f"Failed to open MRC file: {str(e2)}")
        
        # Get volume shape
        volume_shape = mrc.data.shape
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"MRC file shape: {volume_shape}, processing in {num_batches} batches")
        
        # Prepare to store extracted subvolumes
        half_size = subtomo_size // 2
        
        # Process coordinates in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and batch_idx % 5 == 0:
                print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx+1}-{end_idx}/{len(coords)})")
            
            # Iterate through coordinates in this batch, extract subvolumes
            for coord_idx, coord in enumerate(batch_coords):
                try:
                    x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                    
                    # Calculate boundaries
                    z_start = max(0, z - half_size)
                    z_end = min(volume_shape[0], z + half_size)
                    y_start = max(0, y - half_size)
                    y_end = min(volume_shape[1], y + half_size)
                    x_start = max(0, x - half_size)
                    x_end = min(volume_shape[2], x + half_size)
                    
                    # Extract subvolume
                    subvol = np.zeros((subtomo_size, subtomo_size, subtomo_size), dtype=np.float32)
                    
                    # Calculate actual extraction region and target region
                    z_extract = z_end - z_start
                    y_extract = y_end - y_start
                    x_extract = x_end - x_start
                    
                    # Determine position in target array
                    z_target_start = half_size - (z - z_start)
                    y_target_start = half_size - (y - y_start)
                    x_target_start = half_size - (x - x_start)
                    
                    # Copy data
                    subvol[
                        z_target_start:z_target_start+z_extract,
                        y_target_start:y_target_start+y_extract,
                        x_target_start:x_target_start+x_extract
                    ] = mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Add to result list
                    subtomograms.append(subvol)
                except Exception as e:
                    print(f"Error extracting subtomogram at coordinate {coord}: {str(e)}")
                    # Create an empty subvolume as substitute
                    subvol = np.zeros((subtomo_size, subtomo_size, subtomo_size), dtype=np.float32)
                    subtomograms.append(subvol)
            
            # Clean memory after each batch
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        # Close the MRC file
        if mrc is not None:
            mrc.close()
            
        # Convert to torch tensor and add channel dimension
        if subtomograms:
            subtomograms_tensor = torch.tensor(np.stack(subtomograms, axis=0), dtype=torch.float32).unsqueeze(1)
            
            elapsed_time = time.time() - start_time
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Successfully extracted {len(subtomograms)} subtomograms with shape {subtomograms_tensor.shape} in {elapsed_time:.2f} seconds")
            
            return subtomograms_tensor
        else:
            raise ValueError(f"No valid subtomograms extracted from {mrc_file}")
            
    except Exception as e:
        error_msg = f"Error loading data from {mrc_file}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        # If this is a test file, we should return a dummy tensor rather than failing
        if "test" in mrc_file.lower() or any(test_name in mrc_file for test_name in ["010", "011"]):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Creating dummy data for test file {mrc_file}")
            # Create a small set of dummy data
            dummy_size = 5
            dummy_data = torch.zeros((dummy_size, 1, subtomo_size, subtomo_size, subtomo_size), 
                                     dtype=torch.float32)
            return dummy_data
        else:
            raise  # Re-raise the exception for training files
        

def custom_collate_fn(batch):
    """
    Enhanced collate function that:
    1. Replaces NaN/Inf with zeros (or appropriate values)
    2. Handles empty batches gracefully
    3. Returns a valid batch even if some samples are problematic
    """
    inputs, targets, params = [], [], []

    # Process each sample with robust error handling
    for sample in batch:
        if not isinstance(sample, tuple) or len(sample) != 3:
            continue
            
        inp, tgt, pr = sample
        
        # Replace NaN/Inf with zeros
        inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
        pr = torch.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)

        inputs.append(inp)
        targets.append(tgt)
        params.append(pr)

    if len(inputs) == 0:
        # Create a dummy batch with zeros if all samples were invalid
        dummy_size = 1
        dummy_shape = next(iter(batch))[0].shape if batch else (1, 32, 32, 32)
        dummy_params_shape = next(iter(batch))[2].shape if batch else (6,)
        
        return (
            torch.zeros((dummy_size,) + dummy_shape),
            torch.zeros((dummy_size,) + dummy_shape),
            torch.zeros((dummy_size,) + dummy_params_shape)
        )

    # Stack and return the batch
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    params = torch.stack(params, dim=0)
    return inputs, targets, params

def shear_tensor(x, shear_factors, device):
    B, C, D, H, W = x.shape
    sheared = []

    for i in range(B):
        shear_x, shear_y, shear_z = shear_factors[i]
        affine_matrix = torch.tensor([
            [1, shear_x, shear_y, 0],
            [0, 1, shear_z, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32, device=device)
        grid = F.affine_grid(affine_matrix.unsqueeze(0), x[i:i+1].size(), align_corners=True)
        sheared_sample = F.grid_sample(x[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sheared.append(sheared_sample)

    sheared = torch.cat(sheared, dim=0)
    return sheared


# ===========================
# Transformation Functions
# ===========================
def apply_pretrain_transformations(x, device, max_rotation_angle, translation_range, shearing_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    shear_factors = torch.FloatTensor(B, 3).uniform_(-shearing_range, shearing_range).to(device)
    x_sheared = shear_tensor(x_rot, shear_factors, device=device)

    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)
    x_translated = shift_tensor(x_sheared, translations, device=device)

    x_noisy = x_translated

    transformed_input = x_noisy
    targets = x

    params = torch.cat([angles_rad, translations], dim=1)

    return transformed_input, targets, params

def apply_finetune_transformations(x, device, max_rotation_angle, translation_range, shearing_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

    shear_factors = torch.FloatTensor(B, 3).uniform_(-shearing_range, shearing_range).to(device)
    x_sheared = shear_tensor(x_rot, shear_factors, device=device)

    max_trans_d = D * translation_range
    max_trans_h = H * translation_range
    max_trans_w = W * translation_range
    trans_d = torch.FloatTensor(B).uniform_(-max_trans_d, max_trans_d).to(device)
    trans_h = torch.FloatTensor(B).uniform_(-max_trans_h, max_trans_h).to(device)
    trans_w = torch.FloatTensor(B).uniform_(-max_trans_w, max_trans_w).to(device)
    translations = torch.stack((trans_d, trans_h, trans_w), dim=1)
    x_translated = shift_tensor(x_sheared, translations, device=device)

    x_noisy = x_translated

    transformed_input = x_noisy
    targets = x

    params = torch.cat([angles_rad, translations], dim=1)

    return transformed_input, targets, params

def apply_testset_transformations(x, device, max_rotation_angle, translation_range, args):
    B, C, D, H, W = x.shape

    angles_deg = torch.FloatTensor(B, 3).uniform_(-max_rotation_angle, max_rotation_angle).to(device)
    angles_rad = angles_deg * math.pi / 180.0

    x_rot = rotate_tensor(x, angles_rad[:, 0], axes=(2, 3), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 1], axes=(2, 4), device=device)
    x_rot = rotate_tensor(x_rot, angles_rad[:, 2], axes=(3, 4), device=device)

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

def create_self_supervised_pairs(subtomograms, pretrain, num_augmentations, test, max_rotation_angle, translation_range, shearing_range):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Creating pairs with {num_augmentations} augmentations...")
    
    batch_size = 5
    num_batches = (num_augmentations + batch_size - 1) // batch_size
    
    transformed_inputs_list = []
    targets_list = []
    transformation_params_list = []

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_augmentations)
        current_batch_size = end_idx - start_idx
        
        batch_transformed_list = []
        batch_targets_list = []
        batch_params_list = []
        
        for aug_idx in range(current_batch_size):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Processing augmentation {start_idx + aug_idx}/{num_augmentations}")
                
            if test:
                transformed_inputs, targets, transformation_params = apply_testset_transformations(
                    subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, translation_range=translation_range, args = args
                )
            else:
                if pretrain:
                    transformed_inputs, targets, transformation_params = apply_pretrain_transformations(
                        subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, 
                        translation_range=translation_range, shearing_range=shearing_range, args = args
                    )
                else:
                    transformed_inputs, targets, transformation_params = apply_finetune_transformations(
                        subtomograms, device='cpu', max_rotation_angle=max_rotation_angle, 
                        translation_range=translation_range, shearing_range=shearing_range, args = args
                    )
            
            batch_transformed_list.append(transformed_inputs)
            batch_targets_list.append(targets)
            batch_params_list.append(transformation_params)
            
            torch.cuda.empty_cache()
            
            if dist.is_initialized():
                dist.barrier()
                
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  Augmentation {start_idx + aug_idx} completed")
        
        transformed_inputs_list.extend(batch_transformed_list)
        targets_list.extend(batch_targets_list)
        transformation_params_list.extend(batch_params_list)
        
        del batch_transformed_list
        del batch_targets_list
        del batch_params_list
        torch.cuda.empty_cache()
        
        if dist.is_initialized():
            dist.barrier()
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Batch {batch + 1}/{num_batches} completed")


    if transformed_inputs_list:
        transformed_inputs = torch.cat(transformed_inputs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        transformation_params = torch.cat(transformation_params_list, dim=0)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Data augmentation completed. Final shapes: {transformed_inputs.shape}")
        
        return transformed_inputs, targets, transformation_params
    else:
        raise RuntimeError("No data was processed in create_self_supervised_pairs")

def prepare_datasets(train_files, train_coords, valid_files, valid_coords, 
                     test_files, test_coords, split_ratio=0.4, seed=2):
    if dist.is_initialized():
        dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\nInitial GPU Memory Status:")
        print(torch.cuda.memory_summary())
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load and combine training data
    train_subtomograms_list = []
    for train_file, coord_file in zip(train_files, train_coords):
        subtomograms = load_data(train_file, coord_file, subtomo_size=args.subtomo_size)
        train_subtomograms_list.append(subtomograms)
    
    if len(train_subtomograms_list) > 1:
        train_subtomograms = torch.cat(train_subtomograms_list, dim=0)
    else:
        train_subtomograms = train_subtomograms_list[0]
    
    # Load and combine validation data
    valid_subtomograms_list = []
    for valid_file, coord_file in zip(valid_files, valid_coords):
        subtomograms = load_data(valid_file, coord_file, subtomo_size=args.subtomo_size)
        valid_subtomograms_list.append(subtomograms)
    
    if len(valid_subtomograms_list) > 1:
        valid_subtomograms = torch.cat(valid_subtomograms_list, dim=0)
    else:
        valid_subtomograms = valid_subtomograms_list[0]

    # Load test data
    test_subtomograms_list = []
    for test_file, coord_file in zip(test_files, test_coords):
        test_subtomograms = load_data(test_file, coord_file, subtomo_size=args.subtomo_size)
        test_subtomograms_list.append(test_subtomograms)

    def split_tensor(tensor, ratio, seed):
        torch.manual_seed(seed)
        shuffled_indices = torch.randperm(tensor.size(0))
        split_point = int(ratio * tensor.size(0))
        first_indices = shuffled_indices[:split_point]
        second_indices = shuffled_indices[split_point:]
        return tensor[first_indices], tensor[second_indices]
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Starting to split data...")
    
    pretrain_train_subtomograms, finetune_train_subtomograms = split_tensor(train_subtomograms, split_ratio, seed)
    pretrain_valid_subtomograms, finetune_valid_subtomograms = split_tensor(valid_subtomograms, split_ratio, seed)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data split completed.")
        print("Creating self-supervised pairs for pretraining...")

    transformed_pretrain_train, targets_pretrain_train, params_pretrain_train = create_self_supervised_pairs(
        pretrain_train_subtomograms, pretrain=True, num_augmentations=args.num_augmentations_pretrain, test=False,
        max_rotation_angle=args.rotation_range_pretrain, translation_range=args.translation_range_pretrain, 
        shearing_range=args.shearing_range_pretrain
    )
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for pretraining valid...")
    
    transformed_pretrain_valid, targets_pretrain_valid, params_pretrain_valid = create_self_supervised_pairs(
        pretrain_valid_subtomograms, pretrain=True, num_augmentations=args.num_augmentations_pretrain, test=False,
        max_rotation_angle=args.rotation_range_pretrain, translation_range=args.translation_range_pretrain, 
        shearing_range=args.shearing_range_pretrain
    )
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for finetuning...")
    
    transformed_finetune_train, targets_finetune_train, params_finetune_train = create_self_supervised_pairs(
        finetune_train_subtomograms, pretrain=False, num_augmentations=args.num_augmentations_finetune, test=False,
        max_rotation_angle=args.rotation_range_finetune, translation_range=args.translation_range_finetune, 
        shearing_range=args.shearing_range_finetune
    )
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating self-supervised pairs for finetuning valid...")
    
    transformed_finetune_valid, targets_finetune_valid, params_finetune_valid = create_self_supervised_pairs(
        finetune_valid_subtomograms, pretrain=False, num_augmentations=args.num_augmentations_finetune, test=False,
        max_rotation_angle=args.rotation_range_finetune, translation_range=args.translation_range_finetune, 
        shearing_range=args.shearing_range_finetune
    )
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Pretraining pairs created.")
    
    transformed_finetune_test_list = []
    targets_finetune_test_list = []
    params_finetune_test_list = []
    
    for idx, test_subtomograms in enumerate(test_subtomograms_list):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Creating self-supervised pairs for test set {idx+1}...")
            print(f"Test set {idx+1} shape: {test_subtomograms.shape}, "
                  f"min: {test_subtomograms.min().item():.4f}, "
                  f"max: {test_subtomograms.max().item():.4f}")
        
        transformed_test, targets_test, params_test = create_self_supervised_pairs(
            test_subtomograms, pretrain=False, num_augmentations=1, test=True,
            max_rotation_angle=args.rotation_range_test, translation_range=args.translation_range_test, 
            shearing_range=args.shearing_range_finetune
        )
        
        transformed_finetune_test_list.append(transformed_test)
        targets_finetune_test_list.append(targets_test)
        params_finetune_test_list.append(params_test)

    debug_print("\n===== Augmented Dataset Sizes ======")
    debug_print(f"Pretraining Training Set: {transformed_pretrain_train.shape[0]} samples")
    debug_print(f"Pretraining Validation Set: {transformed_pretrain_valid.shape[0]} samples")
    debug_print(f"Fine-Tuning Training Set: {transformed_finetune_train.shape[0]} samples")
    debug_print(f"Fine-Tuning Validation Set: {transformed_finetune_valid.shape[0]} samples")
    
    for idx, test_set in enumerate(transformed_finetune_test_list):
        debug_print(f"Fine-Tuning Test Set {idx+1}: {test_set.shape[0]} samples")
    
    debug_print("====================================\n")
    
    trans_pre_mean = params_pretrain_train.mean(dim=0, keepdim=True)
    trans_pre_std = params_pretrain_train.std(dim=0, keepdim=True)
    
    if dist.is_initialized():
        dist.all_reduce(trans_pre_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_pre_std, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        trans_pre_mean /= world_size
        trans_pre_std /= world_size
    
    trans_fine_mean = params_finetune_train.mean(dim=0, keepdim=True)
    trans_fine_std = params_finetune_train.std(dim=0, keepdim=True)
    
    if dist.is_initialized():
        dist.all_reduce(trans_fine_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_fine_std, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        trans_fine_mean /= world_size
        trans_fine_std /= world_size
    
    params_pretrain_train_normalized = (params_pretrain_train - trans_pre_mean) / trans_pre_std
    params_pretrain_valid_normalized = (params_pretrain_valid - trans_pre_mean) / trans_pre_std
    
    params_finetune_train_normalized = (params_finetune_train - trans_fine_mean) / trans_fine_std
    params_finetune_valid_normalized = (params_finetune_valid - trans_fine_mean) / trans_fine_std
    
    params_finetune_test_normalized_list = []
    for params_test in params_finetune_test_list:
        params_test_normalized = (params_test - trans_fine_mean) / trans_fine_std
        params_finetune_test_normalized_list.append(params_test_normalized)
    
    mean_pre = transformed_pretrain_train.mean()
    std_pre = transformed_pretrain_train.std()
    
    if dist.is_initialized():
        mean_pre = torch.tensor([mean_pre], device=transformed_pretrain_train.device)
        std_pre = torch.tensor([std_pre], device=transformed_pretrain_train.device)
        
        dist.all_reduce(mean_pre, op=dist.ReduceOp.SUM)
        dist.all_reduce(std_pre, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        mean_pre = mean_pre.item() / world_size
        std_pre = std_pre.item() / world_size
    
    transformed_pretrain_train = (transformed_pretrain_train - mean_pre) / std_pre
    targets_pretrain_train = (targets_pretrain_train - mean_pre) / std_pre
    transformed_pretrain_valid = (transformed_pretrain_valid - mean_pre) / std_pre
    targets_pretrain_valid = (targets_pretrain_valid - mean_pre) / std_pre
    
    mean_fine = transformed_finetune_train.mean()
    std_fine = transformed_finetune_train.std()
    
    if dist.is_initialized():
        mean_fine = torch.tensor([mean_fine], device=transformed_finetune_train.device)
        std_fine = torch.tensor([std_fine], device=transformed_finetune_train.device)
        
        dist.all_reduce(mean_fine, op=dist.ReduceOp.SUM)
        dist.all_reduce(std_fine, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        mean_fine = mean_fine.item() / world_size
        std_fine = std_fine.item() / world_size
    
    transformed_finetune_train = (transformed_finetune_train - mean_fine) / std_fine
    targets_finetune_train = (targets_finetune_train - mean_fine) / std_fine
    transformed_finetune_valid = (transformed_finetune_valid - mean_fine) / std_fine
    targets_finetune_valid = (targets_finetune_valid - mean_fine) / std_fine

    transformed_finetune_test_normalized_list = []
    targets_finetune_test_normalized_list = []
    
    for idx, (transformed_test, targets_test) in enumerate(zip(transformed_finetune_test_list, targets_finetune_test_list)):
        # Print diagnostics for test data before normalization
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Test set {idx+1} before normalization - min: {transformed_test.min().item():.4f}, "
                  f"max: {transformed_test.max().item():.4f}, "
                  f"mean: {transformed_test.mean().item():.4f}, "
                  f"std: {transformed_test.std().item():.4f}")
        
        # Compute statistics for normalization
        mean_test = transformed_test.mean()
        std_test = transformed_test.std()
        
        # Ensure std is not too small to avoid division issues
        if std_test < 1e-6:
            print(f"WARNING: Very small std detected in test set {idx+1}: {std_test.item():.8f}, using 1.0 instead")
            std_test = torch.tensor(1.0, device=std_test.device)
        
        # Apply normalization
        transformed_test_norm = (transformed_test - mean_test) / std_test
        targets_test_norm = (targets_test - mean_test) / std_test
        
        # Validate normalized data
        if torch.isnan(transformed_test_norm).any():
            print(f"WARNING: NaNs detected in normalized test data {idx+1}, fixing...")
            transformed_test_norm = torch.nan_to_num(transformed_test_norm)
        
        if torch.isnan(targets_test_norm).any():
            print(f"WARNING: NaNs detected in normalized targets {idx+1}, fixing...")
            targets_test_norm = torch.nan_to_num(targets_test_norm)
        
        # Print diagnostics after normalization
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Test set {idx+1} after normalization - min: {transformed_test_norm.min().item():.4f}, "
                  f"max: {transformed_test_norm.max().item():.4f}, "
                  f"unique values: {torch.unique(transformed_test_norm).shape[0]}")
        
        transformed_finetune_test_normalized_list.append(transformed_test_norm)
        targets_finetune_test_normalized_list.append(targets_test_norm)
    
    train_dataset_pretrain = TensorDataset(transformed_pretrain_train, targets_pretrain_train, params_pretrain_train_normalized)
    valid_dataset_pretrain = TensorDataset(transformed_pretrain_valid, targets_pretrain_valid, params_pretrain_valid_normalized)
    
    train_dataset_finetune = TensorDataset(transformed_finetune_train, targets_finetune_train, params_finetune_train_normalized)
    valid_dataset_finetune = TensorDataset(transformed_finetune_valid, targets_finetune_valid, params_finetune_valid_normalized)

    test_datasets_finetune = []
    for transformed_test, targets_test, params_test in zip(
        transformed_finetune_test_normalized_list,
        targets_finetune_test_normalized_list,
        params_finetune_test_normalized_list
    ):
        test_dataset = TensorDataset(transformed_test, targets_test, params_test)
        test_datasets_finetune.append(test_dataset)
        
    if dist.is_initialized():
        dist.barrier()
        
    return (
        train_dataset_pretrain,
        valid_dataset_pretrain,
        train_dataset_finetune,
        valid_dataset_finetune,
        test_datasets_finetune,
        trans_pre_mean, trans_pre_std,
        trans_fine_mean, trans_fine_std,
        mean_pre, std_pre,
        mean_fine, std_fine
    )

def get_dataloaders(train_pretrain, valid_pretrain, train_finetune, valid_finetune, test_datasets, 
                    pretrain_batch_size=4, finetune_batch_size=4, test_batch_size=4):
    if dist.is_initialized():
        train_pretrain_sampler = torch.utils.data.distributed.DistributedSampler(train_pretrain, shuffle=True)
        val_pretrain_sampler = torch.utils.data.distributed.DistributedSampler(valid_pretrain, shuffle=True)
        train_finetune_sampler = torch.utils.data.distributed.DistributedSampler(train_finetune, shuffle=True)
        val_finetune_sampler = torch.utils.data.distributed.DistributedSampler(valid_finetune, shuffle=True)

        train_pretrain_batch_sampler = torch.utils.data.BatchSampler(train_pretrain_sampler, pretrain_batch_size, drop_last=True)
        val_pretrain_batch_sampler = torch.utils.data.BatchSampler(val_pretrain_sampler, pretrain_batch_size, drop_last=False)
        train_finetune_batch_sampler = torch.utils.data.BatchSampler(train_finetune_sampler, finetune_batch_size, drop_last=True)
        val_finetune_batch_sampler = torch.utils.data.BatchSampler(val_finetune_sampler, finetune_batch_size, drop_last=False)
        
        pretrain_train_loader = DataLoader(train_pretrain, batch_sampler=train_pretrain_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        pretrain_valid_loader = DataLoader(valid_pretrain, batch_sampler=val_pretrain_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_train_loader = DataLoader(train_finetune, batch_sampler=train_finetune_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_valid_loader = DataLoader(valid_finetune, batch_sampler=val_finetune_batch_sampler, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

        test_loaders = []
        for test_dataset in test_datasets:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False)
            test_batch_sampler = torch.utils.data.BatchSampler(
                test_sampler, test_batch_size, drop_last=False)
            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_batch_sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=default_collate  # Using default_collate for test
            )
            test_loaders.append(test_loader)            
            
    else:
        pretrain_train_loader = DataLoader(train_pretrain, batch_size=pretrain_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        pretrain_valid_loader = DataLoader(valid_pretrain, batch_size=pretrain_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_train_loader = DataLoader(train_finetune, batch_size=finetune_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        finetune_valid_loader = DataLoader(valid_finetune, batch_size=finetune_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
        test_loaders = []
        for test_dataset in test_datasets:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=test_batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True, 
                collate_fn=default_collate  # Changed from custom_collate_fn to default_collate
            )
            test_loaders.append(test_loader)

    return pretrain_train_loader, pretrain_valid_loader, finetune_train_loader, finetune_valid_loader, test_loaders

def initialize_model(args, patch_size, device):
    debug_print("Initializing SETransformerEulerSampling model")
    
    # Set seeds for reproducibility across processes
    torch.manual_seed(args.local_rank)
    np.random.seed(args.local_rank)
    
    # Create model
    model = SETransformerEulerSampling(
        in_channels=1,
        num_transformer_blocks=4,
        num_heads=8,
        ff_hidden_dim=512,
        hidden_dim=120,
        feature_type='vector',
        patch_size=patch_size
    )
    
    model = model.to(device)
    
    if dist.is_initialized():
        # Remove the timeout parameter as it's not supported in this PyTorch version
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )
        
        # Initialize after synchronization
        torch.distributed.barrier()
    
    return model


def get_optimizer_and_loss(model, pretrain=True):
    if pretrain:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_PRETRAIN, weight_decay=WEIGHT_DECAY_PRETRAIN)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY_FINETUNE)

    transform_type = 'euler'
    mse_loss_fn = lambda predictions, ground_truths: transformation_loss(predictions, ground_truths, transform_type=transform_type)
    cc_loss_fn = cross_correlation_loss
    alignment_loss_fn = gradient_difference_loss

    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5, 
        verbose=(not dist.is_initialized() or dist.get_rank() == 0)
    )      
    return optimizer, mse_loss_fn, cc_loss_fn, alignment_loss_fn, scaler, scheduler

# ===========================
# Check points
# ===========================
def save_checkpoint(state, filename, rank):
    if rank == 0:
        torch.save(state, filename)
        debug_print(f"Checkpoint saved to '{filename}'")
        
# ===========================
# Train Model
# ===========================
def train_model(model, train_loader, optimizer, mse_loss_fn, cc_loss_fn, alignment_loss_fn, 
                scaler, scheduler, device, epoch_losses, trans_pre_mean, trans_pre_std, 
                phase='Training', transform_type='euler'):
    model.train()
    metrics = DistributedMetricsTracker(device)
    
    alpha_final = 3.0    
    beta_partial = 1.0   
    gamma_reg = 0.1      
    delta_smooth = 0.05  
    
    trans_pre_mean = trans_pre_mean.to(device)
    trans_pre_std = trans_pre_std.to(device)
    
    if hasattr(train_loader.batch_sampler, 'sampler'):
        train_loader.batch_sampler.sampler.set_epoch(len(epoch_losses['total']))
    
    torch.distributed.barrier()
    
    for batch_idx, (inputs, targets, params) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            transform_preds, final_aligned, aligned_volumes = model(input=inputs, target=targets)
            
            gt_transformation = params[:, :6].to(device)
            
            final_trans_pred = transform_preds[-1]
            loss_final_transformation = mse_loss_fn(final_trans_pred, gt_transformation)
            
            loss_partial_transforms = 0
            num_blocks = len(transform_preds)
            for i, trans_pred in enumerate(transform_preds):
                block_weight = (i + 1) / num_blocks
                block_loss = mse_loss_fn(trans_pred, gt_transformation)
                loss_partial_transforms += block_weight * block_loss
            
            if num_blocks > 0:
                loss_partial_transforms /= num_blocks
            
            loss_partial_alignment = 0
            for i, aligned in enumerate(aligned_volumes):
                align_weight = (i + 1) / len(aligned_volumes)
                align_loss = F.mse_loss(aligned, targets)
                loss_partial_alignment += align_weight * align_loss
            
            if aligned_volumes:
                loss_partial_alignment /= len(aligned_volumes)
            
            loss_regularization = 0
            for trans_pred in transform_preds:
                loss_regularization += torch.mean(torch.abs(trans_pred[:, :3]))  
                loss_regularization += torch.mean(torch.abs(trans_pred[:, 3:]))  
            
            if transform_preds:
                loss_regularization /= len(transform_preds)
            
            loss_smoothness = 0
            if len(transform_preds) > 1:
                for i in range(1, len(transform_preds)):
                    loss_smoothness += F.mse_loss(transform_preds[i], transform_preds[i-1])
                loss_smoothness /= (len(transform_preds) - 1)
            
            loss = (
                alpha_final * loss_final_transformation + 
                beta_partial * loss_partial_transforms +
                beta_partial * loss_partial_alignment +
                gamma_reg * loss_regularization +
                delta_smooth * loss_smoothness
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if not (torch.isnan(loss).any() or 
                any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)):
            scaler.step(optimizer)
        
        scaler.update()
        
        metrics.update(loss.detach(), loss_final_transformation.detach(), inputs.size(0))
        torch.distributed.barrier()
        
        current_metrics = metrics.get_metrics()
        debug_print(f"{phase} - GPU{dist.get_rank()}: "
                  f"Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {current_metrics['loss']:.4f}")


# ============================================
# VALIDATE_MODEL 
# ============================================
def validate_model(model, valid_loader, mse_loss_fn, cc_loss_fn, alignment_loss_fn, 
                  device, epoch_losses, scheduler, trans_pre_mean, trans_pre_std, 
                  phase='Validation', transform_type='euler'):
    model.eval()
    metrics = DistributedMetricsTracker(device)
    
    alpha_final = 3.0
    beta_partial = 1.0
    
    trans_pre_mean = trans_pre_mean.to(device)
    trans_pre_std = trans_pre_std.to(device)
    
    torch.distributed.barrier()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, params) in enumerate(valid_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            
            transform_preds, final_aligned, aligned_volumes = model(input=inputs, target=targets)
            
            gt_transformation = params[:, :6].to(device)
            
            final_trans_pred = transform_preds[-1]
            loss_final_transformation = mse_loss_fn(final_trans_pred, gt_transformation)
            
            loss_partial_transforms = 0
            for i, trans_pred in enumerate(transform_preds):
                weight = (i + 1) / len(transform_preds) if transform_preds else 1
                step_loss = mse_loss_fn(trans_pred, gt_transformation)
                loss_partial_transforms += weight * step_loss
            
            if transform_preds:
                loss_partial_transforms /= len(transform_preds)
            
            loss_partial_alignment = 0
            for i, aligned in enumerate(aligned_volumes):
                weight = (i + 1) / len(aligned_volumes) if aligned_volumes else 1
                align_loss = F.mse_loss(aligned, targets)
                loss_partial_alignment += weight * align_loss
            
            if aligned_volumes:
                loss_partial_alignment /= len(aligned_volumes)
            
            loss = (
                alpha_final * loss_final_transformation + 
                beta_partial * loss_partial_transforms +
                beta_partial * loss_partial_alignment
            )
            
            metrics.update(loss.detach(), loss_final_transformation.detach(), inputs.size(0))
    
    metrics.synchronize()
    final_metrics = metrics.get_metrics()
    
    if dist.get_rank() == 0:
        epoch_losses['total'].append(final_metrics['loss'])
        epoch_losses['transformation'].append(final_metrics['trans_loss'])
        print(f"{phase} - Epoch Summary:\n"
             f"Avg Loss: {final_metrics['loss']:.4f}\n"
             f"Trans Loss: {final_metrics['trans_loss']:.4f}")
    
    scheduler.step(final_metrics['loss'])
    torch.distributed.barrier()
    return final_metrics['loss']


def test_model(model, test_loader, device, trans_fine_mean, trans_fine_std, test_file_name, transform_type='euler'):
    model.eval()
    local_size = 0
    predicted_trans = []
    ground_truth_trans = []
    
    aligned_volumes = []
    target_volumes = []

    trans_fine_mean = trans_fine_mean.to(device)
    trans_fine_std = trans_fine_std.to(device)
    
    debug_batch_limit = 5  # Number of batches to show detailed debug info
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets, params) in enumerate(test_loader):
                try:
                    # Handle NaN/Inf in inputs, targets, and params
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                    params = torch.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    params = params.to(device, non_blocking=True)

                    # Debug batch info
                    if batch_idx < debug_batch_limit:
                        print(f"Batch {batch_idx} - input shape: {inputs.shape}, "
                              f"min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}")
 
                    # Forward pass
                    transform_preds, aligned_input, aligned_steps = model(input=inputs, target=targets)
                    
                    # Debug aligned volume info
                    if batch_idx < debug_batch_limit:
                        print(f"Batch {batch_idx} - aligned volume - "
                              f"min: {aligned_input.min().item():.4f}, max: {aligned_input.max().item():.4f}, "
                              f"sum: {aligned_input.abs().sum().item():.4f}")
                    
                    if not transform_preds:
                        print(f"Empty transform_preds for batch {batch_idx+1}")
                        continue
                            
                    transformation_pred = transform_preds[-1]
                    
                    # Check for NaN/Inf in predictions and replace them
                    if torch.isnan(transformation_pred).any() or torch.isinf(transformation_pred).any():
                        print(f"NaN/Inf detected in predictions for batch {batch_idx+1}, fixing...")
                        transformation_pred = torch.nan_to_num(transformation_pred)

                    gt_transformation = params[:, :6]

                    # Unnormalize with robust handling of NaN/Inf
                    transformation_pred_unnorm = transformation_pred * trans_fine_std + trans_fine_mean
                    gt_transformation_unnorm = gt_transformation * trans_fine_std + trans_fine_mean

                    predicted_trans.append(transformation_pred_unnorm.cpu())
                    ground_truth_trans.append(gt_transformation_unnorm.cpu())
                    
                    # Handle NaN/Inf in aligned volumes and check for zero volumes
                    aligned_input = torch.nan_to_num(aligned_input)
                    targets = torch.nan_to_num(targets)
                    
                    # IMPORTANT: Fix for zero volumes - clone to ensure detached copy
                    aligned_volume_copy = aligned_input.clone().detach().cpu()
                    target_volume_copy = targets.clone().detach().cpu()
                    
                    # Check if volume is zero and try to use last aligned step if needed
                    if aligned_volume_copy.abs().sum() < 1e-6 and aligned_steps and len(aligned_steps) > 0:
                        last_step = aligned_steps[-1].clone().detach()
                        if last_step.abs().sum() >= 1e-6:
                            if batch_idx < debug_batch_limit:
                                print(f"Batch {batch_idx} - using last aligned step instead of zero volume")
                            aligned_volume_copy = last_step.cpu()
                    
                    # Verify volumes again
                    if batch_idx < debug_batch_limit:
                        print(f"Batch {batch_idx} - final aligned volume copy - "
                              f"sum: {aligned_volume_copy.abs().sum().item():.4f}")
                    
                    aligned_volumes.append(aligned_volume_copy)
                    target_volumes.append(target_volume_copy)
                    
                    local_size += transformation_pred_unnorm.size(0)
                    
                    if batch_idx % 100 == 0:
                        print(f"Successfully processed batch {batch_idx+1}")
                
                except Exception as batch_error:
                    print(f"Error processing batch {batch_idx+1}: {str(batch_error)}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Count zero volumes before returning
            zero_count = sum(1 for vol in aligned_volumes if vol.abs().sum() < 1e-6)
            print(f"Total processed {local_size} samples, {zero_count}/{len(aligned_volumes)} volumes are zero")
            
    except Exception as e:
        print(f"Error in test_model: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return predicted_trans, ground_truth_trans, aligned_volumes, target_volumes

def evaluate_model(predicted_trans, ground_truth_trans, aligned_volumes=None, target_volumes=None, image_size=32, test_file_name="Test", transform_type='euler', pixel_size=2.0, fsc_threshold=0.5):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
        
    print(f"\nStarting evaluation of alignment quality for '{test_file_name}'...")
    
    if not predicted_trans:
        print(f"No predictions to evaluate for '{test_file_name}'.")
        return
        
    try:
        # 统计预测的转换参数
        for i, pred_batch in enumerate(predicted_trans):
            if i < 3:  # 只检查前3个batch
                print(f"Predicted batch {i} shape: {pred_batch.shape}, min: {pred_batch.min().item():.4f}, max: {pred_batch.max().item():.4f}")
        
        predicted_trans = torch.cat(predicted_trans, dim=0).float()
        ground_truth_trans = torch.cat(ground_truth_trans, dim=0).float()
        
        print(f"Combined predictions shape: {predicted_trans.shape}, Combined GT shape: {ground_truth_trans.shape}")
        print(f"Predictions stats - min: {predicted_trans.min().item():.4f}, max: {predicted_trans.max().item():.4f}")
        print(f"Ground truth stats - min: {ground_truth_trans.min().item():.4f}, max: {ground_truth_trans.max().item():.4f}")
        
        # 按照GUM-Net方法，添加体积平均步骤
        has_volume_data = False
        if aligned_volumes is not None and target_volumes is not None:
            try:
                # 保存结果目录
                fsc_dir = os.path.join(args.output_dir, 'fsc_results')
                os.makedirs(fsc_dir, exist_ok=True)
                
                # Check aligned volumes before concatenation
                for i, vol in enumerate(aligned_volumes[:5]):
                    vol_sum = vol.abs().sum().item()
                    print(f"Volume {i} check - shape: {vol.shape}, sum: {vol_sum:.4f}, "
                          f"is zero: {vol_sum < 1e-6}")
                
                # 统计体积数据
                all_aligned_volumes = torch.cat(aligned_volumes, dim=0).float()
                all_target_volumes = torch.cat(target_volumes, dim=0).float()
                
                print(f"Volume data available for FSC evaluation: {all_aligned_volumes.shape}")
                print(f"Aligned volumes stats - min: {all_aligned_volumes.min().item():.4f}, max: {all_aligned_volumes.max().item():.4f}")
                print(f"Target volumes stats - min: {all_target_volumes.min().item():.4f}, max: {all_target_volumes.max().item():.4f}")
                print(f"Aligned volumes sum: {all_aligned_volumes.abs().sum().item():.4f}")
                print(f"Target volumes sum: {all_target_volumes.abs().sum().item():.4f}")
                
                # 保存一些对齐和目标体积的切片图像
                import matplotlib.pyplot as plt
                for i in range(min(5, all_aligned_volumes.shape[0])):
                    mid_slice = all_aligned_volumes.shape[2] // 2
                    
                    plt.figure(figsize=(12, 5))
                    plt.subplot(121)
                    plt.imshow(all_aligned_volumes[i, 0, mid_slice].numpy(), cmap='gray')
                    plt.title(f"Aligned Volume {i} (slice {mid_slice})")
                    plt.colorbar()
                    
                    plt.subplot(122)
                    plt.imshow(all_target_volumes[i, 0, mid_slice].numpy(), cmap='gray')
                    plt.title(f"Target Volume {i} (slice {mid_slice})")
                    plt.colorbar()
                    
                    plt.savefig(os.path.join(fsc_dir, f"{test_file_name}_sample{i}_slices.png"))
                    plt.close()
                
                # ===== 新增：GUM-Net方式的亚体积平均 =====
                # 将所有对齐的体积平均起来
                print("Performing subtomogram averaging according to GUM-Net approach...")
                
                # 1. 移除任何可能影响平均的填充或无效数据
                valid_mask = torch.ones(all_aligned_volumes.shape[0], dtype=torch.bool)
                for i in range(all_aligned_volumes.shape[0]):
                    # 检查体积是否包含所有零值或NaN/Inf
                    if torch.isnan(all_aligned_volumes[i]).any() or torch.isinf(all_aligned_volumes[i]).any() or \
                       all_aligned_volumes[i].abs().sum() < 1e-6:
                        valid_mask[i] = False
                
                print(f"Valid subtomograms for averaging: {valid_mask.sum().item()}/{all_aligned_volumes.shape[0]}")
                
                if valid_mask.sum() == 0:
                    print("No valid volumes for averaging! Skipping FSC calculation.")
                    has_volume_data = False
                else:
                    # 2. 计算有效体积的平均值
                    valid_aligned_volumes = all_aligned_volumes[valid_mask]
                    valid_target_volumes = all_target_volumes[valid_mask]
                    
                    # 计算平均体积
                    avg_aligned_volume = valid_aligned_volumes.mean(dim=0, keepdim=True)
                    avg_target_volume = valid_target_volumes.mean(dim=0, keepdim=True)
                    
                    # 保存平均体积切片
                    mid_slice = avg_aligned_volume.shape[2] // 2
                    
                    plt.figure(figsize=(12, 5))
                    plt.subplot(121)
                    plt.imshow(avg_aligned_volume[0, 0, mid_slice].numpy(), cmap='gray')
                    plt.title(f"Average Aligned Volume (slice {mid_slice})")
                    plt.colorbar()
                    
                    plt.subplot(122)
                    plt.imshow(avg_target_volume[0, 0, mid_slice].numpy(), cmap='gray')
                    plt.title(f"Average Target Volume (slice {mid_slice})")
                    plt.colorbar()
                    
                    plt.savefig(os.path.join(fsc_dir, f"{test_file_name}_average_slices.png"))
                    plt.close()
                    
                    print("Subtomogram averaging completed successfully.")
                    has_volume_data = True
                    
                    # 存储平均体积以供FSC计算
                    all_aligned_volumes = avg_aligned_volume
                    all_target_volumes = avg_target_volume
                
            except Exception as e:
                print(f"Error processing volume data: {e}")
                import traceback
                traceback.print_exc()
                has_volume_data = False
        else:
            has_volume_data = False
            print("No volume data available for FSC evaluation.")
    except Exception as e:
        print(f"Error concatenating tensors: {e}")
        import traceback
        traceback.print_exc()
        return
    
    expected_dim = 6
    
    if ground_truth_trans.size(1) != expected_dim or predicted_trans.size(1) != expected_dim:
        print(f"[ERROR] Expected {expected_dim} transformation parameters, but got "
              f"{ground_truth_trans.size(1)} ground truth and {predicted_trans.size(1)} predicted "
              f"for '{test_file_name}'. Evaluation aborted.")
        return
    
    fsc_dir = os.path.join(args.output_dir, 'fsc_results')
    os.makedirs(fsc_dir, exist_ok=True)
    
    print(f"\nEvaluating overall alignment quality for '{test_file_name}'...")
    try:
        alignment_results = alignment_eval(
            ground_truth_trans, 
            predicted_trans, 
            image_size=image_size, 
            scale=False,
            transform_type=transform_type
        )
    except Exception as e:
        print(f"Error in overall alignment evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    if has_volume_data:
        print("\nEvaluating FSC on averaged volumes...")
        
        try:
            # 使用平均后的体积计算FSC
            resolution, fsc_curve = calculate_resolution_fsc(
                all_aligned_volumes[0, 0],  # 提取单个平均体积
                all_target_volumes[0, 0],
                pixel_size=pixel_size,
                threshold=fsc_threshold,
                device='cpu'
            )
            
            print(f"FSC Resolution for averaged volumes: {resolution:.2f} Å")
            
            # 绘制FSC曲线
            curve_filename = os.path.join(fsc_dir, f"{test_file_name}_average_fsc_curve.png")
            plot_fsc_curve(fsc_curve, threshold=fsc_threshold, filename=curve_filename)
            
            # 保存FSC结果
            result_file = os.path.join(fsc_dir, f"{test_file_name}_fsc_summary.txt")
            with open(result_file, 'w') as f:
                f.write(f"FSC Evaluation Summary for {test_file_name}\n")
                f.write(f"Pixel Size: {pixel_size} Å\n")
                f.write(f"FSC Threshold: {fsc_threshold}\n")
                f.write(f"Evaluation Method: GUM-Net style subtomogram averaging\n\n")
                f.write(f"Resolution: {resolution:.2f} Å\n")
            
            # 如果支持, 添加到wandb
            if wandb.run is not None:
                wandb.log({
                    f"FSC/{test_file_name}_resolution": resolution,
                    f"FSC/{test_file_name}_avg_curve": wandb.Image(curve_filename)
                })
            
        except Exception as e:
            print(f"Error in FSC calculation: {e}")
            import traceback
            traceback.print_exc()

# ===========================
# Main Execution Block 
# ===========================
def main():
    torch.backends.cudnn.enabled = True
    init_wandb(args)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Distributed learning enabled: GPU{args.local_rank}.")
    else:
        print(f'Not using distributed mode')
        return

    torch.cuda.set_device(args.local_rank)

    torch.backends.cudnn.enabled = True
    if dist.get_rank()==0:
        print("cuDNN enabled.")

    DATA_DIR = '/shared/u/c_mru/EMPIAR-10045/voxel_spacing_2A'
    
    MRC_FILES = [
        os.path.join(DATA_DIR, 'IS002_291013_005.mrc'),
        os.path.join(DATA_DIR, 'IS002_291013_006.mrc'),
        os.path.join(DATA_DIR, 'IS002_291013_007.mrc'),
        os.path.join(DATA_DIR, 'IS002_291013_008.mrc'),
        os.path.join(DATA_DIR, 'IS002_291013_009.mrc'),
        os.path.join(DATA_DIR, 'IS002_291013_010.mrc'),
        # Removed: os.path.join(DATA_DIR, 'IS002_291013_011.mrc')
    ]
    
    COORDS_FILES = [
        os.path.join(DATA_DIR, 'IS002_291013_005.coords'),
        os.path.join(DATA_DIR, 'IS002_291013_006.coords'),
        os.path.join(DATA_DIR, 'IS002_291013_007.coords'),
        os.path.join(DATA_DIR, 'IS002_291013_008.coords'),
        os.path.join(DATA_DIR, 'IS002_291013_009.coords'),
        os.path.join(DATA_DIR, 'IS002_291013_010.coords'),
        # Removed: os.path.join(DATA_DIR, 'IS002_291013_011.coords')
    ]

    if dist.get_rank() == 0:
        try:
            print("Successfully logged into Weights & Biases (mock).")
        except Exception as e:
            print(f"Error logging into Weights & Biases: {e}")
            print("wandb logging will be skipped.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    args.transform_type = 'euler'
    args.use_sampling = True

    TRAIN_FILES = MRC_FILES[:4]  # 使用前4个文件作为训练集
    VALID_FILES = [MRC_FILES[4]]  # 使用第5个文件作为验证集
    TEST_FILES = [MRC_FILES[5]]   # 使用第6个文件作为测试集
    
    TRAIN_COORDS = COORDS_FILES[:4]
    VALID_COORDS = [COORDS_FILES[4]]
    TEST_COORDS = [COORDS_FILES[5]]
    
    (train_dataset_pretrain,
     valid_dataset_pretrain,
     train_dataset_finetune,
     valid_dataset_finetune,
     test_datasets_finetune,
     trans_pre_mean, trans_pre_std,
     trans_fine_mean, trans_fine_std,
     mean_pre, std_pre,
     mean_fine, std_fine) = prepare_datasets(
        train_files=TRAIN_FILES,
        train_coords=TRAIN_COORDS,
        valid_files=VALID_FILES,
        valid_coords=VALID_COORDS,
        test_files=TEST_FILES,
        test_coords=TEST_COORDS,
        split_ratio=0.4,
        seed=SEED
    )
     
    pretrain_train_loader, pretrain_valid_loader, \
    finetune_train_loader, finetune_valid_loader, \
    test_loaders = get_dataloaders(
        train_pretrain=train_dataset_pretrain,
        valid_pretrain=valid_dataset_pretrain,
        train_finetune=train_dataset_finetune,
        valid_finetune=valid_dataset_finetune,
        test_datasets=test_datasets_finetune,
        pretrain_batch_size=PRETRAIN_BATCH_SIZE,
        finetune_batch_size=FINETUNE_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE
    )
    
    torch.cuda.empty_cache()
    debug_print(f"GPU{dist.get_rank()}: GPU memory allocated before model definition: {torch.cuda.memory_allocated(DEVICE) // 1e6} MiB")

    model = initialize_model(args=args, patch_size=PATCH_SIZE, device=DEVICE)
    debug_print(f"GPU{dist.get_rank()}: GPU memory allocated after model definition: {torch.cuda.memory_allocated(DEVICE) // 1e6} MiB")
    pretrain_exists = os.path.exists(BEST_MODEL_PATH_PRETRAIN)
    finetune_exists = os.path.exists(BEST_MODEL_PATH_FINETUNE)


    # ===========================
    # Pretraining Phase
    # ===========================
    pretrain_start_time = time.time() if torch.cuda.is_available() else None
    pretrain_runtime = 0.0
    
    optimizer_pretrain, mse_loss_fn_pretrain, cc_loss_fn_pretrain, alignment_loss_fn_pretrain, scaler_pretrain, scheduler_pretrain = get_optimizer_and_loss(model, pretrain=True)
    pretrain_epoch_losses = {'total': [], 'transformation': []}
    start_epoch = 0
    best_val_loss_pretrain = float('inf')
    epochs_no_improve_pretrain = 0
    
    if args.resume_pretrain and os.path.exists(args.resume_pretrain):
        debug_print(f"Resuming pretraining from checkpoint: {args.resume_pretrain}")
        checkpoint = torch.load(args.resume_pretrain, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer_pretrain.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_pretrain.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss_pretrain = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve_pretrain = checkpoint.get('epochs_no_improve', 0)
            pretrain_epoch_losses = checkpoint.get('epoch_losses', {'total': [], 'transformation': []})
            pretrain_runtime = checkpoint.get('runtime', 0.0)
            pretrain_start_time = time.time() if torch.cuda.is_available() else None
        else:
            debug_print(f"Warning: Invalid checkpoint format in {args.resume_pretrain}")
    elif pretrain_exists and not args.force_train:
        debug_print(f"Found an existing best pre-trained model at '{BEST_MODEL_PATH_PRETRAIN}'. Skipping pretraining.")
        checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss_pretrain = checkpoint.get('best_val_loss', float('inf'))
        else:
            debug_print(f"Warning: Invalid checkpoint format in {BEST_MODEL_PATH_PRETRAIN}")
    else:
        debug_print("Starting Pre-Training Phase from scratch...")
    
    if args.resume_pretrain or not pretrain_exists or args.force_train:
        for epoch in range(start_epoch, PRETRAIN_EPOCHS):
            epoch_start_time = time.time() if torch.cuda.is_available() else None
            debug_print(f"\nPre-Training Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
    
            train_loss = train_model(
                model=model,
                train_loader=pretrain_train_loader,
                optimizer=optimizer_pretrain,
                mse_loss_fn=mse_loss_fn_pretrain,
                cc_loss_fn=cc_loss_fn_pretrain,
                alignment_loss_fn=alignment_loss_fn_pretrain,
                scaler=scaler_pretrain,
                scheduler=scheduler_pretrain,
                device=DEVICE,
                epoch_losses=pretrain_epoch_losses,
                trans_pre_mean=trans_pre_mean,
                trans_pre_std=trans_pre_std,
                phase='Pre-Training',
                transform_type=args.transform_type  
            )
    
            val_loss = validate_model(
                model=model,
                valid_loader=pretrain_valid_loader,
                mse_loss_fn=mse_loss_fn_pretrain,
                cc_loss_fn=cc_loss_fn_pretrain,
                alignment_loss_fn=alignment_loss_fn_pretrain,
                device=DEVICE,
                epoch_losses=pretrain_epoch_losses,
                scheduler=scheduler_pretrain,
                trans_pre_mean=trans_pre_mean,
                trans_pre_std=trans_pre_std,
                phase='Pre-Training Validation',
                transform_type=args.transform_type  
            )
    
            if dist.get_rank() == 0:
                current_runtime = time.time() - epoch_start_time if epoch_start_time else 0
                pretrain_runtime += current_runtime
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_pretrain.state_dict(),
                    'scheduler_state_dict': scheduler_pretrain.state_dict(),
                    'best_val_loss': best_val_loss_pretrain,
                    'epochs_no_improve': epochs_no_improve_pretrain,
                    'epoch_losses': pretrain_epoch_losses,
                    'runtime': pretrain_runtime
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f"pretrain_epoch_{epoch+1}.pth"))
    
                if val_loss < best_val_loss_pretrain:
                    best_val_loss_pretrain = val_loss
                    epochs_no_improve_pretrain = 0
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'best_val_loss': best_val_loss_pretrain
                    }, BEST_MODEL_PATH_PRETRAIN)
                    print(f"Validation loss improved to {best_val_loss_pretrain:.4f}. Saving best pretrain model.")
                else:
                    epochs_no_improve_pretrain += 1
                    print(f"No improvement in validation loss for {epochs_no_improve_pretrain} epoch(s).")
    
                if epochs_no_improve_pretrain >= EARLY_STOPPING_PATIENCE_PRETRAIN:
                    print(f"Early stopping triggered after {epoch+1} epochs of no improvement.")
                    break
    
            torch.distributed.barrier()
        
        if dist.get_rank() == 0:
            if pretrain_runtime > 0:
                hours = pretrain_runtime // 3600
                minutes = (pretrain_runtime % 3600) // 60
                seconds = pretrain_runtime % 60
                debug_print(f"Pre-Training Phase Completed at epoch {epoch+1}. Total runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
    
    torch.distributed.barrier()
    
    # ===========================
    # Fine-Tuning Phase
    # ===========================

    if dist.get_rank() == 0:
        if os.path.exists(BEST_MODEL_PATH_PRETRAIN):
            checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location='cpu', weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else {}
        else:
            state_dict = {}
    else:
        state_dict = {}
    
    state_dict_list = [state_dict]
    dist.broadcast_object_list(state_dict_list, src=0)
    state_dict = state_dict_list[0]
    
    if len(state_dict) > 0:
        model.module.load_state_dict(state_dict)
    
    torch.distributed.barrier()
    
    optimizer_finetune, mse_loss_fn_finetune, cc_loss_fn_finetune, alignment_loss_fn_finetune, scaler_finetune, scheduler_finetune = get_optimizer_and_loss(model, pretrain=False)
    finetune_epoch_losses = {'total': [], 'transformation': []}
    start_epoch = 0
    best_val_loss_finetune = float('inf')
    epochs_no_improve_finetune = 0
    finetune_runtime = 0.0
    finetune_start_time = time.time() if torch.cuda.is_available() else None
    
    if args.resume_finetune and os.path.exists(args.resume_finetune):
        debug_print(f"Resuming finetuning from checkpoint: {args.resume_finetune}")
        checkpoint = torch.load(args.resume_finetune, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer_finetune.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_finetune.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss_finetune = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve_finetune = checkpoint.get('epochs_no_improve', 0)
            finetune_epoch_losses = checkpoint.get('epoch_losses', {'total': [], 'transformation': []})
            finetune_runtime = checkpoint.get('runtime', 0.0)
            finetune_start_time = time.time() if torch.cuda.is_available() else None
        else:
            debug_print(f"Warning: Invalid checkpoint format in {args.resume_finetune}")
    elif finetune_exists and not args.force_train:
        debug_print(f"Found an existing best fine-tuned model at '{BEST_MODEL_PATH_FINETUNE}'. Skipping fine-tuning.")
        checkpoint = torch.load(BEST_MODEL_PATH_FINETUNE, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss_finetune = checkpoint.get('best_val_loss', float('inf'))
        else:
            debug_print(f"Warning: Invalid checkpoint format in {BEST_MODEL_PATH_FINETUNE}")
    else:
        debug_print("Starting Fine-Tuning Phase from scratch...")
    
    if args.resume_finetune or not finetune_exists or args.force_train:
        for epoch in range(start_epoch, FINETUNE_EPOCHS):
            epoch_start_time = time.time() if torch.cuda.is_available() else None
            debug_print(f"\nFine-Tuning Epoch {epoch+1}/{FINETUNE_EPOCHS}")
    
            train_loss = train_model(
                model=model,
                train_loader=finetune_train_loader,
                optimizer=optimizer_finetune,
                mse_loss_fn=mse_loss_fn_finetune,
                cc_loss_fn=cc_loss_fn_finetune,
                alignment_loss_fn=alignment_loss_fn_finetune,
                scaler=scaler_finetune,
                scheduler=scheduler_finetune,
                device=DEVICE,
                epoch_losses=finetune_epoch_losses,
                trans_pre_mean=trans_fine_mean,
                trans_pre_std=trans_fine_std,
                phase='Fine-Tuning',
                transform_type=args.transform_type  
            )
    
            val_loss = validate_model(
                model=model,
                valid_loader=finetune_valid_loader,
                mse_loss_fn=mse_loss_fn_finetune,
                cc_loss_fn=cc_loss_fn_finetune,
                alignment_loss_fn=alignment_loss_fn_finetune,
                device=DEVICE,
                epoch_losses=finetune_epoch_losses,
                scheduler=scheduler_finetune,
                trans_pre_mean=trans_fine_mean,
                trans_pre_std=trans_fine_std,
                phase='Fine-Tuning Validation',
                transform_type=args.transform_type  
            )
    
            if dist.get_rank() == 0:
                current_runtime = time.time() - epoch_start_time if epoch_start_time else 0
                finetune_runtime += current_runtime
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_finetune.state_dict(),
                    'scheduler_state_dict': scheduler_finetune.state_dict(),
                    'best_val_loss': best_val_loss_finetune,
                    'epochs_no_improve': epochs_no_improve_finetune,
                    'epoch_losses': finetune_epoch_losses,
                    'runtime': finetune_runtime
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f"finetune_epoch_{epoch+1}.pth"))
    
                if val_loss < best_val_loss_finetune:
                    best_val_loss_finetune = val_loss
                    epochs_no_improve_finetune = 0
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'best_val_loss': best_val_loss_finetune
                    }, BEST_MODEL_PATH_FINETUNE)
                    print(f"Validation loss improved to {best_val_loss_finetune:.4f}. Saving best finetune model.")
                else:
                    epochs_no_improve_finetune += 1
                    print(f"No improvement in validation loss for {epochs_no_improve_finetune} epoch(s).")
    
                if epochs_no_improve_finetune >= EARLY_STOPPING_PATIENCE_FINETUNE:
                    print(f"Early stopping triggered after {epoch+1} epochs of no improvement.")
                    break
    
            torch.distributed.barrier()
        
        if dist.get_rank() == 0:
            hours = finetune_runtime // 3600
            minutes = (finetune_runtime % 3600) // 60
            seconds = finetune_runtime % 60
            debug_print(f"Fine-Tuning Phase Completed. Total runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
    
    torch.distributed.barrier()

    # ===========================
    # Testing and Evaluation with FSC
    # ===========================
    if not (os.path.exists(BEST_MODEL_PATH_FINETUNE) or os.path.exists(BEST_MODEL_PATH_PRETRAIN)):
        print(f"GPU{dist.get_rank()}: No valid checkpoint found. Evaluation aborted.")
        return
    
    if os.path.exists(BEST_MODEL_PATH_FINETUNE):
        checkpoint = torch.load(BEST_MODEL_PATH_FINETUNE, map_location=f'cuda:{args.local_rank}', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            print(f"GPU{dist.get_rank()}: Invalid checkpoint format in fine-tuned model")
            return
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print(f"GPU{dist.get_rank()}: Loaded the best fine-tuned model from checkpoint for testing.")
    else:
        checkpoint = torch.load(BEST_MODEL_PATH_PRETRAIN, map_location=f'cuda:{args.local_rank}', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            print(f"GPU{dist.get_rank()}: Invalid checkpoint format in pre-trained model")
            return
        model.module.load_state_dict(checkpoint['model_state_dict'])
        print(f"GPU{dist.get_rank()}: Loaded the best pre-trained model from checkpoint for testing.")
    
    fsc_results_dir = os.path.join(args.output_dir, 'fsc_results')
    if dist.get_rank() == 0:
        os.makedirs(fsc_results_dir, exist_ok=True)
        print(f"Created FSC results directory: {fsc_results_dir}")
    
    for idx, test_loader in enumerate(test_loaders):
        test_file_name = f"test_tomogram_{idx+1}"
        if idx < len(TEST_FILES):
            test_file_name = os.path.basename(TEST_FILES[idx])
        
        torch.cuda.empty_cache()
        
        try:
            predicted_trans_test, ground_truth_trans_test, aligned_volumes_test, target_volumes_test = test_model(
                model=model,
                test_loader=test_loader,
                device=torch.device(f'cuda:{args.local_rank}'),
                trans_fine_mean=trans_fine_mean.to(f'cuda:{args.local_rank}'),
                trans_fine_std=trans_fine_std.to(f'cuda:{args.local_rank}'),
                test_file_name=test_file_name,
                transform_type=args.transform_type  
            )
    
            torch.distributed.barrier()
    
            if dist.get_rank() == 0 and predicted_trans_test is not None:
                evaluate_model(
                    predicted_trans=predicted_trans_test,
                    ground_truth_trans=ground_truth_trans_test,
                    aligned_volumes=aligned_volumes_test,
                    target_volumes=target_volumes_test,
                    image_size=32,
                    test_file_name=test_file_name,
                    transform_type=args.transform_type,
                    pixel_size=args.pixel_size,
                    fsc_threshold=args.fsc_threshold
                )
                
                if wandb.run is not None:
                    for filename in os.listdir(fsc_results_dir):
                        if filename.startswith(test_file_name) and filename.endswith('.png'):
                            file_path = os.path.join(fsc_results_dir, filename)
                            wandb.log({f"FSC/{filename}": wandb.Image(file_path)})
                    
                    summary_file = os.path.join(fsc_results_dir, f"{test_file_name}_fsc_summary.txt")
                    if os.path.exists(summary_file):
                        with open(summary_file, 'r') as f:
                            for line in f:
                                if "Average Resolution:" in line:
                                    try:
                                        avg_res = float(line.split(':')[1].split()[0])
                                        wandb.log({f"FSC/{test_file_name}_avg_resolution": avg_res})
                                    except:
                                        pass
                        wandb.save(summary_file)
        
        except Exception as e:
            print(f"Error processing test set {test_file_name}: {e}")
            traceback.print_exc()
            continue
    
        torch.distributed.barrier()
        
    if dist.get_rank() == 0:
        print("\n===== FSC Evaluation Summary =====")
        all_avg_resolutions = []
        for filename in os.listdir(fsc_results_dir):
            if filename.endswith('_fsc_summary.txt'):
                with open(os.path.join(fsc_results_dir, filename), 'r') as f:
                    for line in f:
                        if "Average Resolution:" in line:
                            try:
                                avg_res = float(line.split(':')[1].split()[0])
                                tomogram_name = filename.split('_fsc_summary.txt')[0]
                                all_avg_resolutions.append((tomogram_name, avg_res))
                            except:
                                pass
        
        if all_avg_resolutions:
            print("\nFSC Resolution by Tomogram:")
            for name, res in all_avg_resolutions:
                print(f"{name}: {res:.2f} Å")
            
            overall_avg = sum(res for _, res in all_avg_resolutions) / len(all_avg_resolutions)
            print(f"\nOverall Average FSC Resolution: {overall_avg:.2f} Å")
            
            with open(os.path.join(fsc_results_dir, 'overall_fsc_summary.txt'), 'w') as f:
                f.write("Overall FSC Evaluation Summary\n")
                f.write(f"Pixel Size: {args.pixel_size} Å\n")
                f.write(f"FSC Threshold: {args.fsc_threshold}\n\n")
                
                for name, res in all_avg_resolutions:
                    f.write(f"{name}: {res:.2f} Å\n")
                
                f.write(f"\nOverall Average Resolution: {overall_avg:.2f} Å\n")
            
            if wandb.run is not None:
                wandb.log({"FSC/overall_avg_resolution": overall_avg})
                wandb.save(os.path.join(fsc_results_dir, 'overall_fsc_summary.txt'))
        
        print("All processes completed successfully.")

if __name__ == "__main__":
    init_distributed()
    main()

