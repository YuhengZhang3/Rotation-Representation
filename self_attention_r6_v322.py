import torch
import torch.nn as nn
import torch.nn.functional as F

def r6_to_matrix(r6):
    if r6.size(-1) != 6:
        raise ValueError(f"r6_to_matrix expects last dimension=6, got {r6.size(-1)}")
    with torch.autocast(device_type='cuda', enabled=False):
        r6 = r6.float()
        eps = 1e-7
    
        v1 = torch.nn.functional.normalize(r6[..., :3], dim=-1, eps=eps)
        v2 = r6[..., 3:6]
        v2_proj = torch.sum(v2 * v1, dim=-1, keepdim=True) * v1
        v2 = torch.nn.functional.normalize(v2 - v2_proj, dim=-1, eps=eps)
        v3 = torch.cross(v1, v2, dim=-1)
        R = torch.stack([v1, v2, v3], dim=-1)

    return R

def matrix_to_r6(R):
    v1 = R[..., 0, :]
    v2 = R[..., 1, :]
    return torch.cat([v1, v2], dim=-1)

def apply_rotation_adaptive(X, R, feature_type='vector'):
    # Ensure both tensors have the same dtype
    if X.dtype != R.dtype:
        R = R.to(dtype=X.dtype)
        
    if feature_type == 'vector':
        orig_shape = X.shape
        orig_dim = X.shape[-1]
        
        if orig_dim % 3 == 0:
            num_vectors = orig_dim // 3
            X_reshaped = X.view(*orig_shape[:-1], num_vectors, 3)
            L = X_reshaped.size(-3)
            R_expanded = R.unsqueeze(2).unsqueeze(3)
            R_expanded = R_expanded.expand(-1, -1, L, num_vectors, -1, -1)
            
            X_rotated = torch.matmul(X_reshaped.unsqueeze(-2), R_expanded.transpose(-2, -1)).squeeze(-2)
            
            return X_rotated.reshape(*orig_shape[:-1], -1)

        else:
            pad_size = (3 - (orig_dim % 3)) % 3
            X_padded = F.pad(X, (0, pad_size)) if pad_size > 0 else X

            num_vectors = X_padded.size(-1) // 3
            X_reshaped = X_padded.view(*X_padded.shape[:-1], num_vectors, 3)
            
            L = X_reshaped.size(-3)
            R_expanded = R.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, num_vectors, -1, -1)
            
            X_rotated = torch.matmul(
                X_reshaped.unsqueeze(-2),
                R_expanded.transpose(-2, -1)
            ).squeeze(-2)
            
            out = X_rotated.reshape(*orig_shape[:-1], -1)[..., :orig_dim]
            return out
    return X

def apply_translation(p, t):
    return p + t

def random_valid_r6(num_heads):
    r6 = torch.randn(1, num_heads, 6)  
    v1 = F.normalize(r6[..., :3], dim=-1)
    v2 = r6[..., 3:6]
    v2_proj = (v2 * v1).sum(dim=-1, keepdim=True) * v1
    v2 = F.normalize(v2 - v2_proj, dim=-1)

    return torch.cat([v1, v2], dim=-1)

class PositionalProcessing(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        self.local_processor = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 1)
        )
        
        self.modulation = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.pos_attention = nn.Sequential(
            nn.Conv3d(hidden_dim, num_heads, 1),
            nn.InstanceNorm3d(num_heads),
            nn.GELU()
        )
    
    def forward(self, x, pos_emb):
        local_feat = self.local_processor(pos_emb)
        mod_factors = self.modulation(local_feat)
        x_mod = x * mod_factors
        pos_attn = self.pos_attention(pos_emb)
        return x_mod, pos_attn

class AxialSelfAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads, feature_type='vector', max_relative_position=64):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.feature_type = feature_type
        self.max_relative_position = max_relative_position

        self.qkv = nn.Linear(in_channels, 3 * in_channels)
        
        self.pos_processor = PositionalProcessing(in_channels, num_heads)
        
        self.R6_d = nn.Parameter(torch.randn(1, num_heads, 6) * 0.01)
        self.R6_h = nn.Parameter(torch.randn(1, num_heads, 6) * 0.01)
        self.R6_w = nn.Parameter(torch.randn(1, num_heads, 6) * 0.01)
        
        self.t_d = nn.Parameter(torch.zeros(1, num_heads, 1, 3))  
        self.t_h = nn.Parameter(torch.zeros(1, num_heads, 1, 3))
        self.t_w = nn.Parameter(torch.zeros(1, num_heads, 1, 3))
        
        self.R6_d = nn.Parameter(random_valid_r6(num_heads))
        self.R6_h = nn.Parameter(random_valid_r6(num_heads))
        self.R6_w = nn.Parameter(random_valid_r6(num_heads))
        
        for t_param in [self.t_d, self.t_h, self.t_w]:
            nn.init.normal_(t_param, std=0.01)

        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.pos_bias_layer = nn.Linear(3, 1)

    def _axial_attention(self, x, pos_emb, axis, R6, t):
        B, C, D, H, W = x.shape
        x_mod, pos_attn = self.pos_processor(x, pos_emb)
        
        if axis == 'depth':
            x_perm = x_mod.permute(0, 3, 4, 1, 2).contiguous()
            if pos_attn is not None:
                batch_dim = B * H * W
            L = D
        elif axis == 'height':
            x_perm = x_mod.permute(0, 2, 4, 1, 3).contiguous()
            if pos_attn is not None:
                batch_dim = B * D * W
            L = H
        else:  # width
            x_perm = x_mod.permute(0, 2, 3, 1, 4).contiguous()
            if pos_attn is not None:
                batch_dim = B * D * H
            L = W
        
        x_perm = x_perm.reshape(batch_dim, C, L)
        pos = self._get_axis_positions(L, axis, x.device)
        qkv = self.qkv(x_perm.transpose(-1, -2))
        qkv = qkv.reshape(batch_dim, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        
        pos = pos.unsqueeze(0).unsqueeze(1).expand(batch_dim, self.num_heads, L, 3)
        R = r6_to_matrix(R6).expand(batch_dim, -1, -1, -1)
        t = t.expand(batch_dim, -1, L, -1)
        
        q_rot = apply_rotation_adaptive(q, R, self.feature_type)
        k_rot = apply_rotation_adaptive(k, R, self.feature_type)
        
        attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        
        if pos_attn is not None:
            pos_attn = pos_attn.reshape(batch_dim, self.num_heads, L)  
            pos_attn = pos_attn.unsqueeze(-1).expand(-1, -1, -1, L)   
            attn = attn + pos_attn
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_dim, L, -1)
        
        if axis == 'depth':
            out = out.reshape(B, H, W, D, -1).permute(0, 4, 3, 1, 2)
        elif axis == 'height':
            out = out.reshape(B, D, W, H, -1).permute(0, 4, 1, 3, 2)
        else:  # width
            out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        
        return out

    def forward(self, x, pos_emb):
        out_d = self._axial_attention(x, pos_emb, 'depth', self.R6_d, self.t_d)
        out_h = self._axial_attention(x, pos_emb, 'height', self.R6_h, self.t_h)
        out_w = self._axial_attention(x, pos_emb, 'width', self.R6_w, self.t_w)
        
        out_sum = out_d + out_h + out_w
        out = self.proj(out_sum)
        return out

    def _get_axis_positions(self, length, axis, device):
        pos = torch.linspace(-1, 1, steps=length, device=device)
        if axis == 'depth':
            return torch.stack([torch.zeros_like(pos), torch.zeros_like(pos), pos], dim=-1)
        elif axis == 'height':
            return torch.stack([torch.zeros_like(pos), pos, torch.zeros_like(pos)], dim=-1)
        else:  # width
            return torch.stack([pos, torch.zeros_like(pos), torch.zeros_like(pos)], dim=-1)

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads, feature_type='vector'):
        super().__init__()
        self.attention = AxialSelfAttentionModule(
            in_channels=in_channels,
            num_heads=num_heads,
            feature_type=feature_type
        )
    
    def forward(self, x, pos_emb, mask=None):  
        return self.attention(x, pos_emb)

from torch.testing import assert_close

# =============================================================================
# Tests
# =============================================================================
class PositionalProcessingTests:
    @staticmethod
    def test_shape_consistency(device):
        print("\n=== Testing PositionalProcessing shape consistency ===")
        hidden_dim = 64
        num_heads = 4
        processor = PositionalProcessing(hidden_dim, num_heads).to(device).eval()
        
        B, C, D, H, W = 1, hidden_dim, 32, 32, 32
        x = torch.randn(B, C, D, H, W, device=device)
        pos_emb = torch.randn(B, C, D, H, W, device=device)
        
        with torch.no_grad():
            x_mod, pos_attn = processor(x, pos_emb)
        
        assert x_mod.shape == x.shape, f"Modified input shape mismatch: {x_mod.shape} vs {x.shape}"
        assert pos_attn.shape == (B, num_heads, D, H, W), f"Position attention shape mismatch"
        print("Shape consistency test passed!")

    @staticmethod
    def test_modulation_range(device):
        print("\n=== Testing PositionalProcessing modulation range ===")
        hidden_dim = 64
        num_heads = 4
        processor = PositionalProcessing(hidden_dim, num_heads).to(device).eval()
        
        B, C, D, H, W = 1, hidden_dim, 32, 32, 32
        x = torch.randn(B, C, D, H, W, device=device)
        pos_emb = torch.randn(B, C, D, H, W, device=device)

        with torch.no_grad():
            x_mod, pos_attn = processor(x, pos_emb)
        
        local_feat = processor.local_processor(pos_emb)
        mod_factors = processor.modulation(local_feat)
        
        assert (mod_factors >= 0).all(), "Some mod_factors < 0!"
        assert (mod_factors <= 1).all(), "Some mod_factors > 1!"
        print("Modulation range test passed!")

class AxialSelfAttentionTests:
    @staticmethod 
    def test_rotation_matrix_validity(device):
        print("\n=== Testing rotation matrix validity ===")
        in_channels = 64
        num_heads = 4
        attention = AxialSelfAttentionModule(in_channels, num_heads).to(device).eval()
        
        eps = 1e-5
        for param_name in ['R6_d', 'R6_h', 'R6_w']:
            param = getattr(attention, param_name)
            assert param.size(-1) == 6, f"Parameter {param_name} has wrong size {param.size(-1)}"

            v1 = F.normalize(param[..., :3], dim=-1)
            v2_raw = param[..., 3:]
            v2_orth = v2_raw - torch.sum(v2_raw * v1, dim=-1, keepdim=True) * v1
            v2 = F.normalize(v2_orth, dim=-1)

            param_valid = torch.cat([v1, v2], dim=-1)
            R = r6_to_matrix(param_valid)

            det = torch.linalg.det(R)
            assert_close(det, torch.ones_like(det), rtol=eps, atol=eps)

            RRT = torch.matmul(R, R.transpose(-2, -1))
            identity = torch.eye(3, device=device, dtype=RRT.dtype).expand(RRT.shape)
            assert_close(RRT, identity, rtol=eps, atol=eps)

            v_dot = torch.sum(v1 * v2, dim=-1)
            assert_close(v_dot, torch.zeros_like(v_dot), rtol=eps, atol=eps)
            
            v1_norm = torch.norm(v1, dim=-1)
            v2_norm = torch.norm(v2, dim=-1)
            assert_close(v1_norm, torch.ones_like(v1_norm), rtol=eps, atol=eps)
            assert_close(v2_norm, torch.ones_like(v2_norm), rtol=eps, atol=eps)
        
        print("Rotation matrix validity test passed!")

    @staticmethod
    def test_axial_equivariance(device):
        print("\n=== Testing axial attention equivariance ===")
        in_channels = 64 
        num_heads = 4
        attention = AxialSelfAttentionModule(in_channels, num_heads).to(device).eval()
       
        B, C, D, H, W = 1, in_channels, 16, 16, 16
        x = torch.randn(B, C, D, H, W, device=device)
        pos_emb = torch.randn(B, C, D, H, W, device=device)
       
        shift_d = 2
        x_shifted = torch.roll(x, shifts=shift_d, dims=2)
        pos_emb_shifted = torch.roll(pos_emb, shifts=shift_d, dims=2)
       
        with torch.no_grad():
            out1 = attention._axial_attention(x, pos_emb, 'depth', attention.R6_d, attention.t_d)
            out2 = attention._axial_attention(x_shifted, pos_emb_shifted, 'depth', attention.R6_d, attention.t_d)
           
        out1_shifted = torch.roll(out1, shifts=shift_d, dims=2)
        diff = (out2 - out1_shifted).abs().mean().item()
        print(f"Depth-axis equivariance diff: {diff:.6f}")

    @staticmethod
    def test_attention_output_distribution(device):
        print("\n=== Testing attention output distribution ===")
        in_channels = 64
        num_heads = 4
        attention = AxialSelfAttentionModule(in_channels, num_heads).to(device).eval()
       
        B, C, D, H, W = 1, in_channels, 16, 16, 16
        x = torch.randn(B, C, D, H, W, device=device)
        pos_emb = torch.randn(B, C, D, H, W, device=device)
       
        with torch.no_grad():
            output = attention(x, pos_emb)
       
        mean = output.mean().item()
        std = output.std().item()
        print(f"Output statistics - Mean: {mean:.6f}, Std: {std:.6f}")
        assert abs(mean) < 1.0, f"Mean too large: {mean}"
        assert 0.01 < std < 10.0, f"Std out of range: {std}"

    @staticmethod
    def test_large_scale_stability(device):
        print("\n=== Testing large scale stability ===")
        in_channels = 64
        num_heads = 4
        attention = AxialSelfAttentionModule(in_channels, num_heads).to(device).eval()
       
        B, C, D, H, W = 2, in_channels, 16, 16, 16
        x_large = torch.randn(B, C, D, H, W, device=device) * 1e5
        pos_emb_large = torch.randn(B, C, D, H, W, device=device) * 1e5
       
        with torch.no_grad():
            for axis in ['depth', 'height', 'width']:
                out = attention._axial_attention(
                    x_large, pos_emb_large, axis,
                    getattr(attention, f'R6_{axis[0]}'),  # R6_d/h/w
                    getattr(attention, f't_{axis[0]}')    # t_d/h/w
                )
                assert torch.all(torch.isfinite(out)), f"Non-finite values in {axis} attention output"
        print("Large scale stability test passed!")

    @staticmethod
    def test_gradient_flow(device):
        print("\n=== Testing gradient flow ===")
        B, C, D, H, W = 2, 64, 16, 16, 16
        num_heads = 4
        attention = AxialSelfAttentionModule(in_channels=C, num_heads=num_heads).to(device)
       
        x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
        pos_emb = torch.randn(B, C, D, H, W, device=device)
       
        out = attention(x, pos_emb)
        loss = out.sum()
        loss.backward()
       
        assert x.grad is not None, "No gradients propagated to input"
        assert torch.all(torch.isfinite(x.grad)), "Non-finite gradients"
        assert torch.any(x.grad != 0), "Zero gradients"
        print("Gradient flow test passed!")

    @staticmethod
    def test_shape_consistency_all_axes(device):
        print("\n=== Testing shape consistency across all axes ===")
        B, C, D, H, W = 2, 64, 16, 16, 16
        num_heads = 4
        module = AxialSelfAttentionModule(C, num_heads).to(device).eval()
       
        x = torch.randn(B, C, D, H, W, device=device)
        pos_emb = torch.randn(B, C, D, H, W, device=device)
       
        with torch.no_grad():
            for axis in ['depth', 'height', 'width']:
                out = module._axial_attention(
                    x, pos_emb, axis,
                    getattr(module, f'R6_{axis[0]}'),  # R6_d/h/w
                    getattr(module, f't_{axis[0]}')    # t_d/h/w
                )
                assert out.shape == (B, C, D, H, W), f"Shape mismatch for {axis} attention"
        print("Shape consistency test passed for all axes!")

    @staticmethod
    def test_position_encoding_range(device):
        print("\n=== Testing position encoding range ===")
        module = AxialSelfAttentionModule(64, 8).to(device)
        length = 16
       
        for axis in ['depth', 'height', 'width']:
            pos = module._get_axis_positions(length, axis, device)
            assert pos.shape == (length, 3), f"Position shape wrong for {axis}"
            assert torch.all((pos >= -1) & (pos <= 1)), f"Position values out of range for {axis}"
           
            if axis == 'depth':
                assert_close(pos[:, :2], torch.zeros_like(pos[:, :2]))
            elif axis == 'height':
                assert_close(pos[:, [0,2]], torch.zeros_like(pos[:, [0,2]]))
            else:
                assert_close(pos[:, 1:], torch.zeros_like(pos[:, 1:]))
        print("Position encoding range test passed!")


def run_attention_tests():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning attention tests on device: {device}")
    
    print("\n=== Running PositionalProcessing Tests ===")
    PositionalProcessingTests.test_shape_consistency(device)
    PositionalProcessingTests.test_modulation_range(device)
    
    print("\n=== Running AxialSelfAttention Tests ===")
    AxialSelfAttentionTests.test_rotation_matrix_validity(device)
    AxialSelfAttentionTests.test_axial_equivariance(device)
    AxialSelfAttentionTests.test_attention_output_distribution(device)
    AxialSelfAttentionTests.test_large_scale_stability(device)
    AxialSelfAttentionTests.test_gradient_flow(device)
    AxialSelfAttentionTests.test_shape_consistency_all_axes(device)
    AxialSelfAttentionTests.test_position_encoding_range(device)
    
    print("\nAll attention tests completed successfully.")

if __name__ == "__main__":
    run_attention_tests()