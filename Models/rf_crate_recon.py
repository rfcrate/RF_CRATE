import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
# Import the original RF-CRATE model from the same directory
from .rf_crate import (RF_CRATE, ComplexLayerNorm, CReLU, ZReLU, ModReLU, 
                      ComplexCardioid, ComplexDropout, pair)

class RF_CRATE_Recon(RF_CRATE):
    """
    Enhanced RF-CRATE model that exposes intermediate features for reconstruction
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x):
        # Extract patch embeddings
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # Add class token and positional embeddings
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Store intermediate features
        features = []
        for i, (attn, ff) in enumerate(self.transformer.layers):
            grad_x = attn(x) + x
            x = ff(grad_x)
            features.append(x)
        
        # Continue with classification
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        feature_last = x
        output = self.mlp_head(x)
        
        # Return both classification output and intermediate features
        return output, features, feature_pre 

class ComplexUpsample(nn.Module):
    """Custom upsampling module for complex tensors"""
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        # Split into real and imaginary parts
        real = x.real
        imag = x.imag
        
        # Handle tuple scale factors for different H/W scaling
        if isinstance(self.scale_factor, tuple):
            real_up = F.interpolate(real, scale_factor=self.scale_factor, 
                                  mode=self.mode, align_corners=self.align_corners)
            imag_up = F.interpolate(imag, scale_factor=self.scale_factor,
                                  mode=self.mode, align_corners=self.align_corners)
        else:
            real_up = F.interpolate(real, scale_factor=self.scale_factor, 
                                  mode=self.mode, align_corners=self.align_corners)
            imag_up = F.interpolate(imag, scale_factor=self.scale_factor,
                                  mode=self.mode, align_corners=self.align_corners)
        
        # Recombine into complex tensor
        return torch.complex(real_up, imag_up)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return torch.complex(
            self.real_conv(x.real),
            self.imag_conv(x.imag)
        )

class SimpleDecoder(nn.Module):
    def __init__(self, depth, patch_size, image_size, in_channels, embedding_dim, patch_embedding_method='linear_patch'):
        super().__init__()
        self.depth = depth
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        self.patch_size = pair(patch_size)
        self.image_size = pair(image_size)
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.patch_embedding_method = patch_embedding_method
        
        # Calculate number of patches
        self.patch_height, self.patch_width = self.patch_size
        self.image_height, self.image_width = self.image_size
        
        if patch_embedding_method == 'linear_patch':
            self.h_patches = self.image_height // self.patch_height
            self.w_patches = self.image_width // self.patch_width
        elif patch_embedding_method == 'conv_patch' or patch_embedding_method == 'group_conv_patch':
            self.h_patches = self.image_height // self.patch_height
            self.w_patches = self.image_width // self.patch_width
        elif patch_embedding_method == 'soft_conv_patch':
            self.h_patches = self.image_height // (self.patch_height // 2)
            self.w_patches = self.image_width // (self.patch_width // 2)
        elif patch_embedding_method == 'conv_module_patch':
            # For the CNN module with 3 layers of stride 2
            self.h_patches = self.image_height // (2**3)
            self.w_patches = self.image_width // (2**3)
        else:
            raise ValueError(f"Unsupported patch embedding method: {patch_embedding_method}")
        
        # Build decoder layers
        self.build_decoder()
        
    def build_decoder(self):
        # Transformer feature to 2D conversion - one per transformer layer
        self.transformer_to_2d_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim, dtype=torch.cfloat),
                CReLU()
            ) for _ in range(self.depth)
        ])
        
        # Determine number of expected features from transformer
        self.expected_num_features = self.depth  # Adjust based on model depth
        
        # Feature dimension reduction - one per transformer layer
        self.feature_reduction_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 4, dtype=torch.cfloat),
                CReLU()
            ) for _ in range(self.depth)
        ])
        
        # Calculate input channels for the first transposed conv
        reduced_dim = self.embedding_dim // 4
        concat_channels = reduced_dim * self.expected_num_features
        
        # Calculate upsampling factors needed to reach original image size
        height_scale = self.image_height // self.h_patches
        width_scale = self.image_width // self.w_patches
        
        # Dynamic upsampling layers based on the required scales
        layers = []
        
        # Add appropriate number of transposed convolution layers to reach target dimensions
        current_h, current_w = self.h_patches, self.w_patches
        channels = [concat_channels, 256, 128, 64, self.in_channels]
        
        # First layers handle major upsampling
        scale_steps_h = self._get_scale_steps(current_h, self.image_height)
        scale_steps_w = self._get_scale_steps(current_w, self.image_width)
        
        # Create upsampling blocks
        for i in range(len(channels) - 1):
            # Determine if we need to scale in this step
            if i < len(scale_steps_h):
                h_scale = scale_steps_h[i]
                w_scale = scale_steps_w[i] if i < len(scale_steps_w) else 1
                
                # Use stride > 1 for upsampling when needed
                stride_h = 2 if h_scale > 1 else 1
                stride_w = 2 if w_scale > 1 else 1
                
                # Create appropriate kernel sizes
                if stride_h > 1 or stride_w > 1:
                    kernel_h = 4 if stride_h > 1 else 3
                    kernel_w = 4 if stride_w > 1 else 3
                    
                    # Create custom ConvTranspose2d that can handle different H/W scaling
                    layers.append(ComplexConvTranspose2d(
                        channels[i], channels[i+1], 
                        kernel_size=(kernel_h, kernel_w),
                        stride=(stride_h, stride_w),
                        padding=1
                    ))
                    current_h *= stride_h
                    current_w *= stride_w
                else:
                    # Regular convolution when not upsampling
                    layers.append(ComplexConvTranspose2d(
                        channels[i], channels[i+1], 
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ))
                
                # Add activation except for the last layer
                if i < len(channels) - 2:
                    layers.append(CReLU())
            
            else:
                # Regular convolution when we've done enough upsampling
                layers.append(ComplexConvTranspose2d(
                    channels[i], channels[i+1], 
                    kernel_size=3,
                    stride=1,
                    padding=1
                ))
                if i < len(channels) - 2:
                    layers.append(CReLU())
        
        # Final adjustment layer if we still need to match exact dimensions
        if current_h < self.image_height or current_w < self.image_width:
            layers.append(ComplexUpsample(
                scale_factor=(self.image_height / current_h, self.image_width / current_w),
                mode='bilinear',
                align_corners=False
            ))
        
        self.upsample_blocks = nn.Sequential(*layers)
    
    def _get_scale_steps(self, current_dim, target_dim):
        """Calculate efficient upsampling steps to reach target dimension"""
        scale_steps = []
        while current_dim < target_dim:
            # Use power-of-2 scaling where possible
            if current_dim * 2 <= target_dim:
                scale_steps.append(2)
                current_dim *= 2
            else:
                # Use exact scaling for final step if needed
                scale_steps.append(target_dim / current_dim)
                break
        return scale_steps
    
    def forward(self, features):
        # Process each feature to 2D using layer-specific transformations
        processed_features = []
        
        for idx, feat in enumerate(features):
            # Remove class token
            feat = feat[:, 1:, :]
            # Transform to 2D using layer-specific transformation
            feat_2d = self.transformer_to_2d_modules[idx](feat)
            feat_2d = feat_2d.reshape(feat_2d.shape[0], self.h_patches, self.w_patches, self.embedding_dim)
            feat_2d = feat_2d.permute(0, 3, 1, 2)  # [B, C, H, W]
            # Reduce feature dimension using layer-specific reduction
            feat_2d = self.feature_reduction_modules[idx](feat_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            processed_features.append(feat_2d)
        
        # Concatenate all features along channel dimension
        x = torch.cat(processed_features, dim=1)
        
        # Upsample to original image size
        x = self.upsample_blocks(x)
        
        return x

def rf_crate_recon_tiny(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', 
                        relu_type='crelu', patch_embedding_method='linear_patch',
                        mlp_head_type='crate_version', output_type='magnitude'):
    """Create a tiny RF-CRATE model with reconstruction capability"""
    model = RF_CRATE_Recon(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
    )
    
    decoder = SimpleDecoder(
        depth=model.transformer.depth,
        patch_size=patch_size,
        image_size=image_size,
        in_channels=in_channels,
        embedding_dim=384,
        patch_embedding_method=patch_embedding_method
    )
    
    return model, decoder

def rf_crate_recon_small(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', 
                         relu_type='crelu', patch_embedding_method='linear_patch',
                         mlp_head_type='crate_version', output_type='magnitude'):
    """Create a small RF-CRATE model with reconstruction capability"""
    model = RF_CRATE_Recon(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
    )
    
    decoder = SimpleDecoder(
        depth=model.transformer.depth,
        patch_size=patch_size,
        image_size=image_size,
        in_channels=in_channels,
        embedding_dim=576,
        patch_embedding_method=patch_embedding_method
    )
    
    return model, decoder

def rf_crate_recon_base(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', 
                        relu_type='crelu', patch_embedding_method='linear_patch',
                        mlp_head_type='crate_version', output_type='magnitude'):
    """Create a base RF-CRATE model with reconstruction capability"""
    model = RF_CRATE_Recon(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
    )
    
    decoder = SimpleDecoder(
        depth=model.transformer.depth,
        patch_size=patch_size,
        image_size=image_size,
        in_channels=in_channels,
        embedding_dim=768,
        patch_embedding_method=patch_embedding_method
    )
    
    return model, decoder

def rf_crate_recon_large(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', 
                         relu_type='crelu', patch_embedding_method='linear_patch',
                         mlp_head_type='crate_version', output_type='magnitude'):
    """Create a large RF-CRATE model with reconstruction capability"""
    model = RF_CRATE_Recon(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
    )
    
    decoder = SimpleDecoder(
        depth=model.transformer.depth,
        patch_size=patch_size,
        image_size=image_size,
        in_channels=in_channels,
        embedding_dim=1024,
        patch_embedding_method=patch_embedding_method
    )
    
    return model, decoder
