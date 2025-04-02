import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import math

init_method = 'unitary'  # 'kaiming_uniform', 'gorot','wirtinger', 'unitary'
layer_norm_method = 'layer_norm_naive'  # 'layer_norm_naive', 'layer_norm_joint', 'dynamic_tanh'

def complex_glorot_uniform(weight):
    fan_in, fan_out = weight.shape
    scale = np.sqrt(6.0 / (fan_in + fan_out)) / np.sqrt(2)  # Divide by sqrt(2) for complex values
    
    # Initialize real and imaginary parts
    init.uniform_(weight.real, -scale, scale)
    init.uniform_(weight.imag, -scale, scale)
    return weight

def complex_unitary_initialization(weight):
    rows, cols = weight.shape
    size = min(rows, cols)
    
    # Create a random complex matrix
    q = torch.randn(size, size, dtype=torch.cfloat)
    
    # Perform QR decomposition on the complex matrix
    u, _ = torch.linalg.qr(q)  # QR returns Q and R, we just need Q
    
    # Pad with zeros if necessary
    if rows > cols:
        weight.data[:size, :] = u
        weight.data[size:, :] = 0
    elif cols > rows:
        weight.data[:, :size] = u
        weight.data[:, size:] = 0
    else:
        weight.data = u
    
    return weight

def wirtinger_initialization(weight):
    fan_in, fan_out = weight.shape
    scale = np.sqrt(1.0 / fan_in)
    
    # Initialize magnitude with Rayleigh distribution
    magnitude = torch.randn(weight.shape).abs() * scale
    
    # Initialize phase uniformly in [0, 2π)
    phase = torch.rand(weight.shape) * 2 * np.pi
    
    # Convert to complex
    weight.data = torch.polar(magnitude, phase)
    return weight



class ModReLU(nn.Module):
    def __init__(self, c: float = 1e-3):
        '''
        Martin Arjovsky, Amar Shah, and Yoshua Bengio. 2016. Unitary Evolution Recurrent Neural Networks. arXiv:1511.06464 [cs, stat
        '''
        super(ModReLU, self).__init__()
        self.b = nn.Parameter(torch.tensor(0.5))  # Initialize b with a default value, e.g., 0.5
        self.c = c

    def forward(self, z):
        modulus = torch.abs(z)
        scale = F.relu(modulus + self.b) / (modulus + self.c)
        return scale * z

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, z):
        real_relu = F.relu(z.real)
        imag_relu = F.relu(z.imag)
        return torch.complex(real_relu, imag_relu)
    
class ZReLU(nn.Module):
    def __init__(self):
        '''
        Nitzan Guberman. 2016. On Complex Valued Convolutional Neural Networks. arXiv:1602.09046 [cs]
        '''
        super(ZReLU, self).__init__()

    def forward(self, z):
        phase = torch.angle(z)
        mask = (phase >= 0) & (phase <= torch.pi / 2)
        return z * mask

class ComplexCardioid(nn.Module):
    def __init__(self):
        '''
        Patrick Virtue, Stella X. Yu, and Michael Lustig. 2017. Better than Real: Complex-valued Neural Nets for MRI Fingerprinting. In 2017
        IEEE International Conference on Image Processing (ICIP). 3953–3957. https://doi.org/10.1109/ICIP.2017.8297024
        '''
        super(ComplexCardioid, self).__init__()

    def forward(self, z):
        phase = torch.angle(z)
        scale = (1 + torch.cos(phase)) / 2
        return scale * z


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.real_LayerNorm = nn.LayerNorm(self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine)
        self.imag_LayerNorm = nn.LayerNorm(self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine)

    def forward(self, input):
        return self.real_LayerNorm(input.real) + 1j * self.imag_LayerNorm(input.imag)  # torch.complex(self.real_LayerNorm(input.real), self.imag_LayerNorm(input.imag)) not used because of memory requirements

class ComplexDynamicTanh(nn.Module):
    # Zhu, Jiachen, Xinlei Chen, Kaiming He, Yann LeCun and Zhuang Liu. "Transformers without Normalization." (2025).
    def __init__(self, normalized_shape, alpha_init_value=0.5, channels_last=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.cfloat) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.cfloat))

    def forward(self, x):
        # Complex tanh: apply tanh to real and imaginary parts separately
        real_part = torch.tanh(self.alpha.real * x.real - self.alpha.imag * x.imag)
        imag_part = torch.tanh(self.alpha.real * x.imag + self.alpha.imag * x.real)
        x = torch.complex(real_part, imag_part)
        
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class ComplexLayerNorm2(nn.Module):
    # Eilers, Florian, and Xiaoyi Jiang. "Building blocks for a complex-valued transformer architecture." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            if isinstance(normalized_shape, int):
                self.embed_dim = normalized_shape
            else:
                self.embed_dim = normalized_shape[-1]  # Take last dimension for compatibility
                
            # Parameters for controlling the normalization
            self.weights = nn.Parameter((torch.tensor([1.0, 1.0, 0.0]).repeat([self.embed_dim, 1])).unsqueeze(-1))
            self.bias = nn.Parameter(torch.zeros([1, 1, self.embed_dim], dtype=torch.cfloat))

    def forward(self, input):
        # Calculate mean
        ev = torch.unsqueeze(torch.mean(input, dim=-1), dim=-1)
        
        # Calculate variances
        var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=-1), dim=-1), dim=-1)
        var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=-1), dim=-1), dim=-1)
        
        # Center the input
        centered_input = input - ev
        
        # Calculate covariance between real and imaginary parts
        cov = torch.unsqueeze(torch.unsqueeze(torch.mean(centered_input.real * centered_input.imag, dim=-1), dim=-1), dim=-1)
        
        # Construct covariance matrix
        cov_m_0 = torch.cat((var_real, cov), dim=-1)
        cov_m_1 = torch.cat((cov, var_imag), dim=-1)
        cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=-3)
        
        # Prepare input for matrix operations
        in_concat = torch.unsqueeze(
            torch.cat((
                torch.unsqueeze(centered_input.real, dim=-1), 
                torch.unsqueeze(centered_input.imag, dim=-1)
            ), dim=-1), 
            dim=-1
        )
        
        # Calculate square root of covariance matrix
        cov_sqr = self.sqrt_2x2(cov_m)
        
        if self.elementwise_affine:
            # Apply learnable parameters
            real_var_weight = (self.weights[:, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            
            weights_mult = torch.cat([
                torch.cat([real_var_weight, cov_weight], dim=-1), 
                torch.cat([cov_weight, imag_var_weight], dim=-1)
            ], dim=-2).unsqueeze(0)
            
            mult_mat = self.sqrt_2x2(weights_mult).matmul(self.inv_2x2(cov_sqr))
            out = mult_mat.matmul(in_concat)
        else:
            # Normalize without learnable parameters
            out = self.inv_2x2(cov_sqr).matmul(in_concat)
            
        # Convert back to complex
        out = out[..., 0, 0] + 1j * out[..., 1, 0]
        
        if self.elementwise_affine:
            return out + self.bias
        return out

    def inv_2x2(self, input):
        # Calculate inverse of 2x2 matrix
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        
        divisor = a * d - b * c
        # Add small epsilon to avoid division by zero
        divisor = divisor + self.eps * torch.ones_like(divisor)
        
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        
        return mat / divisor

    def sqrt_2x2(self, input):
        # Calculate square root of 2x2 matrix
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        # Add small epsilon to ensure numerical stability
        a = a + self.eps * torch.ones_like(a)
        d = d + self.eps * torch.ones_like(d)
        
        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        
        return torch.cat((
            torch.cat((a + s, b), dim=-2), 
            torch.cat((c, d + s), dim=-2)
        ), dim=-1) / t


class ComplexSoftmax(nn.Module):
    def __init__(self, dim):
        super(ComplexSoftmax, self).__init__()
        self.dim = dim

    def forward(self, z):
        """
        Complex-valued Neural Networks with Non-parametric Activation Functions
        (Eq. 36)
        https://arxiv.org/pdf/1802.08026.pdf
        """
        if torch.is_complex(z):
            magnitude = torch.abs(z)
            result = torch.softmax(magnitude, dim=self.dim)
            phase = torch.angle(z)
            return torch.polar(result, phase)
        else:
            return torch.softmax(z, dim=self.dim)


class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, z):
        if self.training:
            real_dropout = F.dropout(z.real, self.p, self.training)
            imag_dropout = F.dropout(z.imag, self.p, self.training)
            return torch.complex(real_dropout, imag_dropout)
        else:
            return z


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        if layer_norm_method == 'layer_norm_naive':
            self.norm = ComplexLayerNorm(dim)
        elif layer_norm_method == 'layer_norm_joint':
            self.norm = ComplexLayerNorm2(dim)
        elif layer_norm_method == 'dynamic_tanh':
            self.norm = ComplexDynamicTanh(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, init_step_size=0.1, init_lambd=0.1, relu_type='crelu'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim, dim), dtype=torch.cfloat))
        with torch.no_grad():
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(self.weight.real)
                init.kaiming_uniform_(self.weight.imag)
            elif init_method == 'wirtinger':
                wirtinger_initialization(self.weight)
            elif init_method == 'unitary':
                complex_unitary_initialization(self.weight)
            elif init_method == 'gorot':
                complex_glorot_uniform(self.weight)
            else:
                raise ValueError('Unknown initialization method')
            
        # Make step_size and lambd learnable parameters
        self.step_size = nn.Parameter(torch.tensor(init_step_size, dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor(init_lambd, dtype=torch.float32))
        
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')

    def forward(self, x):
        # Ensure positive values using softplus or abs
        pos_step_size = F.softplus(self.step_size)
        pos_lambd = F.softplus(self.lambd)
        
        x1 = F.linear(x, self.weight, bias=None)
        grad_update = pos_step_size * x1 - pos_step_size * pos_lambd
        output = self.relu(x + grad_update)
        return output

class FeedForward2(nn.Module):
    def __init__(self, dim, init_step_size=0.1, relu_type='crelu'):
        '''
        This implementation is identical to the one in the original CRATE paper.
        '''
        super().__init__()
        # self.weight = nn.Parameter(torch.Tensor((dim, dim), dtype=torch.cfloat))
        self.weight = nn.Parameter(torch.randn((dim, dim), dtype=torch.cfloat))
        with torch.no_grad():
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(self.weight.real)
                init.kaiming_uniform_(self.weight.imag)
            elif init_method == 'wirtinger':
                wirtinger_initialization(self.weight)
            elif init_method == 'unitary':
                complex_unitary_initialization(self.weight)
            elif init_method == 'gorot':
                complex_glorot_uniform(self.weight)
            else:
                raise ValueError('Unknown initialization method')
            
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')
        self.step_size = init_step_size
        self.lambd = 0.1

    def forward(self, x):
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        output = self.relu(x + grad_update)
        return output


# class OvercompleteISTABlock(nn.Module):
#     """Transformer MLP / feed-forward block."""
#     eta: float = 0.1
#     lmbda: float = 0.1
#     dropout: float = 0.0
 
#     @nn.compact
#     def __call__(self, x, deterministic=True):
#         """Applies CRATE OvercompleteISTABlock module."""
#         n, l, d = x.shape  # pylint: disable=unused-variable
#         D = self.param("D",
#                        nn.initializers.kaiming_uniform(),
#                        (d, 4 * d))
        
        
#         D1 = self.param("D1",
#                            nn.initializers.kaiming_uniform(),
#                            (d, 4 * d))

       
#         negative_lasso_grad = jnp.einsum("p d, n l p -> n l d", D, x)
#         z1 = nn.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

       
#         Dz1= jnp.einsum("d p, n l p -> n l d", D, z1)
#         lasso_grad = jnp.einsum("p d, n l p -> n l d", D, Dz1 - x)
#         z2 = nn.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        
#         xhat = jnp.einsum("d p, n l p -> n l d", D1, z2)
     
#         return xhat

# this is a further improvement based on the inspiration from the CRATE-alpha model
class FeedForward3(nn.Module):
    def __init__(self, dim, init_step_size=0.1, init_lambd=0.1, relu_type='crelu'):
        '''
        This implementation is inspired by the CRATE-alpha model.
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim, 4*dim), dtype=torch.cfloat))
        self.weight2 = nn.Parameter(torch.randn((dim, dim), dtype=torch.cfloat))
        with torch.no_grad():
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(self.weight.real)
                init.kaiming_uniform_(self.weight.imag)
                init.kaiming_uniform_(self.weight2.real)
                init.kaiming_uniform_(self.weight2.imag)
            elif init_method == 'wirtinger':
                wirtinger_initialization(self.weight)
                wirtinger_initialization(self.weight2)
            elif init_method == 'unitary':
                complex_unitary_initialization(self.weight)
                complex_unitary_initialization(self.weight2)
            elif init_method == 'gorot':
                complex_glorot_uniform(self.weight)
                complex_glorot_uniform(self.weight2)
            else:
                raise ValueError('Unknown initialization method')
            
        # Make step_size and lambd learnable parameters
        self.step_size = nn.Parameter(torch.tensor(init_step_size, dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor(init_lambd, dtype=torch.float32))
        
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')

    def forward(self, x):
        # Ensure positive values using softplus or abs
        pos_step_size = F.softplus(self.step_size)
        pos_lambd = F.softplus(self.lambd)
        
        # over-parameterized
        x1 = F.linear(x, self.weight.t(), bias=None)
        grad_update = pos_step_size * x1 - pos_step_size * pos_lambd
        reconstruction = F.linear(grad_update, self.weight, bias=None)
        x2 = self.relu(x + reconstruction)
        xhat = F.linear(x2, self.weight2, bias=None)
        return xhat
    
    
class FeedForward4(nn.Module):
    def __init__(self, dim, init_step_size=0.1, init_lambd=0.1, relu_type='crelu'):
        '''
        PyTorch implementation of the OvercompleteISTABlock from CRATE-alpha paper.
        '''
        super().__init__()
        self.D = nn.Parameter(torch.randn((dim, 4*dim), dtype=torch.cfloat))
        self.D1 = nn.Parameter(torch.randn((dim, 4*dim), dtype=torch.cfloat))
        
        with torch.no_grad():
            if init_method == 'kaiming_uniform':
                init.kaiming_uniform_(self.D.real)
                init.kaiming_uniform_(self.D.imag)
                init.kaiming_uniform_(self.D1.real)
                init.kaiming_uniform_(self.D1.imag)
            elif init_method == 'wirtinger':
                wirtinger_initialization(self.D)
                wirtinger_initialization(self.D1)
            elif init_method == 'unitary':
                complex_unitary_initialization(self.D)
                complex_unitary_initialization(self.D1)
            elif init_method == 'gorot':
                complex_glorot_uniform(self.D)
                complex_glorot_uniform(self.D1)
            else:
                raise ValueError('Unknown initialization method')
            
        # Make step_size and lambd learnable parameters
        self.step_size = nn.Parameter(torch.tensor(init_step_size, dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor(init_lambd, dtype=torch.float32))
        
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')

    def forward(self, x):
        # Ensure positive values using softplus or abs
        pos_step_size = F.softplus(self.step_size)
        pos_lambd = F.softplus(self.lambd)
        
        # Negative lasso gradient
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.D, x)
        z1 = self.relu(pos_step_size * negative_lasso_grad - pos_step_size * pos_lambd)
        
        # Compute lasso gradient
        Dz1 = torch.einsum("dp,nlp->nld", self.D, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.D, Dz1 - x)
        z2 = self.relu(z1 - pos_step_size * lasso_grad - pos_step_size * pos_lambd)
        
        # Final output
        xhat = torch.einsum("dp,nlp->nld", self.D1, z2)
        out = xhat + x
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = ComplexSoftmax(dim=-1)
        self.dropout = ComplexDropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False, dtype=torch.cfloat)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, dtype=torch.cfloat),
            ComplexDropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2).conj()) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., init_step_size=0.1, relu_type='crelu', feedforward='type1'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        
        if feedforward == 'type1':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward(dim, init_step_size=init_step_size, relu_type=relu_type))
                        ]
                    )
                )
        elif feedforward == 'type2':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward2(dim, init_step_size=init_step_size, relu_type=relu_type))
                        ]
                    )
                )
        elif feedforward == 'type3':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward3(dim, init_step_size=init_step_size, relu_type=relu_type))
                        ]
                    )
                )
        elif feedforward == 'type4':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward4(dim, init_step_size=init_step_size, relu_type=relu_type))
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=init_step_size))
                        ]
                    )
                )
        
        
    def forward(self, x):
        for attn, ff in self.layers:
            grad_x = attn(x) + x
            x = ff(grad_x)
        return x



class ComplexToRealImag(nn.Module):
    def __init__(self, output_dim):
        super(ComplexToRealImag, self).__init__()
        # Define real-valued layers to process the combined real and imaginary parts
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        # Concatenate real and imaginary parts
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat((x_real, x_imag), dim=-1)
        # Process through real-valued fully connected layer
        out = self.fc(x_combined)
        return out

class ComplexToAmpPhase(nn.Module):
    def __init__(self, output_dim):
        super(ComplexToAmpPhase, self).__init__()
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        x_amp = x.abs()
        x_phase = x.angle()
        x_combined = torch.cat((x_amp, x_phase), dim=-1)
        out = self.fc(x_combined)
        return out
        

class ComplexMagnitude(nn.Module):
    def forward(self, x):
        return x.abs()  # Compute the magnitude of the complex numbers

class ComplexToReal(nn.Module):
    def forward(self, x):
        return x.real  # Extract the real part



class RF_CRATE(nn.Module):
    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            dropout=0., emb_dropout=0., init_step_size=0.1, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
        super().__init__()
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if patch_embedding_method == 'linear_patch':
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                ComplexLayerNorm(patch_dim),
                nn.Linear(patch_dim, dim, dtype=torch.cfloat),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'conv_patch':
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, dtype=torch.cfloat),
                # nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, dtype=torch.cfloat, padding=padding),
                Rearrange('b c h w -> b (h w) c', c=dim),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'group_conv_patch':
            # Find the greatest common divisor (GCD) of channels and dim
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            max_groups = gcd(channels, dim)
            # Choose a reasonable number of groups that's not too small or too large
            groups = max_groups
                
            print(f"Using {groups} groups for group convolution")
            print(f"Channels: {channels}, Dim: {dim}")
            
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(
                    channels, 
                    dim, 
                    kernel_size=patch_size, 
                    stride=patch_size, 
                    groups=groups, 
                    dtype=torch.cfloat
                ),
                Rearrange('b c h w -> b (h w) c', c=dim),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'group_conv_linear_patch':
            groups = channels
            
            print(f"Using {groups} groups for group convolution with linear projection")
            print(f"Channels: {channels}, Dim: {dim}")
            
            self.to_patch_embedding = nn.Sequential(
                # Group convolution where each channel has its own filter
                nn.Conv2d(
                    channels, 
                    channels,  # Output same number of channels
                    kernel_size=patch_size, 
                    stride=patch_size, 
                    groups=groups,  # Each channel processed independently
                    dtype=torch.cfloat
                ),
                Rearrange('b c h w -> b (h w) c'),  # Reshape to sequence format
                # Linear projection to map from channels to dim
                nn.Linear(channels, dim, dtype=torch.cfloat),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'soft_conv_patch': # Soft Convolutional Patch Embedding with the stride is half of the patch size
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=(patch_size[0] // 2, patch_size[1] // 2), padding=(patch_size[0] // 4, patch_size[1] // 4), dtype=torch.cfloat),
                Rearrange('b c h w -> b (h w) c', c=dim),
                ComplexLayerNorm(dim),
            )
            stride_height = patch_height // 2
            stride_width = patch_width // 2
            num_patches = (image_height // stride_height) * (image_width // stride_width)
            
        elif patch_embedding_method == 'conv_module_patch':
            # Hybrid CNN + Transformer approach with a three-layer CNN module
            # Each layer has: Conv2D -> Activation -> BatchNorm
            
            conv_stride = 2  # Using stride of 2 for each conv layer
            
            # Choose activation function based on relu_type
            if relu_type == 'crelu':
                activation = CReLU()
            elif relu_type == 'zrelu':
                activation = ZReLU()
            elif relu_type == 'modrelu':
                activation = ModReLU()
            elif relu_type == 'cardioid':
                activation = ComplexCardioid()
            else:
                activation = CReLU()  # Default to CReLU
            
            # Use our custom ComplexBatchNorm2d instead of nn.BatchNorm2d
            self.to_patch_embedding = nn.Sequential(
                # First conv layer
                nn.Conv2d(channels, dim // 4, kernel_size=3, stride=conv_stride, padding=1, dtype=torch.cfloat),
                activation,
                ComplexBatchNorm2d(dim // 4),
                
                # Second conv layer
                nn.Conv2d(dim // 4, dim // 2, kernel_size=3, stride=conv_stride, padding=1, dtype=torch.cfloat),
                activation,
                ComplexBatchNorm2d(dim // 2),
                
                # Third conv layer
                nn.Conv2d(dim // 2, dim, kernel_size=3, stride=conv_stride, padding=1, dtype=torch.cfloat),
                activation,
                ComplexBatchNorm2d(dim),
                
                # Reshape to sequence format
                Rearrange('b c h w -> b (h w) c', c=dim),
            )
            
            # Calculate number of patches based on output shape after 3 convolutions with stride 2
            h_patches = image_height // (conv_stride**3)
            w_patches = image_width // (conv_stride**3)
            num_patches = h_patches * w_patches
        else:
            raise ValueError('Unknown patch embedding method')
        
        self.pos_embedding = nn.Parameter(torch.randn((1, num_patches + 1, dim), dtype=torch.cfloat))
        self.cls_token = nn.Parameter(torch.randn((1, 1, dim), dtype=torch.cfloat))
        self.dropout = ComplexDropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, init_step_size=init_step_size, relu_type=relu_type, feedforward=feedforward)
        self.pool = pool
        self.to_latent = nn.Identity()
        if mlp_head_type == 'crate_version':
            self.mlp_head = nn.Sequential(
            ComplexLayerNorm(dim),
            nn.Linear(dim, num_classes, dtype=torch.cfloat),
            ComplexMagnitude() if output_type == 'magnitude' else ComplexToAmpPhase(num_classes) if output_type == 'phase_amp' else ComplexToRealImag(num_classes) if output_type == 'real_imag' else ComplexToReal(),
            )
        elif mlp_head_type == 'vit_version':
            self.mlp_head = nn.Sequential(
                ComplexLayerNorm(dim),
                nn.Linear(dim, dim * 2, dtype=torch.cfloat),
                CReLU() if relu_type == 'crelu' else ZReLU() if relu_type == 'zrelu' else ModReLU() if relu_type == 'modrelu' else ComplexCardioid(),
                ComplexLayerNorm(dim*2),
                nn.Linear(dim * 2, num_classes, dtype=torch.cfloat),
                ComplexMagnitude() if output_type == 'magnitude' else ComplexToAmpPhase(num_classes) if output_type == 'phase_amp' else ComplexToRealImag(num_classes) if output_type == 'real_imag' else ComplexToReal(),
            )
        else:
            raise ValueError('Unknown mlp head type')

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        feature_last = x
        output = self.mlp_head(x)
        return output, feature_pre
    
    # @staticmethod
    def get_selfattention(self, rf_data, layer=5):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            # x = self.dropout(x)
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    attn_map = attn(x)
                    return attn_map
    
    # @staticmethod
    def get_feature(self, rf_data, layer=11):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    return x

    # @staticmethod
    def get_qkv(self, rf_data, layer=11):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    qkv = attn.fn.qkv(x)
                    qkv = qkv[None, :, :, :]  # Adjust dimensions if necessary
                    # Exclude the class token
                    return qkv[:, :, 1:, :]  # Shape: [1, num_heads, num_patches, dim]
                


def rf_crate_mini(num_classes, image_size, patch_size, in_channels=3,feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=6,
        heads=4,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=192 // 4,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
        )


def rf_crate_wide_tiny(num_classes, image_size, patch_size, in_channels=3,feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=24,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 24,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
    )
    

def rf_crate_tiny(num_classes, image_size, patch_size, in_channels=3,feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
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
    
def rf_crate_small(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
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

def rf_crate_base(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu', patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
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
    
def rf_crate_large(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
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

# Add this new class to handle complex batch normalization
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.imag_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        
    def forward(self, x):
        return torch.complex(
            self.real_bn(x.real),
            self.imag_bn(x.imag)
        )

