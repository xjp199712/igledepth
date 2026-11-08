import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiallyVaryingDepthFilter(nn.Module):
    """
    Depthwise per-pixel dynamic conv guided by IGLEM features.
    - inputs: [guide, x_depth] (B, 2C, H, W)   # noisy depth latent (4ch in SD/Marigold)
    Return:
    - y:         (B, D, H, W)   # filtered depth latent (same shape as x_depth)

    Notes:
    * Kernel is predicted per-pixel & per-channel from `guide`.
    * Softmax-normalized across K*K for stability.
    * Identity init -> start as "do-nothing" pass-through.
    """
    def __init__(self, c_depth: int = 4, c_guide: int = 4, ksize: int = 3, out_dim: int = 320):
        super().__init__()
        assert ksize in (3, 5, 7), "Keep kernels small for efficiency."
        self.c_depth = c_depth
        self.ksize = ksize
        self.pad = ksize // 2

        # weight generator: predicts (C_depth * K*K) logits per pixel
        self.weight_gen = nn.Conv2d(c_guide, c_depth * ksize * ksize, kernel_size=1)
        self.new_conv_in = nn.Conv2d(c_depth, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.reset_parameters_identity()

    def reset_parameters_identity(self):
        # Zero weights => rely on bias to set an identity kernel at init.
        nn.init.zeros_(self.weight_gen.weight)
        nn.init.zeros_(self.weight_gen.bias)
        # Bias to favor the center of KxK so that softmax ~ one-hot(center)
        with torch.no_grad():
            bias = self.weight_gen.bias.view(self.c_depth, self.ksize * self.ksize)
            center = (self.ksize * self.ksize) // 2
            bias[:, center] = 4.0  # large positive => softmax ~1 at center
            self.weight_gen.bias.copy_(bias.view(-1))

    def forward(self, inputs:list) -> torch.Tensor:
        guide, x_depth = inputs.split(4, dim=1)
        B, C, H, W = x_depth.shape
        assert C == self.c_depth, "Channel mismatch for depth latent."

        # 1) generate per-pixel per-channel kernels
        k_logits = self.weight_gen(guide)                   # (B, C*K*K, H, W)
        k_logits = k_logits.view(B, C, self.ksize*self.ksize, H, W)
        k = F.softmax(k_logits, dim=2)                      # normalize across K*K

        # 2) unfold depth latent into local patches
        patches = F.unfold(x_depth, kernel_size=self.ksize, padding=self.pad)  # (B, C*K*K, H*W)
        patches = patches.view(B, C, self.ksize*self.ksize, H, W)

        # 3) weighted sum over K*K
        y_depth = (patches * k).sum(dim=2)   # (B, C, H, W)
        
        y = torch.cat((guide, y_depth), dim=1)
        y = self.new_conv_in(y)
        # y = self.new_conv_in(y_depth)
        return y
