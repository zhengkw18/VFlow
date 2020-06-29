import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow.modules import ActNorm, Coupling, InvConv, GaussianDiagLogp, Split2d, AffineCouplingPrior, ShallowProcesser, Sigmoid


class Glow(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        num_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """

    def __init__(self, img_shape, num_channels, num_levels, num_steps):
        super(Glow, self).__init__()

        self.B, self.C, self.W, self.H = img_shape

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _Glow(img_shape=(self.B, 2 * self.C, self.W, self.H),  # RGB image after squeeze
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)
        self.augment = Augment(img_shape, num_channels)

    def forward(self, x, reverse=False):
        if reverse:
            sldj = torch.zeros(self.B, device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'.format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)
        if not reverse:
            z, ldj = self.augment(x)
            x = torch.cat((x, z), dim=1)
            sldj = sldj - ldj
        x, sldj = self.flows(x, sldj, reverse)
        if reverse:
            x = x[:, 0:self.C, ...]
        return x, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        x = (x * 255. + torch.rand_like(x)) / 256.
        x = (2 * x - 1) * self.bounds
        x = (x + 1) / 2
        x = x.log() - (1. - x).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(x) + F.softplus(-x) - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return x, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """

    def __init__(self, img_shape, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        B, C, W, H = img_shape
        C *= 4
        W //= 2
        H //= 2
        self.B, self.C, self.W, self.H = B, C, W, H

        self.steps = nn.ModuleList([_FlowStep(in_channels=C,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])
        if num_levels > 1:
            self.split2d = Split2d(num_channels=C)
            self.next = _Glow(img_shape=(B, C // 2, W, H),
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None
            self.diaglogp = GaussianDiagLogp()

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            x = squeeze(x, reverse=False)
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)
            if self.next is not None:
                x, sldj = self.split2d(x, sldj)
                x, sldj = self.next(x, sldj, reverse)
                return x, sldj
            else:
                sldj = sldj + self.diaglogp(x)
                return None, sldj
        else:
            if self.next is not None:
                x, sldj = self.next(x, sldj, reverse)
                x, sldj = self.split2d(x, sldj, reverse=True)
            else:
                x = torch.randn((self.B, self.C, self.W, self.H), device=sldj.device)
                sldj = sldj + self.diaglogp(x)
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)
            x = squeeze(x, reverse=True)
            return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x


class AugmentStep(nn.Module):
    def __init__(self, shape, mid_channels):
        super(AugmentStep, self).__init__()
        self.B, self.C, self.W, self.H = shape
        self.norm = ActNorm(self.C, return_ldj=True)
        self.conv = InvConv(self.C)
        self.coup = AffineCouplingPrior(self.C // 2, mid_channels, self.W, self.H)

    def forward(self, x, a, sldj):
        x, sldj = self.norm(x, sldj)
        x, sldj = self.conv(x, sldj)
        x, sldj = self.coup(x, a, sldj)
        return x, sldj


class Augment(nn.Module):
    def __init__(self, shape, mid_channels):
        super(Augment, self).__init__()
        self.B, self.C, self.W, self.H = shape
        self.diaglogp = GaussianDiagLogp()
        self.shallow = ShallowProcesser(self.C * 4, mid_channels)
        self.step1 = AugmentStep((self.B, self.C * 4, self.W // 2, self.H // 2), mid_channels)
        self.step2 = AugmentStep((self.B, self.C * 4, self.W // 2, self.H // 2), mid_channels)
        self.step3 = AugmentStep((self.B, self.C * 4, self.W // 2, self.H // 2), mid_channels)
        self.step4 = AugmentStep((self.B, self.C * 4, self.W // 2, self.H // 2), mid_channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        ldj = torch.zeros(x.size()[0], device=x.device)
        x = squeeze(x)
        a = self.shallow(x)
        eps = torch.randn((x.size()[0], self.C * 4, self.W // 2, self.H // 2), device=x.device)
        eps_ldj = self.diaglogp(eps)
        z = eps
        z, ldj = self.step1(z, a, ldj)
        z, ldj = self.step2(z, a, ldj)
        z, ldj = self.step3(z, a, ldj)
        z, ldj = self.step4(z, a, ldj)
        z, ldj = self.sigmoid(z, ldj)
        z = squeeze(z, reverse=True)
        return z, eps_ldj - ldj
