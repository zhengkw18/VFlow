import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import mean_dim


class ActNorm(nn.Module):
    """Activation normalization for 2D inputs.

    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """

    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, weight_std=0.05):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.normal_(self.linear.weight, 0., weight_std)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input):
        input = input.transpose(1, 3)
        output = self.linear(input)
        output = output.transpose(1, 3)
        output = output + self.bias
        return output


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3.0):
        super(LinearZeros, self).__init__()
        self.logscale_factor = logscale_factor
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input):
        input = input.transpose(1, 3)
        output = self.linear(input)
        output = output.transpose(1, 3)
        output = output + self.bias
        h = self.logs.mul(self.logscale_factor).exp()
        output = output * h
        return output


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, weight_std=0.05):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        nn.init.normal_(self.conv2d.weight, 0., weight_std)
        nn.init.zeros_(self.conv2d.bias)

    def forward(self, input):
        x = self.conv2d(input)
        return x


class Conv2dActNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, weight_std=0.05):
        super(Conv2dActNorm, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.normal_(self.conv2d.weight, 0., weight_std)
        self.actnorm = ActNorm(out_channels)

    def forward(self, input):
        x = self.conv2d(input)
        x = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, logscale_factor=3.0):
        super(Conv2dZeros, self).__init__()
        self.logscale_factor = logscale_factor
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.zeros_(self.conv2d.weight)

    def forward(self, input):
        conv_output = self.conv2d(input)
        conv_output = conv_output + self.bias
        h = self.logs.mul(self.logscale_factor).exp()
        output = conv_output * h
        return output


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
    """

    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.

    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """

    def __init__(self, in_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, mid_channels, 2 * in_channels)

    def forward(self, x, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id)
        shift, scale = st[:, 0::2, ...], st[:, 1::2, ...]
        scale = torch.sigmoid(scale.add(2.))
        # Scale and translate
        if reverse:
            x_change = x_change / scale - shift
            ldj = ldj - scale.log().flatten(1).sum(-1)
        else:
            x_change = (x_change + shift) * scale
            ldj = ldj + scale.log().flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super(Split2d, self).__init__()
        self.conv2dzeros = Conv2dZeros(num_channels // 2, num_channels)
        self.diaglogp = GaussianDiagLogp()

    def forward(self, x, ldj, reverse=False):
        if not reverse:
            z1, z2 = x.chunk(2, dim=1)
            h = self.conv2dzeros(z1)
            means, logs = h[:, 0::2, ...], h[:, 1::2, ...]
            h = self.diaglogp(z2, means, logs)
            ldj = ldj + h
            return z1, ldj
        else:
            z1 = x
            h = self.conv2dzeros(z1)
            means, logs = h[:, 0::2, ...], h[:, 1::2, ...]
            z2 = torch.normal(means, logs.exp())
            z = torch.cat((z1, z2), dim=1)
            return z, ldj


class GaussianDiagLogp(nn.Module):
    def __init__(self):
        super(GaussianDiagLogp, self).__init__()

    def forward(self, x, means=None, logs=None):
        if means is None:
            means = torch.zeros_like(x)
        if logs is None:
            logs = torch.zeros_like(x)
        sldj = -0.5 * ((x - means) ** 2 / (2 * logs).exp() + np.log(2 * np.pi) + 2 * logs)
        sldj = sldj.flatten(1).sum(-1)
        return sldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(NN, self).__init__()

        self.in_conv = Conv2dActNorm(in_channels, mid_channels)
        self.mid_conv = Conv2dActNorm(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.out_conv = Conv2dZeros(mid_channels, out_channels)

    def forward(self, x):
        x = self.in_conv(x)
        x = F.relu(x)
        x = self.mid_conv(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return x


class GatedConv(nn.Module):
    def __init__(self, channels):
        super(GatedConv, self).__init__()
        self.conv1 = Conv2d(channels * 2, channels)
        self.conv2 = Conv2d(channels * 2, channels * 2)

    def forward(self, x):
        y = nonlinearity(x)
        y = self.conv1(y)
        y = nonlinearity(y)
        y = self.conv2(y)
        return x + gate(y)


class GatedConvPrior(nn.Module):
    def __init__(self, channels):
        super(GatedConvPrior, self).__init__()
        self.conv = Conv2d(channels * 2, channels)
        self.linear1 = Linear(channels * 2, channels * 2)
        self.linear2 = Linear(channels * 2, channels * 2)

    def forward(self, x, a):
        y = nonlinearity(x)
        y = self.conv(y)
        y = nonlinearity(y)
        y = self.linear1(y)
        a = nonlinearity(a)
        a = self.linear2(a)
        return x + gate(y + a)


class ShallowProcesser(nn.Module):
    def __init__(self, channels, mid_channels):
        super(ShallowProcesser, self).__init__()
        self.conv = Conv2d(channels, mid_channels)
        self.gate1 = GatedConv(mid_channels)
        self.gate2 = GatedConv(mid_channels)
        self.gate3 = GatedConv(mid_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gate1(x)
        x = self.gate2(x)
        x = self.gate3(x)
        return x


class AffineCouplingPrior(nn.Module):
    def __init__(self, in_channels, mid_channels, width, height):
        super(AffineCouplingPrior, self).__init__()
        self.conv_in = Conv2d(in_channels, mid_channels)
        self.gated_conv = GatedConvPrior(mid_channels)
        self.layernorm = nn.LayerNorm([mid_channels, width, height])
        self.conv_out = Conv2d(mid_channels, in_channels * 2)

    def forward(self, x, a, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.conv_in(x_id)
        st = self.gated_conv(st, a)
        st = self.layernorm(st)
        st = self.conv_out(st)
        shift, scale = st[:, 0::2, ...], st[:, 1::2, ...]
        scale = torch.sigmoid(scale.add(2.))
        # Scale and translate
        if reverse:
            x_change = x_change / scale - shift
            ldj = ldj - scale.log().flatten(1).sum(-1)
        else:
            x_change = (x_change + shift) * scale
            ldj = ldj + scale.log().flatten(1).sum(-1)

        x = torch.cat((x_id, x_change), dim=1)

        return x, ldj


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            y = torch.sigmoid(x)
            ldj = -F.softplus(x) - F.softplus(-x)
            ldj = ldj.flatten(1).sum(-1)
            sldj = sldj + ldj
            return y, sldj
        else:
            y = -(torch.reciprocal(x) - 1.).log()
            ldj = -x.log() - (1. - x).log()
            ldj = ldj.flatten(1).sum(-1)
            sldj = sldj + ldj
            return y, sldj


def nonlinearity(x):
    return F.elu(torch.cat((x, -x), dim=1))


def gate(x):
    a, b = x.chunk(2, dim=1)
    return a * torch.sigmoid(b)
