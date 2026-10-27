"""The single-pass U-Net surrogate: one pass of a U-Net gives the displacement fluctuation w of a periodic polycrystal, and the fixed B-bar stress
layer of `src/solvers/fem/element_torch.py` turns w into the localization tensor A.

The input has 39 channels at every voxel: the 21 independent entries of the Mandel stiffness C, standardized with the statistics of the training
volumes, and the 18 nodal forces b of the six unit strains. b is divided by the stiffness scale c0 and by its root-mean-square size s for each
volume and load case; the divisor is s, but at least the smallest positive float, so that a load case with b = 0 stays finite. The 18 outputs
are multiplied by the same s, and their spatial mean is removed. The U-Net has 4 levels with the widths
(32, 64, 64, 64), two 3 x 3 x 3 convolutions with circular padding per level, the smooth leaky activation, average pooling, nearest-neighbour
upsampling and concatenated skips. A final 1 x 1 x 1 convolution starts at zero, so the untrained network returns w = 0 and its prediction is
A = C. The network has 1,445,106 weights.

`forward(c21)` takes the raw stiffness (B, 21, n, n, n) and returns a dict with w, b, A (`stress=True`) and the out-of-balance nodal forces
b - K w (`out_of_balance=True`), all in the dtype of the input.
"""
import math

import torch
import torch.nn as nn

from src.solvers.fem import element_torch

WIDTHS = (32, 64, 64, 64)
N_INPUT = 39
N_OUTPUT = 18


class SmoothLeaky(nn.Module):
    """a x + (1 - a) (softplus_b(x) - ln 2 / b): a smooth leaky unit with f(0) = 0."""

    def __init__(self, a=0.2, b=3.0):
        super().__init__()
        self.a, self.b = float(a), float(b)

    def forward(self, x):
        return self.a * x + (1.0 - self.a) * (nn.functional.softplus(x, beta=self.b) - math.log(2.0) / self.b)

    def extra_repr(self):
        return f"a={self.a:g}, b={self.b:g}"


def _conv(cin, cout):
    return nn.Conv3d(cin, cout, 3, padding=1, padding_mode="circular")


def _block(cin, cout):
    return nn.Sequential(_conv(cin, cout), SmoothLeaky(0.2, 3.0), _conv(cout, cout), SmoothLeaky(0.2, 3.0))


class UNet(nn.Module):
    def __init__(self, widths=WIDTHS, cin=N_INPUT, cout=N_OUTPUT):
        super().__init__()
        w = self.widths = tuple(widths)
        self.enc = nn.ModuleList([_block(cin if i == 0 else w[i - 1], w[i]) for i in range(len(w))])
        self.dec = nn.ModuleList([_block(w[i + 1] + w[i], w[i]) for i in range(len(w) - 1)])
        self.pool = nn.AvgPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Conv3d(w[0], cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if x.shape[-1] % 2 ** (len(self.enc) - 1):
            raise ValueError(f"the grid edge must be a multiple of {2 ** (len(self.enc) - 1)}, not {x.shape[-1]}")
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for blk in reversed(self.dec):
            x = blk(torch.cat([self.up(x), skips.pop()], 1))
        return self.out(x)


def load_rms(x):
    """Root-mean-square size over the three components and the nodes of every load case of a field (B, 18, n, n, n), channel 6 c + l: (B, 6)."""
    return x.reshape(x.shape[0], 3, 6, -1).pow(2).mean((1, 3)).sqrt()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_copy(model):
    """A copy of the weights and the buffers of `model`."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _per_load(v):
    return v.reshape(-1, 1, 6, 1, 1, 1)


class Surrogate(nn.Module):
    """The U-Net with its standardization: `stats` holds xm, xs (1, 21, 1, 1, 1), ym, ys (1, 36, 1, 1, 1) and the stiffness scale c0."""

    def __init__(self, stats):
        super().__init__()
        for k in ("xm", "xs", "ym", "ys"):
            self.register_buffer(k, stats[k].detach().clone().float())
        self.c0 = float(stats["c0"])
        self.net = UNet()
        self._elements = {}

    def element(self, like):
        key = (like.dtype, like.device)
        if key not in self._elements:
            self._elements[key] = element_torch.Element(like.dtype, like.device)
        return self._elements[key]

    def forward(self, c21, stress=True, out_of_balance=False):
        el = self.element(c21)
        n = c21.shape[-1]
        C6 = element_torch.c21_to_c6(c21)
        ke, b = el.ke(C6), el.rhs(C6)
        xin = (c21 - self.xm) / self.xs
        r = b / self.c0
        s = load_rms(r)
        den = s.clamp_min(torch.finfo(r.dtype).tiny)
        shape = (r.shape[0], 3, 6) + r.shape[2:]
        w = self.net(torch.cat([xin, (r.reshape(shape) / _per_load(den)).reshape(r.shape)], 1))
        w = (w.reshape(shape) * _per_load(s)).reshape(w.shape)
        w = w - w.mean((2, 3, 4), keepdim=True)
        out = dict(w=w, b=b)
        if stress or out_of_balance:
            u = el.gather(w)
            if stress:
                out["A"] = el.stress_u(C6, u, n)
            if out_of_balance:
                out["out_of_balance"] = el.residual_u(ke, b, u, n)
        return out


def save_checkpoint(path, model, **info):
    """Writes the weights (with the standardization buffers) and the stiffness scale c0, plus the plain values in `info`."""
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, c0=model.c0, **info), path)


def load_checkpoint(path, dtype=torch.float32, device="cpu"):
    """(model, checkpoint dict): the model is rebuilt from the file and cast to `dtype`."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    state = ck["state"]
    model = Surrogate(dict(xm=state["xm"], xs=state["xs"], ym=state["ym"], ys=state["ys"], c0=ck["c0"]))
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval(), ck


def load_initial_weights(model, path):
    """Loads the network weights (the keys `net.*`) of a checkpoint file into `model`; the standardization of the model is not touched."""
    state = torch.load(path, map_location="cpu", weights_only=True)["state"]
    net = {k: v for k, v in state.items() if k.startswith("net.")}
    if net.keys() != {k for k in model.state_dict() if k.startswith("net.")}:
        raise ValueError(f"{path}: the network keys differ from those of the model")
    model.load_state_dict(net, strict=False)
