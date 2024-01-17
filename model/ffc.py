# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumReLU(nn.Module):

    def __init__(self, spatial_size):
        super(SpectrumReLU, self).__init__()

        self.weight = nn.Parameter(torch.ones(spatial_size))
        self.bias = nn.Parameter(torch.zeros(spatial_size))

    def forward(self, x):
        x = torch.add(torch.mul(self.weight, x), self.bias)
        x = nn.functional.relu(x, inplace=True)
        return x

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class one_hot_2d(object):
    """
    Convert a [bs x 2D] ID array into an one hot
        [bs x max_dim x 2D] binary array.
    """

    def __init__(self,
                 max_dim=None,
                 ignore_label=None,
                 **kwargs):
        """
        Args:
            max_dim: An integer tells largest one_hot dimension.
                None means auto-find the largest ID as max_dim.
            ignore_label: An integer or array tells the ignore ID(s).
        """
        self.max_dim = max_dim
        if isinstance(ignore_label, int):
            self.ignore_label = [ignore_label]
        elif ignore_label is None:
            self.ignore_label = []
        else:
            self.ignore_label = ignore_label

    def __call__(self,
                 x,
                 mask=None):

        if mask is not None:
            x *= mask == 1

        check = []
        for i, n in enumerate(np.bincount(x.flatten())):
            if (i not in self.ignore_label) and (n > 0):
                check.append(i)

        if self.max_dim is None:
            max_dim = check[-1] + 1
        else:
            max_dim = self.max_dim
        batch_n, h, w = x.shape

        oh = np.zeros((batch_n, max_dim, h, w)).astype(np.uint8)
        for c in check:
            if c >= max_dim:
                continue
            oh[:, c, :, :] = x == c

        if mask is not None:
            # remove the unwanted one-hot zeros
            oh[:, 0, :, :] *= mask == 1
        return oh

def make_cweight(half_size, half_sample, type='piecewise_linear', oddeven_aligned=True, device='cuda'):
    """
    Make a coordinate based weighting.
    Args:
        half_size: [int, int],
            the size of height and width,
            height will be normalized to -1 to 1
            width will be normalized to 0 to 1
        half_sample: [int, int],
            the size of height and width on sampling
        type: str,
            tells how to sample these points:
        oddeven_aligned: bool,
            If half_sample use the oddeven align rule, and if height is even number
                it means it will not be noamlized to exact [-1, 1],
                but to [-1+2/height_sample, 1],
                so the [0, 0] origin is at [height_sample//2+1, 0]
    Outputs:
        cweight:
            a one-hot on each location of the half_size matrix will be prepared.
            a grid sampling based on the type will be perform on half_sampling grid.
            the result is cweight
    """

    h0, w0 = half_size
    hs, ws = half_sample

    reference_id = np.array([i for i in range(h0 * w0)]).reshape(1, h0, w0)
    reference_oh = one_hot_2d(max_dim=h0 * w0)(reference_id)
    reference_oh = torch.Tensor(reference_oh).float().to(device)

    # expand so the reference is on the whole [-1, 1]^2 plane
    reference_oh = F.pad(reference_oh, pad=(w0 - 1, 0, 0, 0), mode='reflect')

    if oddeven_aligned and hs % 2 == 0:
        h_grid = np.array([-1 + i / (hs) * 2 for i in range(hs + 1)])[1:]
    else:
        h_grid = np.array([-1 + i / (hs - 1) * 2 for i in range(hs)])
    w_grid = np.array([0 + i / (ws - 1) for i in range(ws)])
    w_grid, h_grid = np.meshgrid(w_grid, h_grid)
    grid = np.stack([w_grid, h_grid], axis=-1)  # format '[h x w x wh]'
    grid = torch.Tensor(grid).float().unsqueeze(0).to(device)

    if type == 'piecewise_linear':
        cweight = F.grid_sample(reference_oh, grid, mode='bilinear', padding_mode='border', align_corners=True)
        cweight = cweight.squeeze(0)
    elif type == 'bicubic':
        cweight = F.grid_sample(reference_oh, grid, mode='bicubic', padding_mode='border', align_corners=True)
        cweight = cweight.squeeze(0)
    else:
        raise NotImplementedError
    return cweight
class heterogeneous_filter(nn.Module):
    def __init__(self, in_channels, out_channels, freedom, type, init='ones'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freedom = freedom
        self.type = type
        self.cw_cache = None
        self.size_cache = None

        if type in ['piecewise_linear', 'bicubic']:
            fh, fw = freedom
            self.weight = nn.Parameter(
                torch.empty(in_channels, out_channels*fh*fw), requires_grad=True)
        else:
            raise NotImplementedError

        if init == 'ones':
            nn.init.ones_(self.weight)

    def forward(self, x):
        bs, c, h, w = x.shape
        if self.cw_cache is not None and self.size_cache == x.shape:
            cw = self.cw_cache
        elif self.type in ['piecewise_linear', 'bicubic']:
            cw = make_cweight(
                half_size=self.freedom,
                half_sample=x.shape[2:],
                type=self.type,
                oddeven_aligned=True,
                device=x.device)
            self.cw_cache = cw
            self.size_cache = x.shape

        weight = self.weight.T.unsqueeze(-1).unsqueeze(-1)
        y = F.conv2d(x, weight).view(bs, c, -1, h, w)
        o = (y * cw.unsqueeze(0).unsqueeze(0)).sum(2)
        return o


class SHU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        '''
        Args:
            tail_sigma_mult: float,
                tells how much does the sigma extend.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv0 = nn.Conv2d(in_channels * 2, in_channels * 2, 1, 1, 0)
        self.df1 = heterogeneous_filter(
            in_channels * 2, out_channels * 2,
            freedom=dfilter_freedom, type=dfilter_type)
        torch.nn.init.normal_(self.df1.weight, mean=1 / (out_channels * 2), std=0.1 / (out_channels * 2))
        self.act = torch.nn.ReLU(inplace=True)


    def forward(self, x):
        ffted = torch.fft.rfftn(x, dim=(2, 3), norm='forward')
        # Shift is necessary because the top-left is low frequency (make it at center)
        ffted = torch.cat([
            ffted[:, :, ffted.size(2) // 2 + 1:, :],
            ffted[:, :, :ffted.size(2) // 2 + 1, :]], dim=2)

        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.conv0(ffted)
        ffted = self.act(ffted)
        ffted = self.df1(ffted)
        ffted = torch.complex(ffted[:, 0:self.out_channels], ffted[:, self.out_channels:])

        ffted = torch.cat([
            ffted[:, :, ffted.size(2) // 2 - 1:, :],
            ffted[:, :, :ffted.size(2) // 2 - 1, :]], dim=2)
        output = torch.fft.irfftn(ffted, dim=(2, 3), norm='forward')  # /10*resi
        return output

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1,
                  ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            #Local Fourier Unit (LFU)
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


###frequency upsample
class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H / 2), 0:int(W / 2)] = output[:, :, 0:int(H / 2), 0:int(W / 2)]
        crop[:, :, int(H / 2):H, 0:int(W / 2)] = output[:, :, int(H * 1.5):2 * H, 0:int(W / 2)]
        crop[:, :, 0:int(H / 2), int(W / 2):W] = output[:, :, 0:int(H / 2), int(W * 1.5):2 * W]
        crop[:, :, int(H / 2):H, int(W / 2):W] = output[:, :, int(H * 1.5):2 * H, int(W * 1.5):2 * W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, channels):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        # self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r = x.size(2)  # h
        c = x.size(3)  # w

        I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()

        if r % 2 == 1:  # odd
            ir1, ir2 = r // 2 + 1, r // 2 + 1
        else:  # even
            ir1, ir2 = r // 2 + 1, r // 2
        if c % 2 == 1:  # odd
            ic1, ic2 = c // 2 + 1, c // 2 + 1
        else:  # even
            ic1, ic2 = c // 2 + 1, c // 2

        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
            I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
            I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
            I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
            I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output


## the plug-and-play operator

class fresadd(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x

        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x3 = self.Fup(x1)

        xm = x2 + x3
        xn = self.fuse(xm)

        return xn


class frescat(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(2 * channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x

        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x3 = self.Fup(x1)

        xn = self.fuse(torch.cat([x2, x3], dim=1))

        return xn