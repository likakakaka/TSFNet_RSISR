import torch
from torch import nn
from model import common
import torch.nn.functional as F
import math
''' Two-stage Spatial-Frequency Joint Learning for Large-Factor Remote Sensing Image Super-Resolution'''
def make_model(args):
    return TSFNet(scale=args.scale[0])

class TSFNet(nn.Module):
    def __init__(self, scale=8,base_num_every_group=2,num_features=64,act='PReLU'):
        super(TSFNet, self).__init__()
        self.scale=scale
        num_every_group = base_num_every_group
        ### spatial branch
        modules_head = [common.ConvBNReLU2D(3, out_channels=num_features,
                                            kernel_size=3, padding=1, act=act)]
        self.head = nn.Sequential(*modules_head)
        self.down1 = common.DownSample(num_features, False, False)
        self.down1_fre = AGPF(num_features // 2)
        self.down1_fre_mo = nn.Sequential(AGPF(num_features // 2))
        self.down1_spa = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down1_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        self.down2 = common.DownSample(num_features, False, False)
        self.down2_fre = AGPF(num_features // 2)
        self.down2_fre_mo = nn.Sequential(AGPF(num_features // 2))
        self.down2_spa = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down2_spamo= nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        self.down3 = common.DownSample(num_features, False, False)
        self.down3_fre = AGPF(num_features // 2)
        self.down3_fre_mo = nn.Sequential(AGPF(num_features // 2))
        self.down3_spa = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down3_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        modules_neck = [common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
        ]
        self.neck_spa = nn.Sequential(*modules_neck)
        self.neck_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.neck_fre = AGPF(num_features // 2)
        self.neck_fre_mo = nn.Sequential(AGPF(num_features // 2))

        self.up1 = common.UpSampler(2, num_features)
        self.up1_spa= nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up1_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up1_fre = AGPF(num_features // 2)
        self.up1_fre_mo = nn.Sequential(AGPF(num_features // 2))

        self.up2 = common.UpSampler(2, num_features)
        self.up2_spa = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up2_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up2_fre = AGPF(num_features // 2)
        self.up2_fre_mo = nn.Sequential(AGPF(num_features // 2))

        self.up3 = common.UpSampler(2, num_features)
        self.up3_spa= nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up3_spamo = nn.Sequential(common.RSPB(
            num_features // 2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up3_fre = AGPF(num_features // 2)
        self.up3_fre_mo = nn.Sequential(AGPF(num_features // 2))

        # define tail module
        modules_tail = [
            common.ConvBNReLU2D(num_features, out_channels=3, kernel_size=3, padding=1,
                         act=act)]

        self.tail = nn.Sequential(*modules_tail)


        ### fusion part
        conv_fuse = []
        for i in range(8):
            conv_fuse.append(selfFuseBlock(num_features))
        self.conv_fuse = nn.Sequential(*conv_fuse)
        ### spatial branch

        modules_head2 = [common.ConvBNReLU2D(3, out_channels=num_features,
                                            kernel_size=3, padding=1, act=act)]
        self.head2 = nn.Sequential(*modules_head2)

        self.down12 = common.DownSample(num_features, False, False)
        self.down1_fre2 = AGPF(num_features//2)
        self.down1_fre_mo2 = nn.Sequential(AGPF(num_features//2))
        self.down1_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down1_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        self.down22 = common.DownSample(num_features, False, False)
        self.down2_fre2 = AGPF(num_features//2)
        self.down2_fre_mo2 = nn.Sequential(AGPF(num_features//2))
        self.down2_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down2_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        self.down32 = common.DownSample(num_features, False, False)
        self.down3_fre2 = AGPF(num_features//2)
        self.down3_fre_mo2 = nn.Sequential(AGPF(num_features//2))
        self.down3_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.down3_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_neck2 = [common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
        ]
        self.neck_spa2 = nn.Sequential(*modules_neck2)
        self.neck_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.neck_fre2 = AGPF(num_features//2)
        self.neck_fre_mo2 = nn.Sequential(AGPF(num_features//2))

        self.up12 = common.UpSampler(2, num_features)
        self.up1_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up1_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up1_fre2 = AGPF(num_features//2)
        self.up1_fre_mo2 = nn.Sequential(AGPF(num_features//2))

        self.up22 = common.UpSampler(2, num_features)
        self.up2_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up2_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up2_fre2 = AGPF(num_features//2)
        self.up2_fre_mo2 = nn.Sequential(AGPF(num_features//2))

        self.up32 = common.UpSampler(2, num_features)
        self.up3_spa2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up3_spamo2 = nn.Sequential(common.RSPB(
            num_features//2, 3, 4, act=act, n_resblocks=num_every_group, norm=None))
        self.up3_fre2 = AGPF(num_features//2)
        self.up3_fre_mo2 = nn.Sequential(AGPF(num_features//2))

        # define tail module
        modules_tail2 = [
            common.ConvBNReLU2D(num_features, out_channels=3, kernel_size=3, padding=1,
                                act=act)]

        self.tail2 = nn.Sequential(*modules_tail2)

        ### fusion part
        conv_fuse2 = []
        for i in range(4):
            conv_fuse2.append(crossFuseBlock(num_features))
        for i in range(4):
            conv_fuse2.append(selfFuseBlock(num_features))
        self.conv_fuse2 = nn.Sequential(*conv_fuse2)
        self.conv_dec_fuse = SPB(num_features, is_shu=False)
    def forward(self, lr):
        #### fre
        stage1_dec=[]
        lr=F.interpolate(lr, scale_factor=self.scale, mode="nearest")
        x = self.head(lr)  # 128
        x = self.conv_fuse[0](x)
        down1 = self.down1(x) # 64
        down1spa, down1freq = torch.split(down1, [down1.size(1)//2, down1.size(1)//2], dim=1)
        down1spa=self.down1_spa(down1spa)
        down1spa = self.down1_spamo(down1spa)
        down1freq = self.down1_fre(down1freq)
        down1freq = self.down1_fre_mo(down1freq)
        down1_mo = torch.cat([down1spa, down1freq], dim=1)
        down1_fuse_mo = self.conv_fuse[1](down1_mo)

        down2 = self.down2(down1_fuse_mo) # 32
        down2spa, down2freq = torch.split(down2, [down2.size(1) // 2, down2.size(1) // 2], dim=1)
        down2spa = self.down2_spa(down2spa)
        down2spa = self.down2_spamo(down2spa)
        down2freq = self.down2_fre(down2freq)
        down2freq = self.down2_fre_mo(down2freq)
        down2_mo = torch.cat([down2spa, down2freq], dim=1)
        down2_fuse_mo = self.conv_fuse[2](down2_mo)


        down3 = self.down3(down2_fuse_mo) # 16
        down3spa, down3freq = torch.split(down3, [down3.size(1) // 2, down3.size(1) // 2], dim=1)
        down3spa = self.down3_spa(down3spa)
        down3spa = self.down3_spamo(down3spa)
        down3freq = self.down3_fre(down3freq)
        down3freq = self.down3_fre_mo(down3freq)
        down3_mo = torch.cat([down3spa, down3freq], dim=1)
        down3_fuse_mo = self.conv_fuse[3](down3_mo)

        neckspa, neckfreq = torch.split(down3_fuse_mo, [down3_fuse_mo.size(1) // 2, down3_fuse_mo.size(1) // 2], dim=1)
        neckspa = self.neck_spa(neckspa)
        neckspa = self.neck_spamo(neckspa)
        neckfreq = self.neck_fre(neckfreq)
        neckfreq = self.neck_fre_mo(neckfreq)
        neck_mo = torch.cat([neckspa, neckfreq], dim=1)
        neck_fuse_mo = self.conv_fuse[4](neck_mo, down3_fuse_mo)
        stage1_dec.append(neck_fuse_mo)

        up1 = self.up1(neck_fuse_mo) # 32
        up1spa, up1freq = torch.split(up1, [up1.size(1) // 2, up1.size(1) // 2], dim=1)
        up1spa = self.up1_spa(up1spa)
        up1spa = self.up1_spamo(up1spa)
        up1freq = self.up1_fre(up1freq)
        up1freq = self.up1_fre_mo(up1freq)
        up1_mo = torch.cat([up1spa, up1freq], dim=1)
        up1_fuse_mo = self.conv_fuse[5](up1_mo,down2_fuse_mo)

        stage1_dec.append(up1_fuse_mo)
        up2 = self.up2(up1_fuse_mo) # 64
        up2spa, up2freq = torch.split(up2, [up2.size(1) // 2, up2.size(1) // 2], dim=1)
        up2spa = self.up2_spa(up2spa)
        up2spa = self.up2_spamo(up2spa)
        up2freq = self.up2_fre(up2freq)
        up2freq = self.up2_fre_mo(up2freq)
        up2_mo = torch.cat([up2spa, up2freq], dim=1)
        up2_fuse_mo = self.conv_fuse[6](up2_mo, down1_fuse_mo)
        stage1_dec.append(up2_fuse_mo)

        up3 = self.up3(up2_fuse_mo) # 128
        up3spa, up3freq = torch.split(up3, [up3.size(1) // 2, up3.size(1) // 2], dim=1)
        up3spa = self.up3_spa(up3spa)
        up3spa = self.up3_spamo(up3spa)
        up3freq = self.up3_fre(up3freq)
        up3freq = self.up3_fre_mo(up3freq)
        up3_mo = torch.cat([up3spa, up3freq], dim=1)
        up3_fuse_mo = self.conv_fuse[7](up3_mo, x)
        stage1_dec.append(up3_fuse_mo)

        res = self.tail(up3_fuse_mo)
        output1=res + lr

        decfeas_new=self.conv_dec_fuse(stage1_dec)
        x2 = self.head2(output1)  # 128
        x2 = self.conv_fuse2[0](x2, decfeas_new[-1])
        down12 = self.down12(x2)  # 64
        down1spa2, down1freq2 = torch.split(down12, [down12.size(1) // 2, down12.size(1) // 2], dim=1)
        down1spa2 = self.down1_spa2(down1spa2)
        down1spa2 = self.down1_spamo2(down1spa2)
        down1freq2 = self.down1_fre2(down1freq2)
        down1freq2 = self.down1_fre_mo2(down1freq2)
        down1_mo2 = torch.cat([down1spa2, down1freq2], dim=1)
        down1_fuse_mo2 = self.conv_fuse2[1](down1_mo2, decfeas_new[-2])

        down22 = self.down22(down1_fuse_mo2)  # 32
        down2spa2, down2freq2 = torch.split(down22, [down22.size(1) // 2, down22.size(1) // 2], dim=1)
        down2spa2 = self.down2_spa2(down2spa2)
        down2spa2 = self.down2_spamo2(down2spa2)
        down2freq2 = self.down2_fre2(down2freq2)
        down2freq2 = self.down2_fre_mo2(down2freq2)
        down2_mo2 = torch.cat([down2spa2, down2freq2], dim=1)
        down2_fuse_mo2 = self.conv_fuse2[2](down2_mo2, decfeas_new[-3])

        down32 = self.down32(down2_fuse_mo2)  # 16
        down3spa2, down3freq2 = torch.split(down32, [down32.size(1) // 2, down32.size(1) // 2], dim=1)
        down3spa2 = self.down3_spa2(down3spa2)
        down3spa2 = self.down3_spamo2(down3spa2)
        down3freq2 = self.down3_fre2(down3freq2)
        down3freq2 = self.down3_fre_mo2(down3freq2)
        down3_mo2 = torch.cat([down3spa2, down3freq2], dim=1)
        down3_fuse_mo2 = self.conv_fuse2[3](down3_mo2, decfeas_new[-4])

        neckspa2, neckfreq2 = torch.split(down3_fuse_mo2, [down3_fuse_mo2.size(1) // 2, down3_fuse_mo2.size(1) // 2], dim=1)
        neckspa2 = self.neck_spa2(neckspa2)
        neckspa2 = self.neck_spamo2(neckspa2)
        neckfreq2 = self.neck_fre2(neckfreq2)
        neckfreq2 = self.neck_fre_mo2(neckfreq2)
        neck_mo2 = torch.cat([neckspa2, neckfreq2], dim=1)
        neck_fuse_mo2 = self.conv_fuse2[4](neck_mo2,down3_fuse_mo2)

        up12 = self.up12(neck_fuse_mo2)  # 32
        up1spa2, up1freq2 = torch.split(up12, [up12.size(1) // 2, up12.size(1) // 2], dim=1)
        up1spa2 = self.up1_spa2(up1spa2)
        up1spa2 = self.up1_spamo2(up1spa2)
        up1freq2 = self.up1_fre2(up1freq2)
        up1freq2 = self.up1_fre_mo2(up1freq2)
        up1_mo2 = torch.cat([up1spa2, up1freq2], dim=1)
        up1_fuse_mo2 = self.conv_fuse2[5](up1_mo2, down2_fuse_mo2)

        up22 = self.up22(up1_fuse_mo2)  # 64
        up2spa2, up2freq2 = torch.split(up22, [up22.size(1) // 2, up22.size(1) // 2], dim=1)
        up2spa2 = self.up2_spa2(up2spa2)
        up2spa2 = self.up2_spamo2(up2spa2)
        up2freq2 = self.up2_fre2(up2freq2)
        up2freq2 = self.up2_fre_mo2(up2freq2)
        up2_mo2 = torch.cat([up2spa2, up2freq2], dim=1)
        up2_fuse_mo2 = self.conv_fuse2[6](up2_mo2, down1_fuse_mo2)

        up32 = self.up32(up2_fuse_mo2)  # 128up2spa2, up2freq2 = torch.split(up22, [up22.size(1) // 2, up22.size(1) // 2], dim=1)
        up3spa2, up3freq2 = torch.split(up32, [up32.size(1) // 2, up32.size(1) // 2], dim=1)
        up3spa2 = self.up3_spa2(up3spa2)
        up3spa2 = self.up3_spamo2(up3spa2)
        up3freq2 = self.up3_fre2(up3freq2)
        up3freq2 = self.up3_fre_mo2(up3freq2)
        up3_mo2 = torch.cat([up3spa2, up3freq2], dim=1)
        up3_fuse_mo2 = self.conv_fuse2[7](up3_mo2, x2)

        res2 = self.tail2(up3_fuse_mo2)
        output = res2 +output1
        return output, output1
from einops import rearrange

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class BiFusion_block(nn.Module):
    def __init__(self, ch,chout, r_2=1):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.conv3 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, ch, 1, 1)), requires_grad=True)

        # self.fc1 = nn.Conv2d(ch, ch // r_2, kernel_size=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(ch // r_2, ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        #
        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=False, relu=False, bias=False)
        self.conv_spa = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, stride=1,
                               groups=ch, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, ch, 1, 1)), requires_grad=True)

        # # bi-linear modelling for both
        # self.W_g = Conv(ch, ch, 1, bn=False, relu=False)
        # self.W_x = Conv(ch, ch, 1, bn=False, relu=False)
        # self.W = Conv(ch, ch, 3, bn=False, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch + ch, chout)


    def forward(self, g, x):
        # # bilinear pooling
        # W_g = self.W_g(g)
        # W_x = self.W_x(x)
        # bp = self.W(W_g * W_x)
        #
        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in
        g=self.conv_spa(g) * self.beta+g_in



        # channel attetion for global branch
        x_in = x
        x = x_in * self.sigmoid(self.sca(x))
        x=self.conv3(x)*self.gamma+x_in

        # x = x.mean((2, 3), keepdim=True)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x], 1))
        return fuse

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class BiFusion_block(nn.Module):
    def __init__(self, ch,chout, r_2=1):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.conv3 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, ch, 1, 1)), requires_grad=True)

        # self.fc1 = nn.Conv2d(ch, ch // r_2, kernel_size=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(ch // r_2, ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        #
        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=False, relu=False, bias=False)
        self.conv_spa = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, stride=1,
                               groups=ch, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, ch, 1, 1)), requires_grad=True)

        # # bi-linear modelling for both
        # self.W_g = Conv(ch, ch, 1, bn=False, relu=False)
        # self.W_x = Conv(ch, ch, 1, bn=False, relu=False)
        # self.W = Conv(ch, ch, 3, bn=False, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch + ch, chout)


    def forward(self, g, x):
        # # bilinear pooling
        # W_g = self.W_g(g)
        # W_x = self.W_x(x)
        # bp = self.W(W_g * W_x)
        #
        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in
        g=self.conv_spa(g) * self.beta+g_in



        # channel attetion for global branch
        x_in = x
        x = x_in * self.sigmoid(self.sca(x))
        x=self.conv3(x)*self.gamma+x_in

        # x = x.mean((2, 3), keepdim=True)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x], 1))
        return fuse

# class selfFuseBlock(nn.Module):
#     def __init__(self, channels, is_shu=True,  shu_reduction=2):
#         super(selfFuseBlock, self).__init__()
#         self.bifusion=BiFusion_block(channels//2,channels)
#         self.channels = channels
#         self.is_shu=is_shu
#         self.channels = channels
#         self.shu_channels = channels // shu_reduction
#         if is_shu:
#             self.shu = shu(self.shu_channels)
#
#     def forward(self, decfea, encfea=None):
#         fa, fb = torch.split(decfea, [decfea.size(1) - self.shu_channels, self.shu_channels], dim=1)
#         if self.is_shu:
#             shu_vi = self.shu(fb)
#             fb = fb + shu_vi
#
#         decfea =self.bifusion(fa,fb)
#         if torch.isnan(decfea).sum()>0:
#             print('feature include NAN!!!!')
#             print(decfea.shape)
#             decfea = torch.nan_to_num(decfea, nan=1e-5, posinf=1e-5, neginf=1e-5)
#         if encfea==None:
#             spa=decfea
#         else:
#             spa=encfea + decfea
#         return spa
class selfFuseBlock(nn.Module):
    def __init__(self, channels):
        super(selfFuseBlock, self).__init__()
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa_att = MDTA(dim=channels)

    def forward(self, decfea, encfea=None):
        decfea = self.spa(decfea)
        decfea = self.spa_att(decfea)+decfea
        if torch.isnan(decfea).sum()>0:
            print('dec feature include NAN!!!!')
            assert torch.isnan(decfea).sum() == 0

            decfea = torch.nan_to_num(decfea, nan=1e-5, posinf=1e-5, neginf=1e-5)
        if encfea==None:
            spa=decfea
        else:
            spa=encfea + decfea
        return spa

class SPB(nn.Module):
    def __init__(self, channels, is_shu=True, shu_reduction=2):
        super(SPB, self).__init__()
        self.channels = channels
        self.shu_channels = channels // shu_reduction
        self.shu = shu_org(self.shu_channels, input_res=256,lowest_res = 256//8, is_shu=is_shu)

    def forward(self, decfeas):
        decshufea=decfeas[-1]
        fa, fb = torch.split(decshufea, [decshufea.size(1) - self.shu_channels, self.shu_channels], dim=1)
        shu_vis = self.shu(fb)
        decfeas_new=[]
        for i,decfea in enumerate(decfeas):
            fa, fb = torch.split(decfea, [decfea.size(1) - self.shu_channels, self.shu_channels], dim=1)
            resi=fa.shape[-1]
            fb = fb + shu_vis[resi]
            decfea_new = torch.cat([fa, fb], dim=1)
            decfeas_new.append(decfea_new)
        return decfeas_new
class shu_org(nn.Module):
    def __init__(self, dim,dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear', input_res = 64,
                 lowest_res = 64//8,
                 tail_sigma_mult = 3,
                 gaussian_at_input_res = False,
                 is_shu=True):
        # bn_layer not used
        super(shu_org, self).__init__()
        self.input_res = input_res
        self.lowest_res = lowest_res
        self.is_shu=is_shu
        if is_shu:
            self.df1 = heterogeneous_filter(
                dim * 2, dim * 2,
                freedom=dfilter_freedom, type=dfilter_type)
            torch.nn.init.normal_(self.df1.weight, mean=1 / (dim * 2), std=0.1 / (dim * 2))
            self.conv = nn.Conv2d(dim*2, dim*2, 1, 1, 0)
        self.act = nn.ReLU()

        self.tail_sigma_mult = tail_sigma_mult
        self.gaussian_at_input_res = gaussian_at_input_res

        self.reslist = [2 ** i for i in range(int(np.log2(self.lowest_res)), int(np.log2(self.input_res)) + 1)]
        reslistrev = self.reslist[::-1]
        self.gaussian_weight_map = {}
        for idx, resi in enumerate(reslistrev):
            if idx != 0:
                gaussianf = gaussian_heatmap_2d(size=[resi, resi // 2 + 1])
                center = np.array([resi // 2 - 1, 0], dtype=float)
                sigma = (resi // 2) / tail_sigma_mult
                variance = np.array([
                    [sigma ** 2, 0],
                    [0, sigma ** 2], ], dtype=float)
                self.gaussian_weight_map[resi] = gaussianf(c=center[None], v=variance[None])
                resi_prev = reslistrev[idx - 1]
                self.gaussian_weight_map[resi_prev][
                (resi_prev // 2 - resi // 2):(resi_prev // 2 + resi // 2), 0:(resi // 2 + 1)] \
                    -= self.gaussian_weight_map[resi]
            elif gaussian_at_input_res:
                gaussianf = gaussian_heatmap_2d(size=[resi, resi // 2 + 1])
                center = np.array([resi // 2 - 1, 0], dtype=float)
                sigma = (resi // 2) / tail_sigma_mult
                variance = np.array([
                    [sigma ** 2, 0],
                    [0, sigma ** 2], ], dtype=float)
                self.gaussian_weight_map[resi] = gaussianf(c=center[None], v=variance[None])
            else:
                self.gaussian_weight_map[resi] = torch.ones([resi, resi // 2 + 1]).float()

        for resi in self.reslist:
            self.gaussian_weight_map[resi] = torch.Tensor(self.gaussian_weight_map[resi]).float()

    def forward(self, x):
        batch,c,h,w=x.shape
        # fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x+1e-8, dim=(-2, -1))
        # ffted = torch.fft.rfftn(x_spectral, dim=fft_dim, norm='ortho')
        # Shift is necessary because the top-left is low frequency (make it at center)
        ffted = torch.cat([
            ffted[:, :, ffted.size(2) // 2 + 1:, :],
            ffted[:, :, :ffted.size(2) // 2 + 1, :]], dim=2)

        if self.is_shu:
            ffted = torch.cat([ffted.real, ffted.imag], dim=1)

            ffted = self.conv(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.act(ffted)

            ffted = self.df1(ffted)

            ffted = torch.complex(ffted[:, 0:c], ffted[:, c:])

        output = {}

        for resi in self.reslist:
            splited_ffted = ffted[:, :, (self.input_res // 2 - resi // 2):(self.input_res // 2 + resi // 2),
                            0:(resi // 2 + 1)].clone()
            splited_ffted = splited_ffted * self.gaussian_weight_map[resi].to(x.device)[None, None]

            # Shift back
            splited_ffted = torch.cat([
                splited_ffted[:, :, resi - resi // 2 - 1:, :],
                splited_ffted[:, :, :resi - resi // 2 - 1, :]], dim=2)
            output[resi] = torch.fft.irfftn(splited_ffted+1e-8, dim=(2, 3))  # /10*resi
        return output
class MDTA(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x=self.norm(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out
from model.ffc import heterogeneous_filter
class shu(nn.Module):
    def __init__(self, dim,dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        # bn_layer not used
        super(shu, self).__init__()

        self.conv = nn.Conv2d(dim*2, dim*2, 1, 1, 0)
        self.act = nn.ReLU()
        self.df1 = heterogeneous_filter(
            dim * 2, dim * 2,
            freedom=dfilter_freedom, type=dfilter_type)
        torch.nn.init.normal_(self.df1.weight, mean=1 / (dim * 2), std=0.1 / (dim * 2))

    def forward(self, x):
        batch,c,h,w=x.shape
        # fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x+1e-8, dim=(-2, -1))
        # ffted = torch.fft.rfftn(x_spectral, dim=fft_dim, norm='ortho')
        # Shift is necessary because the top-left is low frequency (make it at center)
        ffted = torch.cat([
            ffted[:, :, ffted.size(2) // 2 + 1:, :],
            ffted[:, :, :ffted.size(2) // 2 + 1, :]], dim=2)

        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(ffted)

        # ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ifft_shape_slice = [h,w]
        ffted = self.df1(ffted)

        ffted = torch.complex(ffted[:, 0:c], ffted[:,c:])

        ffted = torch.cat([
            ffted[:, :, ffted.size(2) // 2 - 1:, :],
            ffted[:, :, :ffted.size(2) // 2 - 1, :]], dim=2)
        # output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        output = torch.fft.irfftn(ffted+1e-8, dim=(-2, -1))
        return output


import numpy as np
class gaussian_heatmap_2d(object):
    """
    Given a [n x 2] center point coord,
        and [n x 2 x 2] gaussian sigma (std)
        Compute [h x w] np array
        which is 1 at gaussian center and difuss to
        elsewhere as the sigma shows.
    """
    def __init__(self, size, merge_type='max'):
        """
        Args:
            size: (int, int),
                h, w of the output size.
            merge_type: str,
                'max', output value will take the max one
                'add', add values from multiple gaussians.
                    (GMM is such)
        """
        self.size = size
        self.merge_type = merge_type
        h, w = size
        coordh = np.arange(0, h)[:, np.newaxis] * np.ones((1, w))
        coordw = np.arange(0, w)[np.newaxis, :] * np.ones((h, 1))
        self.coord = np.stack([coordh, coordw])
        self.speedup = True

    def __call__(self, c, v):
        """
        Args:
            c: [n x 2] float array,
                the center of each gaussian.
            v: [n x 2 x 2] float array,
                the 2x2 variance matrices on each gaussian.
        Returns:
            x: [h x w] float array,
                the output gaussian heatmap.
        """
        if c.shape[0] != v.shape[0]:
            raise ValueError
        x = np.zeros(self.size, dtype=float)
        for ci, vi in zip(c, v):
            ci = ci[:, np.newaxis, np.newaxis]
            dx = self.coord-ci

            if self.speedup:
                # for speed up
                # only update the value within -1.5sigma <-> 1.5sigma
                try:
                    _, singv, _ = np.linalg.svd(vi)
                except:
                    continue
                singvmax = np.max(singv)
                # this is the variance on the max spread direction
                maxstd = np.sqrt(singvmax) # this is the std
                searchr = int(3*maxstd+1)
                # from -searchr:searchr is the range of search
                ciint = ci.astype(int)
                chint, cwint = ciint[0, 0, 0], ciint[1, 0, 0]
                searchh = [max(min(i, self.size[0]), 0) \
                    for i in [chint-searchr, chint+searchr]]
                searchw = [max(min(i, self.size[1]), 0) \
                    for i in [cwint-searchr, cwint+searchr]]
                sh, sw = searchh[1]-searchh[0], searchw[1]-searchw[0]
                dx = dx[:, searchh[0]:searchh[1], searchw[0]:searchw[1]]
                if sh==0 or sw==0:
                    continue
                # a slide of x
                xref = x[searchh[0]:searchh[1], searchw[0]:searchw[1]]
            else:
                xref = x
                sh, sw = self.size

            try:
                vi_inv = np.linalg.inv(vi)
            except:
                continue
            dx = dx.transpose(1, 2, 0).reshape(-1, 2)
            xi = dx @ vi_inv
            xi = (xi * dx).sum(-1)
            xi = xi.reshape(sh, sw)
            xi = np.exp(-0.5*xi)
            if self.merge_type == 'max':
                # update the memory
                xref[:, :] = np.maximum(xref, xi)
            elif self.merge_type == 'add':
                xref[:, :] = xref + xi
            else:
                raise ValueError
        return x
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input * self.scale
class FreBlock_chaattn(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FreBlock_chaattn, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)
        self.relu=nn.ReLU()
        self.re_scale = Scale(1)


    def forward(self, x_org):
        # print("x: ", x.shape)
        n, c, H, W = x_org.shape
        x=self.conv(x_org)
        y = self.avg_pool(x)
        y = self.conv_du(y.view(n, c,1,1))
        x=x_org+x*y
        msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
        # msF = torch.fft.fftshift(msF, dim=(-2, -1))
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        # out=torch.fft.ifftshift(out, dim=(-2, -1))
        out = torch.abs(torch.fft.irfft2(out+1e-8, s=(H, W)))
        out = self.post(out)
        out = out + self.re_scale(x_org)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out

class Freq_block(nn.Module):
    def __init__(self, dim,dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        super().__init__()
        self.dim = dim
        self.dw_amp_conv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.df1 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.df2 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.dw_pha_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, groups=dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        b,c,h,w = x.shape
        msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
        msF = torch.cat([
            msF[:, :, msF.size(2) // 2 + 1:, :],
            msF[:, :, :msF.size(2) // 2 + 1, :]], dim=2)
        # msF = torch.fft.fftshift(msF, dim=(-2, -1))
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.dw_amp_conv(msF_amp)
        avg_attn = torch.mean(amp_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(amp_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg=self.df1(agg)
        amp_fuse=amp_fuse*agg
        amp_res = amp_fuse - msF_amp
        pha_guide=torch.cat((msF_pha,amp_res),dim=1)
        pha_fuse = self.dw_pha_conv(pha_guide)
        avg_attn = torch.mean(pha_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(pha_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df2(agg)
        pha_fuse = pha_fuse * agg
        pha_fuse=pha_fuse*(2.*math.pi)-math.pi
        # pha_fuse = torch.clamp(pha_fuse, -math.pi, math.pi)
        ## amp_fuse = amp_fuse + msF_amp
        # pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        # out=torch.fft.ifftshift(out, dim=(-2, -1))
        out = torch.cat([
            out[:, :, out.size(2) // 2 - 1:, :],
            out[:, :, :out.size(2) // 2 - 1, :]], dim=2)
        out = torch.abs(torch.fft.irfft2(out+1e-8, s=(h, w)))
        if torch.isnan(out).sum()>0:
            print('freq feature include NAN!!!!')
            assert torch.isnan(out).sum() == 0
            out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out = out + x
        return F.relu(out)
class AGPF(nn.Module):
    def __init__(self, n_feat, n_resblocks=1):
        super(AGPF, self).__init__()
        modules_body = [
            Freq_block(n_feat) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)
class CFS(nn.Module):
    def __init__(self, channels):
        super(CFS, self).__init__()
        self.fre = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, stride=1, padding=1)
        self.spa = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, stride=1, padding=1)
        self.conv_squeeze=nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, spa, fre):
        fre = self.fre(fre)
        spa = self.spa(spa)
        attn = torch.cat([fre, spa], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        fuse = fre * sig[:, 0, :, :].unsqueeze(1) + spa * sig[:, 1, :, :].unsqueeze(1)
        res=self.conv(fuse)
        if torch.isnan(res).sum()>0:
            print('[lsk feature include NAN!!!!')
            assert torch.isnan(res).sum()==0

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res
class crossFuseBlock(nn.Module):
    def __init__(self, channels):
        super(crossFuseBlock, self).__init__()
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa_att = MDTA(dim=channels)
        self.cross_lsk=CFS(channels)

    def forward(self, decfea, encfea=None):
        decfea = self.spa(decfea)
        decfea = self.spa_att(decfea)+decfea
        if torch.isnan(decfea).sum()>0:
            print('decfea feature include NAN!!!!')
            assert torch.isnan(decfea).sum() == 0
            decfea = torch.nan_to_num(decfea, nan=1e-5, posinf=1e-5, neginf=1e-5)
        if encfea==None:
            spa=decfea
        else:
            spa=self.cross_lsk(encfea,decfea)
        return spa

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
if __name__ == '__main__':
    import psutil
    import time
    import os
    net=TSFNet(scale=8)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.4fM" % (total / 1e6))
    net = net.cuda()
    x = torch.rand(1, 3, 32, 32).cuda()
    from thop import profile
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))
    net.eval()
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y,_ = net(x)
        timer.toc()
    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))



