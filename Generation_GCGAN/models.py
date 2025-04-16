import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import random


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)
        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class GCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GCA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        context = self.global_avg_pool(x)
        context = self.fc1(context)
        context = F.relu(context)
        context = self.fc2(context)
        context = context.view(batch_size, channels, 1, 1)
        spatial_attention = self.softmax(x.view(batch_size, channels, -1))
        spatial_attention = spatial_attention.view(batch_size, channels, height, width)
        out = x + context * spatial_attention
        return out


class GCBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        # Global Context Attention (GCA) block: computes a global context vector
        # from the input features (see Equations (8)-(11) in the paper)
        # Main branch:
        #  - Applies adaptive average pooling to produce a reduced spatial map.
        #  - Then applies a convolution (kernel size 4x4), Swish activation,
        #    and a 1x1 convolution followed by a Sigmoid to generate attention weights.
        # This branch corresponds to the processing that yields the spatial attention
        # and scaling factor (refer to Equations (12)-(13) in the paper).

        super(GCBlock, self).__init__()
        self.global_context_block = GCA(ch_in)
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False)), Swish(),
            spectral_norm(nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False)), nn.Sigmoid()
        )

    def forward(self, feat_small, feat_big):
        feat_small = self.global_context_block(feat_small)
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU()
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    skip = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        batchNorm2d(out_planes)
    )
    return block, skip


class UpBlockCompRes(nn.Module):
    # UpBlockCompRes consists of two parallel branches:
    # 1. The main branch (self.block) applies:
    #    - Nearest-neighbor upsampling (Equation (1): x_up = upsample(x))
    #    - A 3×3 convolution with spectral normalization,
    #      Noise Injection, Batch Normalization, and GLU activation.
    #    - A second convolution block for further feature processing.
    #
    # 2. The skip branch (self.skip) performs a simple upsampling followed by
    #    a 1×1 convolution to adjust channel dimensions.
    #
    # The final output is given by the element-wise sum:
    #   output = Block(x) + Skip(x)     (Corresponds to Equation (7) in the paper)

    def __init__(self, in_planes, out_planes):
        super(UpBlockCompRes, self).__init__()
        self.block, self.skip = UpBlockComp(in_planes, out_planes)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}
        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])
        self.feat_8 = UpBlockCompRes(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockCompRes(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockCompRes(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = GCBlock(nfc[4], nfc[64])
        self.se_128 = GCBlock(nfc[8], nfc[128])
        self.se_256 = GCBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockCompRes(nfc[256], nfc[512])
            self.se_512 = GCBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input):
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)
        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
        )
        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
        )

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {k: int(v * ndf) for k, v in nfc_multi.items()}
        self.im_size = im_size

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False)
        )

        self.se_2_16 = GCBlock(nfc[512], nfc[64])
        self.se_4_32 = GCBlock(nfc[256], nfc[32])
        self.se_8_64 = GCBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32])
        )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


class SimpleDecoder(nn.Module):
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {k: int(v * 32) for k, v in nfc_multi.items()}

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes * 2), GLU()
            )
            return block

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            upBlock(nfc_in, nfc[16]),
            upBlock(nfc[16], nfc[32]),
            upBlock(nfc[32], nfc[64]),
            upBlock(nfc[64], nfc[128]),
            conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
