import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

_assert_if_size_mismatch = True


def get_norm_layer(name='bn', **kwargs):
    if name == 'bn':
        return nn.BatchNorm2d
    elif name == 'in':
        return nn.InstanceNorm2d
    elif name == 'none':
        return Identity
    else:
        raise ValueError('{norm_layer} not supported')


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):

        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                   normalization(out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                                   normalization(out_channels),
                                   nn.ReLU())

    def forward(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class PartialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = PartialConv2d(
            in_channels, out_channels, kernel_size, padding=1)

        self.conv2 = nn.Sequential(
            normalization(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            normalization(out_channels),
            nn.ReLU()
        )

    def forward(self, inputs, mask=None):
        outputs = self.conv1(inputs, mask)
        outputs = self.conv2(outputs)
        return outputs


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect',
                 act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                    padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                    padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=BasicBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, same_num_filt=False, conv_block=BasicBlock):
        super().__init__()

        num_filt = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    # Before refactoring, it was a nn.Sequential with only one module.
                                    # Need this for backward compatibility with model checkpoints.
                                    nn.Sequential(
                                        nn.Conv2d(num_filt, out_channels, 3, padding=1)
                                    )
                                    )
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            if _assert_if_size_mismatch:
                raise ValueError(
                    f'input2 size ({inputs2.shape[2:]}) does not match upscaled inputs1 size ({in1_up.shape[2:]}')
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


class UNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        more_layers: Additional down/up-sample layers.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
        norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
        last_act: Last layer activation. One of 'sigmoid', 'tanh' or None.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """

    def __init__(
            self,
            num_input_channels=3,
            num_output_channels=3,
            feature_scale=4,
            more_layers=0,
            upsample_mode='bilinear',
            norm_layer='bn',
            last_act='sigmoid',
            conv_block='partial'
    ):
        super().__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))

        self.num_input_channels = num_input_channels[:5]

        if conv_block == 'basic':
            self.conv_block = BasicBlock
        elif conv_block == 'partial':
            self.conv_block = PartialBlock
        elif conv_block == 'gated':
            self.conv_block = GatedBlock
        else:
            raise ValueError('bad conv block {}'.format(conv_block))

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # norm_layer = get_norm_layer(norm_layer)

        self.start = self.conv_block(self.num_input_channels[0], filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels[1], conv_block=self.conv_block)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels[2], conv_block=self.conv_block)
        self.down3 = DownsampleBlock(filters[2], filters[3] - self.num_input_channels[3], conv_block=self.conv_block)
        self.down4 = DownsampleBlock(filters[3], filters[4] - self.num_input_channels[4], conv_block=self.conv_block)

        if self.more_layers > 0:
            self.more_downs = [
                DownsampleBlock(filters[4], filters[4], conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [UpsampleBlock(filters[4], upsample_mode, same_num_filt=True, conv_block=self.conv_block)
                             for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = UpsampleBlock(filters[3], upsample_mode, conv_block=self.conv_block)
        self.up3 = UpsampleBlock(filters[2], upsample_mode, conv_block=self.conv_block)
        self.up2 = UpsampleBlock(filters[1], upsample_mode, conv_block=self.conv_block)
        self.up1 = UpsampleBlock(filters[0], upsample_mode, conv_block=self.conv_block)

        # Before refactoring, it was a nn.Sequential with only one module.
        # Need this for backward compatibility with model checkpoints.
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1)
        )

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())
        else:
            self.final = self.final

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        if isinstance(self.conv_block, PartialBlock):
            eps = 1e-9
            masks = [(x.sum(1) > eps).float() for x in inputs]
        else:
            masks = [None] * len(inputs)

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        in64 = self.start(inputs[0], mask=masks[0])

        mask = masks[1] if self.num_input_channels[1] else None
        down1 = self.down1(in64, mask)

        if self.num_input_channels[1]:
            down1 = torch.cat([down1, inputs[1]], 1)

        mask = masks[2] if self.num_input_channels[2] else None
        down2 = self.down2(down1, mask)

        if self.num_input_channels[2]:
            down2 = torch.cat([down2, inputs[2]], 1)

        mask = masks[3] if self.num_input_channels[3] else None
        down3 = self.down3(down2, mask)

        if self.num_input_channels[3]:
            down3 = torch.cat([down3, inputs[3]], 1)

        mask = masks[4] if self.num_input_channels[4] else None
        down4 = self.down4(down3, mask)
        if self.num_input_channels[4]:
            down4 = torch.cat([down4, inputs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more_layers - idx - 2]
                up_ = l(up_, prevs[self.more_layers - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)

class UNet2(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        more_layers: Additional down/up-sample layers.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
        norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
        last_act: Last layer activation. One of 'sigmoid', 'tanh' or None.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """

    def __init__(
            self,
            num_input_channels=3,
            num_output_channels=3,
            feature_scale=4,
            more_layers=0,
            upsample_mode='bilinear',
            norm_layer='bn',
            last_act='sigmoid',
            conv_block='partial'
    ):
        super().__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))

        self.num_input_channels = num_input_channels[:5]

        if conv_block == 'basic':
            self.conv_block = BasicBlock
        elif conv_block == 'partial':
            self.conv_block = PartialBlock
        elif conv_block == 'gated':
            self.conv_block = GatedBlock
        else:
            raise ValueError('bad conv block {}'.format(conv_block))

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # norm_layer = get_norm_layer(norm_layer)

        self.start = self.conv_block(self.num_input_channels[0], filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels[1], conv_block=self.conv_block)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels[2], conv_block=self.conv_block)
        self.down3 = DownsampleBlock(filters[2], filters[3] - self.num_input_channels[3], conv_block=self.conv_block)
        # self.down4 = DownsampleBlock(filters[3], filters[4] - self.num_input_channels[4], conv_block=self.conv_block)

        if self.more_layers > 0:
            self.more_downs = [
                DownsampleBlock(filters[4], filters[4], conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [UpsampleBlock(filters[4], upsample_mode, same_num_filt=True, conv_block=self.conv_block)
                             for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        # self.up4 = UpsampleBlock(filters[3], upsample_mode, conv_block=self.conv_block)
        self.up3 = UpsampleBlock(filters[2], upsample_mode, conv_block=self.conv_block)
        self.up2 = UpsampleBlock(filters[1], upsample_mode, conv_block=self.conv_block)
        self.up1 = UpsampleBlock(filters[0], upsample_mode, conv_block=self.conv_block)

        # Before refactoring, it was a nn.Sequential with only one module.
        # Need this for backward compatibility with model checkpoints.
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1)
        )

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())
        else:
            self.final = self.final

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        if isinstance(self.conv_block, PartialBlock):
            eps = 1e-9
            masks = [(x.sum(1) > eps).float() for x in inputs]
        else:
            masks = [None] * len(inputs)

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        in64 = self.start(inputs[0], mask=masks[0])

        mask = masks[1] if self.num_input_channels[1] else None
        down1 = self.down1(in64, mask)

        if self.num_input_channels[1]:
            down1 = torch.cat([down1, inputs[1]], 1)

        mask = masks[2] if self.num_input_channels[2] else None
        down2 = self.down2(down1, mask)

        if self.num_input_channels[2]:
            down2 = torch.cat([down2, inputs[2]], 1)

        mask = masks[3] if self.num_input_channels[3] else None
        down3 = self.down3(down2, mask)

        if self.num_input_channels[3]:
            down3 = torch.cat([down3, inputs[3]], 1)

        # mask = masks[4] if self.num_input_channels[4] else None
        # down4 = self.down4(down3, mask)
        # if self.num_input_channels[4]:
        #     down4 = torch.cat([down4, inputs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more_layers - idx - 2]
                up_ = l(up_, prevs[self.more_layers - idx - 2])
        else:
            up_ = down3

        # up4 = self.up4(up_, down3)
        up3 = self.up3(up_, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)

if __name__ == "__main__":
    net = UNet(5, 3, 2, 0).cuda()
    input = torch.randn(1, 5, 512, 640).cuda()
    out = net(input)
    print(out.shape)