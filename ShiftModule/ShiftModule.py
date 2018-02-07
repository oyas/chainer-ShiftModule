#
# Shift module
#
# Reference:
#   Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions
#   https://arxiv.org/abs/1711.08141
#

import chainer
import chainer.functions as F
import chainer.links as L

from ShiftModule.ChannelWiseShift import channelwise_shift


class ShiftModule(chainer.Chain):

    def __init__(self, in_channels, mid_channels, out_channels, ksize=3, stride=1, dilate=1,
                nobias=False, initialW=None, initial_bias=None, pre_shift=False):
        '''
            Args:
                in_channels(int) : input channels. Not allow to set 'None'.
                mid_channels(int): mid_channels = (expansion rate) * in_channels
        '''
        super(ShiftModule, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pre_shift = pre_shift

        if self.in_channels*2 == self.out_channels:
            out_ch = in_channels
        else:
            out_ch = out_channels

        with self.init_scope():
            if pre_shift:
                self.pre_shift_ope = ShiftOperation(ksize, dilate)

            self.pre_bn    = L.BatchNormalization(in_channels)
            self.pre_conv  = L.Convolution2D(in_channels, mid_channels, ksize=1,
                                nobias=nobias, initialW=initialW, initial_bias=initial_bias)

            self.shift     = ShiftOperation(ksize, dilate)

            self.post_bn   = L.BatchNormalization(mid_channels)
            self.post_conv = L.Convolution2D(mid_channels, out_ch, ksize=1, stride=stride,
                                nobias=nobias, initialW=initialW, initial_bias=initial_bias)

    def __call__(self, x):
        h = x

        if self.pre_shift:
            h = self.pre_shift_ope(h)

        h = F.relu(self.pre_bn(h))
        h = self.pre_conv(h)

        h = self.shift(h)  # When using chainer v4.0.0b3, you can replace by F.shift()

        h = F.relu(self.post_bn(h))
        h = self.post_conv(h)

        if self.stride > 1:
            x = F.average_pooling_2d(x, ksize=self.stride, stride=self.stride)

        if self.in_channels == self.out_channels:
            h = h + x
        elif self.in_channels*2 == self.out_channels:
            h = F.concat([h, x])

        return h


class ShiftConv(ShiftModule):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, dilate=1,
                 nobias=False, initialW=None, initial_bias=None,
                 expansion=3, pre_shift=False):
        mid_channels = expansion * in_channels
        super(ShiftConv, self).__init__(
                in_channels, mid_channels, out_channels, ksize, stride, dilate,
                nobias, initialW, initial_bias, pre_shift)

#
# Shift Operataion
# This is the same as Shift function in Chainer v4.0.0b3.
#

class ShiftOperation(chainer.Chain):

    def __init__(self, ksize=3, dilate=1):
        super(ShiftOperation, self).__init__()
        self.ksize = _pair(ksize)
        self.dilate = _pair(dilate)
        self.degree = None

        # make list of shift degree in kernel
        kh, kw = self.ksize
        dh, dw = kh//2, kw//2
        self.kernel = [
            (x,y)
            for y in range(-dh, kh-dh, self.dilate[0])
            for x in range(-dw, kw-dw, self.dilate[1])
        ]

        self.not_build = True

    def __call__(self, x):
        if self.not_build:
            b, ch, h, w = x.shape
            gs = ch // len(self.kernel)  # group size
            assert gs > 0, 'channel size is too fewer than kernel size'

            self.degree = self.xp.zeros((ch,2), dtype=self.xp.int32)
            for i in range(len(self.kernel)):
                dx, dy = self.kernel[i]
                self.degree[gs*i:gs*(i+1), 0] = dy
                self.degree[gs*i:gs*(i+1), 1] = dx

            self.not_build = False

        return channelwise_shift(x, self.degree)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


#
# When using CPU, replace ShiftOperation() by old_ShiftOperation()
#

class old_ShiftOperation(chainer.Chain):

    def __init__(self, ksize=3, dilate=1):
        super(old_ShiftOperation, self).__init__()
        self.ksize = _pair(ksize)
        self.dilate = _pair(dilate)

        # make list of shift degree in kernel
        kh, kw = self.ksize
        dh, dw = kh//2, kw//2
        self.kernel = [
            (x,y)
            for y in range(-dh, kh-dh, self.dilate[0])
            for x in range(-dw, kw-dw, self.dilate[1])
        ]

    def __call__(self, x):
        b, ch, h, w = x.shape
        #kh, kw = self.ksize
        #assert h >= kh and w >= kw, 'kernel is too large'
        gs = ch // len(self.kernel)  # group size
        assert gs > 0, 'channel size is too fewer than kernel size'

        out = None
        for i in range(len(self.kernel)):
            # shift
            dx, dy = self.kernel[i]
            group = x[:, gs*i:gs*(i+1), :, :]
            group = shift(group, 3, dx)
            group = shift(group, 2, dy)
            # concat group
            if out is None:
                out = group
            else:
                out = F.concat([out, group])

        if ch % len(self.kernel) > 0:
            # The remaining channels are not shifted
            out = F.concat([out, x[:, gs*len(self.kernel):, :, :]])

        return out


def shift(x, axis, degree):

    if degree == 0:
        return x

    xp = chainer.cuda.get_array_module(x)

    # padding
    s_z = list(x.shape)
    s_z[axis] = abs(degree)
    z = chainer.Variable(xp.zeros(s_z, dtype=xp.float32))

    # shift x
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(max(0,-degree), x.shape[axis]-max(0,degree))
    x = x[slc]

    # concat
    if degree > 0:
        x = F.concat([z, x], axis=axis)
    elif degree < 0:
        x = F.concat([x, z], axis=axis)

    return x


def pad_shift(x, axis, degree):
    # same as shift(), but short and slowly implemented

    if degree == 0:
        return x

    # shift x
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(max(0,-degree), x.shape[axis]-max(0,degree))
    x = x[slc]

    # padding
    pd = [(0,0)] * len(x.shape)
    pd[axis] = (max(0,degree), max(0,-degree))
    x = F.pad(x, pd, 'constant')

    return x
