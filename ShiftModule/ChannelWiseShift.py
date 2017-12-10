# coding: utf-8

import numpy
import chainer
from chainer import function_node
from chainer.utils import type_check


def _channelwise_shift_gpu(x, degree):
    b, c, h, w = x.shape
    xp = chainer.cuda.get_array_module(x)
    y = xp.zeros_like(x, dtype=xp.float32)
    chainer.cuda.elementwise(
        'raw T x, int32 ch, int32 h, int32 w, raw int32 degree',
        'T y',
        '''
            int c0 = i / (h * w);
            int in_y = i / w % h;
            int in_x = i % w;
            int c = c0 % ch;
            in_x -= degree[2*c + 1];
            in_y -= degree[2*c + 0];
            if(in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                y = x[in_x + w * (in_y + h * c0)];
            } else {
                y = 0;
            }
        ''',
        'channelwise_shift',
    )(x.reduced_view(), c, h, w, degree.view(), y)

    return y


def channelwise_shift_function(x, degree=None):
    '''
        apply shift operation at each channels in x

        Args:
            x(array): input image
            degree(array): list of shift degree; [[dy,dx], ...]

        Returns:
            array: shifted image
    '''
    xp = chainer.cuda.get_array_module(x)
    if degree is None:
        b, c, h, w = x.shape
        degree = xp.zeros((c, 2))
    elif not isinstance(degree, xp.ndarray):
        b, c, h, w = x.shape
        degree = xp.array(degree)
        cd = degree.shape[0]
        if cd < c:
            degree = xp.vstack([degree, xp.zeros((c - cd, 2))])
    if isinstance(x, numpy.ndarray):
        assert False, 'not implement error'
    return _channelwise_shift_gpu(x, degree)


class ChannelWiseShiftFunction(function_node.FunctionNode):

    def __init__(self, degree):
        '''
            Args:
                degree(numpy.array or cupy.array): degree of shift [[dy1, dx1], ...]
        '''
        self.degree = degree

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4
        )

    def forward(self, inputs):
        x, = inputs
        y = channelwise_shift_function(x, self.degree)
        return y,

    def backward(self, indexes, grad_outputs):
        return ChannelWiseShiftFunction(self.degree * -1).apply(grad_outputs)


def channelwise_shift(x, degree):
    '''
        Args:
            x(Variable): Input image
            degree(numpy.array or cupy.array): The degree of shift [[dy1, dx1], ...]

        Returns:
            ~chainer.Variable: Output variable
    '''
    fnode = ChannelWiseShiftFunction(degree)
    y, = fnode.apply((x,))
    return y
