# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from ShiftModule import ShiftModule


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            #self.conv2 = models.ShiftModule.ShiftModule(ch)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2, shift=False, ex_ch=None):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            if shift:
                self.add_link(ShiftModule(out_size, ex_ch, out_size, ksize=(3,3), dilate=2, pre_shift=True))
            else:
                self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, shift=False):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 16, 3, 1, 1, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(16)
            if shift:
                self.res2 = Block(3, 16, 16, 32, 1, shift=shift, ex_ch=64)
                self.res3 = Block(3, 32, 16, 32, shift=shift, ex_ch=64)
                self.res4 = Block(3, 32, 16, 64, shift=shift, ex_ch=64)
                self.res5 = Block(3, 64, 32, 128, shift=shift, ex_ch=128)
            else:
                self.res2 = Block(3, 16, 16, 32, 1, shift=shift)
                self.res3 = Block(3, 32, 16, 64, shift=shift)
                self.res4 = Block(3, 64, 32, 128, shift=shift)
                self.res5 = Block(3, 128,64, 128, shift=shift)
            self.fc = L.Linear(None, 10)

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x))
        #h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        s = h.shape
        h = F.average_pooling_2d(h, (s[2],s[3]), stride=1)
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
