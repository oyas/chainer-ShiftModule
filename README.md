# ShiftModule for chainer

Reference:
   [Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions](https://arxiv.org/abs/1711.08141)

## Train

	$ python3 train_shift.py -m shift -g 0

## Usage

	L.Convolution2D(in_ch, out_ch, ksize=3, stride=1, pad=1)

replace to

	from ShiftModule import ShiftModule
	ShiftModule(in_ch, mid_ch, out_ch, ksize=3)
