
""" TasNet training network wrapper. """

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        net (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn
        self.cast = ops.Cast()

    def construct(self, padded_mixture, mixture_lengths, padded_source):
       # Print = ops.Print()
        #Print(padded_mixture.shape)
        padded_mixture = self.cast(padded_mixture, mindspore.float32)
        padded_source = self.cast(padded_source, mindspore.float32)
        estimate_source = self._net(padded_mixture)
        loss, _, _, _ = self._loss(padded_source, estimate_source, mixture_lengths)
       # ops.Print()(loss)
        return loss
