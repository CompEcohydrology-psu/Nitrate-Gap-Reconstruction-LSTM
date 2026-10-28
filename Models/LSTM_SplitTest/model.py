"""CudnnLstm and LSTM model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from dropout import createMask, DropMask


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW", gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):

        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)


        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]


        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,
                self.hiddenSize,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,
                self.hiddenSize,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )


        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, args, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.args = args
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr
        )


        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.relu = F.relu

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(outLSTM)

        return out


class CudnnLstmModel2(torch.nn.Module):
    """
    Same layout as CudnnLstmModel: Linear(nx→H)+ReLU → LSTM → LSTM → Linear(H→ny).
    Two stacked CudnnLstm layers (both H→H); readout uses the top layer sequence output.
    """

    def __init__(self, *, nx, ny, hiddenSize, args, dr=0.5):
        super(CudnnLstmModel2, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 2
        self.args = args
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm1 = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr
        )
        self.lstm2 = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr
        )
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.relu = F.relu

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        h1, _ = self.lstm1(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        h2, _ = self.lstm2(h1, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(h2)
        return out
