import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import numpy as np

limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_decay=1.0,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        lamba=1.0,
        local_rep=False,
        **kwargs
    ):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = (
            torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        )
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode="fan_out")

        self.qz_loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2
        )

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else -torch.sum(0.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = (
                F.sigmoid(self.qz_loga)
                .view(1, self.in_features)
                .expand(batch_size, self.in_features)
            )
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample=self.training)
            xin = input.mul(z)
            output = xin.mm(self.weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            output.add_(self.bias)
        return output

    def __repr__(self):
        s = (
            "{name}({in_features} -> {out_features}, droprate_init={droprate_init}, "
            "lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, "
            "local_rep={local_rep}"
        )
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        **kwargs
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.floatTensor = (
            torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        )
        self.use_bias = False
        self.weights = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.qz_loga = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode="fan_in")

        self.qz_loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2
        )

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = (
            torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 3).sum(2).sum(1)
        )
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = (
            0
            if not self.use_bias
            else -torch.sum((1 - q0) * (0.5 * self.prior_prec * self.bias.pow(2) - self.lamba))
        )
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        ppos = torch.sum(1 - self.cdf_qz(0))
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]
        ) + 1  # for rows
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]
        ) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(
            self.dim_z, 1, 1, 1
        )
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def prune(self, botk):
        qz = self.qz_loga
        qz_shape = qz.size()

        if len(qz_shape) == 4:
            qz = qz.view(-1, qz_shape[-1])

        idx = int(botk * float(qz.size()[0]))
        qz_sorted, _ = qz.sort(dim=0)
        threshold = qz_sorted[idx : idx + 1]
        mask = qz >= threshold
        qz = mask.float() * qz
        self.qz_loga = torch.nn.Parameter(qz.view(qz_shape))
        
        #apply to weights
        w = self.weights
        w = w * qz.view(self.dim_z, 1, 1, 1)
        self.weights = torch.nn.Parameter(w)

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        if self.local_rep or not self.training:
            print(np.count_nonzero(self.weights.cpu().detach().numpy()) / np.prod(self.weights.size()))
            output = F.conv2d(
                input_, self.weights, b, self.stride, self.padding, self.dilation, self.groups
            )
            z = self.sample_z(output.size(0), sample=self.training)
            return output.mul(z)
        else:
            weights = self.sample_weights()
            output = F.conv2d(
                input_, weights, None, self.stride, self.padding, self.dilation, self.groups
            )
            return output

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, "
            "lamba={lamba}, local_rep={local_rep}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class TDConv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dropout=0.5,
        dropout_botk=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        **kwargs
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param weight_decay: Strength of the L2 penalty
        """
        super(TDConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.weight_decay = weight_decay
        self.floatTensor = (
            torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.weight = Parameter(
            self.floatTensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.dropout = dropout
        self.dropout_botk = dropout_botk

        self.reset_parameters()
        self.input_shape = None
        print(self)

        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode="fan_in")

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, thres_std=1.0):
        pass

    def _reg_w(self, **kwargs):
        logpw = -torch.sum(self.weight_decay * 0.5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = -torch.sum(self.weight_decay * 0.5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        ppos = self.out_channels
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]
        ) + 1  # for rows
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]
        ) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.bias is not None:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops, expected_l0

    def targeted_dropout(self, w):
        drop_rate = self.dropout
        targ_perc = self.dropout_botk

        # print("w_orig: ", w)

        w_shape = w.size()
        w = w.view(-1, w_shape[-1])

        norm = w.abs()
        idx = int(targ_perc * float(w.size()[0]))
        norm_sorted, _ = norm.sort(dim=0)
        threshold = norm_sorted[idx]
        mask = norm < threshold[None, :]

        if not self.training:
            w = (1.0 - mask.float()) * w
            w = w.view(w_shape)
            return w

        cuda0 = torch.device("cuda:0")
        dropout_mask = torch.rand(w.size(), device=cuda0) < drop_rate
        mask = dropout_mask & mask
        w = (1.0 - mask.float()) * w
        w = w.view(w_shape)
        return w

    def prune(self, botk):
        w = self.weight
        w_shape = w.size()
        w = w.view(-1, w_shape[-1])
        norm = w.abs()
        idx = int(botk * float(w.size()[0]))
        norm_sorted, _ = norm.sort(dim=0)
        threshold = norm_sorted[idx : idx + 1]
        mask = norm >= threshold
        w = mask.float() * w
        w = w.view(w_shape)
        self.weight = torch.nn.Parameter(w)

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        weight = self.targeted_dropout(self.weight)
        output = F.conv2d(
            input_, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return output

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "dropout={dropout}, dropout_botk={dropout_botk}, "
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
