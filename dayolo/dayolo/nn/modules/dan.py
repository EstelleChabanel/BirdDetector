"""Domain Adaptation modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = ('GradReversal', 'ConvReLU', 'Conv_', 'AdaptiveAvgPooling', 'Dense', 'Classifyy')


class RevGrad(Function):
    """
    inspired from https://github.com/janfreyberg/pytorch-revgrad/tree/master
    """

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradReversal(nn.Module):
    """
    Gradient Reversal Layer for domain classifier network
    This layer has no parameters, and simply reverses the gradient
    in the backward pass.
    inspired from https://github.com/janfreyberg/pytorch-revgrad/tree/master
    """

    def __init__(self, alpha=1., *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)
    
    def forward(self, input_):
        revgrad = RevGrad.apply
        return revgrad(input_, self._alpha)
    

class Conv_(nn.Module):
    """Convolution layer for the Domain Classifier, with ReLU activation function"""
    def __init__(self, c1, c2, c3, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c3)

        self.conv3 = nn.Conv2d(c3, 1, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)   
        x = self.pool(x)    
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x
    

class AdaptiveAvgPooling(nn.Module):
    """Adatvie Average Pooling layer for the Domain Classifier"""
    def __init__(self, c1=1, c2=2):
        """Initialize AvgPooling layer with given arguments including activation."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(c1)
        self.fc1 = nn.Linear(c1, c2)
        self.flat = nn.Flatten()
        #self.pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.flat(x)
        return x
    



class Classifyy(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        #self.fc1 = nn.Linear(128 * 2 * 2, 64)
        print("what is c1 ?!", c1)
        self.fc1 = nn.Linear(1, c2)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, c2)

    def forward(self, x):
        print(x.shape, "before fc1")
        x = self.fc1(x)
        print(x.shape, "after fc1")
        #x = self.relu3(x)
        #print(x.shape, "before fc2")
        #x = self.fc2(x)
        #print(x.shape, "la taille")
        return x




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvReLU(nn.Module):
    """Convolution layer for the Domain Classifier, with ReLU activation function"""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.conv(x))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))




class Dense(nn.Module):
    """Convolution layer for the Domain Classifier, with ReLU activation function"""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.dense = nn.Linear(c1, 1)
        #self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return torch.squeeze(self.dense(x))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.dense(x)



