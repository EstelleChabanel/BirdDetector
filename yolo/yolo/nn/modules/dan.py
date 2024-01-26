"""Domain Adaptation modules"""

import torch
import torch.nn as nn
from torch.autograd import Function


__all__ = ('GradReversal', 'Conv_', 'AdaptiveAvgPooling', 'AvgPooling', 'Conv_BN', 'MaxPool', 'Conv_BN_MaxPool')


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
        #print("grad output entering GRL", grad_output)
        #print("what is ctx", ctx)
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

revgrad = RevGrad.apply

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
        '''
        self.pool.register_full_backward_hook(
                lambda pool, grad_input, grad_output: print(f"Pool: input shape: {grad_input[0].shape}, output shape: {grad_output[0].shape}")
            )
        self.fc1.register_backward_hook(
                lambda fc1, grad_input, grad_output: print(f"Linear: input shap: {grad_input[0].shape}, output shape: {grad_output[0].shape}")
            )
        self.flat.register_full_backward_hook(
                lambda flat, grad_input, grad_output: print(f"Flat: input shap: {grad_input[0].shape}, output shape: {grad_output[0].shape}")
            )
        
        self.pool.register_forward_hook(
                lambda layer, _, output: print(f"{layer}: {output.shape}")
            )
        self.fc1.register_forward_hook(
                lambda layer, _, output: print(f"{layer}: {output.shape}")
            )
        self.flat.register_forward_hook(
                lambda layer, _, output: print(f"{layer}: {output.shape}")
            ) 
        '''       
        
        #self.pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        return x



class AvgPooling(nn.Module):
    """Adatvie Average Pooling layer for the Feature Distance computation"""
    def __init__(self, c1=1):
        """Initialize AvgPooling layer with given arguments including activation."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(c1)
        self.flat = nn.Flatten()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.pool(x)
        x = self.flat(x)
        return x


class Conv_BN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.relu = nn.SiLU() #(inplace=True)  #nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Conv_BN_MaxPool(nn.Module):
    def __init__(self, c1, c2, k=2, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MaxPool(nn.Module):
    def __init__(self, c1, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # Convolutional layers
        #self.pool = nn.MaxPool2d(kernel_size=c1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(c1)

    def forward(self, x): 
        x = self.pool(x)
        return x
