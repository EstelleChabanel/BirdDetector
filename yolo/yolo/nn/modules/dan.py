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
                
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        return x
    

class AdaptiveAvgPooling_bis(nn.Module):
    """Adatvie Average Pooling layer for the Domain Classifier"""
    def __init__(self, c1=1, c2=2):
        """Initialize AvgPooling layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(576, c1, kernel_size=3, stride=1, padding=1)
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
        x = self.conv(x)
        print("before pooling", x.size())
        x = self.pool(x)
        print(x.size())
        x = self.flat(x)
        print(x.size())
        x = self.fc1(x)
        print("after fc1", x.size())
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
        self.relu = nn.ReLU()
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
    
class MaxPool_(nn.Module):
    def __init__(self, c1, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # Convolutional layers
        #self.pool = nn.MaxPool2d(kernel_size=c1, stride=2)
        self.pool = nn.MaxPool2d(c1)

    def forward(self, x): 
        x = self.pool(x)
        return x
    
class Concat_(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x1, x2):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat((x1,x2), self.d)


DomainClassifierNetwork = nn.Sequential(
    GradReversal(),
    Conv_(576, 256, 128),
    AdaptiveAvgPooling(1),
).to('cuda')


DomainClassifierNetwork_m = nn.Sequential(
    GradReversal(),
    Conv_(384, 128, 64),
    AdaptiveAvgPooling(1),
).to('cuda')

DomainClassifierNetwork_s = nn.Sequential(
    GradReversal(),
    Conv_(192, 64, 32),
    AdaptiveAvgPooling(1),
).to('cuda')

class MultiDomainClassifier(nn.Module):
    def __init__(self): 
        super().__init__()
        self._alpha = torch.tensor(1., requires_grad=False)
        self.conv_l = Conv_(576, 256, 128)
        self.avgpool = AdaptiveAvgPooling(1)
        self.conv_m = Conv_(384, 128, 64)
        self.conv_s = Conv_(192, 64, 32)

    def forward(self, x1, x2, x3):
        x1 = revgrad(x1, self._alpha)
        x1 = self.conv_s(x1)
        x1 = self.avgpool(x1)

        x2 = revgrad(x2, self._alpha)
        x2 = self.conv_m(x2)
        x2 = self.avgpool(x2)

        x3 = revgrad(x3, self._alpha)
        x3 = self.conv_l(x3)
        x3 = self.avgpool(x3)

        return x1, x2, x3


class multiFeatDomainClassifier(nn.Module):
    def __init__(self): 
        super().__init__()
        self._alpha = torch.tensor(1., requires_grad=False)
        self.conv0 = Conv_BN(192, 64)
        self.conv1 = Conv_BN(128, 64)
        self.conv2 = Conv_BN(384, 128)
        self.conv2bis = Conv_BN(256, 128)
        self.pool = MaxPool_(4)
        self.pool1 = MaxPool_(2)
        self.concat = Concat_(1)
        self.conv30 = Conv_BN(128, 32)
        self.conv3 = Conv_BN(64, 32)
        self.pool2 = MaxPool(10)
        self.conv4 = Conv_BN(576, 256)
        self.pool3 = MaxPool(3)
        self.conv5 = Conv_BN(32, 16)
        self.conv6 = Conv_BN(16, 1)
        self.adaptavg = AdaptiveAvgPooling(1)

    def forward(self, x1, x2, x3):
        x1 = revgrad(x1, self._alpha)
        x1 = self.conv0(x1)
        x1 = self.pool(x1)

        x2 = revgrad(x2, self._alpha)
        x2 = self.conv2(x2)
        x2 = self.pool1(x2)
        x2 = self.conv1(x2)

        x12 = self.concat(x1,x2)
        x12 = self.pool1(self.conv3(self.conv1(x12)))

        x3 = revgrad(x3, self._alpha)
        x3 = self.conv4(x3)
        x3 = self.conv2bis(x3)
        x3 = self.pool1(x3)
        x3 = self.conv1(x3)
        x3 = self.conv3(x3)

        x = self.concat(x12, x3)
        x = self.pool1(self.conv3(x))
        x = self.conv6(self.conv5(x))
        x = self.adaptavg(x)
        return x


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
multiFeatDomainClassifierNetwork = mySequential(
    multiFeatDomainClassifier()
).to('cuda')


FeaturesUnspacifier = nn.Sequential(
    AvgPooling(1),
).to('cuda')


MultiDomainClassifierNetwork = mySequential(
    MultiDomainClassifier()
)



FeaturesDomainClassifierNetwork = nn.Sequential(
    GradReversal(),
    AdaptiveAvgPooling_bis(1),
).to('cuda')