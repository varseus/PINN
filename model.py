import torch
from torch import nn

########################################
######## Complex Valued Layers #########
########################################

class PhaseAmpRelu(nn.ReLU):
    """Complex NN Layer that applies ReLU to the magnitude of the inputs, and does nothing to the phase
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.complex(super().forward(torch.real(input)), torch.imag(input))
    
class ComplexLinear(nn.Module):
    """Complex NN Layer that uses to linear layers to model real and imaginary components. 
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real_linear = nn.Linear(in_features, out_features)
        self.imag_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        real = self.real_linear(x.real)
        imag = self.imag_linear(x.imag)
        return torch.complex(real, imag)
    
class ComplexSoftmax(nn.Softmax):
    """Complex NN Layers that softmaxes the real component of input.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.complex(super().forward(input.real), input.imag)

########################################
######### Feed Forward Models ##########
########################################

class FullyConnectedNetwork_Psi(nn.Module):
    """Fully connectred complex valued neural network with 500 inputs, 1 hidden layer, and 3x500 outputs
    """
    def __init__(self):
        super().__init__()
        self.fully_connected_stack = nn.Sequential(
            ComplexSoftmax(),
            nn.Flatten().to(torch.cfloat),
            ComplexLinear(500, 500*30),
            PhaseAmpRelu(),
            ComplexLinear(500*30, 500*3),
            nn.Unflatten(1, (500, 3))
        )
    def forward(self, x):
        x = x.cfloat()
        return self.fully_connected_stack(x)
    

class FullyConnectedNetwork_E(nn.Module):
    """Fully connectred neural network with 500 inputs, 1 hidden layer, and 3 outputs
    """
    def __init__(self):
        super().__init__()
        self.fully_connected_stack = nn.Sequential(
            nn.Flatten(),
            nn.Sigmoid(),
            nn.Linear(500, 500*50),
            nn.ReLU(),
            nn.Linear(500*50, 3),
        )


    def forward(self, x):
        return self.fully_connected_stack(x)


def init_weights_Psi(m):
    """ Reinitialize weights for better fitting for Psi. """
    
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.003, std=0.1)
        if m.bias is not None:
            m.bias.data.zero_()

def init_weights_E(m):
    """ Reinitialize weights for better fitting for E. """

    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.00003, std=.00005)
        if m.bias is not None:
            m.bias.data.zero_()