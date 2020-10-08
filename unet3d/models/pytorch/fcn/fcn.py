from torch import nn


class FCN(nn.Module):
    def __init__(self, hidden_layers_list, n_inputs, n_outputs):
        super().__init__()
        _layers = list()
        _n_inputs = n_inputs
        for hidden_layer in hidden_layers_list:
            _layer = nn.Linear(_n_inputs, hidden_layer)
            _n_inputs = hidden_layer
            _layers.append(_layer)
            _layers.append(nn.ReLU())
        _layers.append(nn.Linear(_n_inputs, n_outputs))
        self.network = nn.Sequential(*_layers)

    def forward(self, x):
        return self.network(x)
