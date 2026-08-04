from torch import nn


class WAFModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(WAFModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
