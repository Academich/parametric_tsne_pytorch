from torch import nn


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


class NeuralMapper(nn.Module):

    def __init__(self, dim_input, dim_emb=2):
        super().__init__()
        self.linear_1 = nn.Linear(dim_input, dim_input)
        self.bn_1 = nn.BatchNorm1d(dim_input)
        self.linear_2 = nn.Linear(dim_input, dim_input)
        self.bn_2 = nn.BatchNorm1d(dim_input)
        self.linear_3 = nn.Linear(dim_input, dim_input)
        self.bn_3 = nn.BatchNorm1d(dim_input)
        self.linear_4 = nn.Linear(dim_input, dim_emb)
        self.relu = nn.ReLU()

        self.apply(weights_init)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.linear_2(self.relu(x))
        x = self.bn_2(x)
        x = self.linear_3(self.relu(x))
        x = self.bn_3(x)
        x = self.linear_4(self.relu(x))
        return x
