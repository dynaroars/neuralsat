import torch
import torch.nn as nn

from prophecy.torch import CachedReLU


class CorinaNet(nn.Module):
    def __init__(self):
        super(CorinaNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 2),
            CachedReLU(),
            nn.Linear(2, 2),
            CachedReLU(),
            nn.Linear(2, 2),
        ])
        self._init_weights()

    def _init_weights(self):
        self.layers[0].weight.data = torch.tensor([
            [1., -1.],
            [1., 1.]
        ])
        self.layers[0].bias.data.fill_(0)
        self.layers[2].weight.data = torch.tensor([
            [0.5, -0.2],
            [-0.5, 0.1]
        ])
        self.layers[2].bias.data.fill_(0)
        self.layers[4].weight.data = torch.tensor([
            [1., -1.],
            [-1., 1.]
        ])
        self.layers[4].bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@torch.no_grad()
def test_concrete():
    model = CorinaNet()
    x = torch.tensor([[2.5, 2.5]])
    y = model(x).argmax()
    print(y)

    decision_signatures = []
    for m in model.modules():
        if isinstance(m, CachedReLU):
            decision_signatures.append(m.decision.squeeze().tolist())
    print(decision_signatures)


@torch.no_grad()
def test_range():
    model = CorinaNet()
    x_lower = torch.tensor([[10, -10]])
    x_upper = torch.tensor([[-10, -10]])
    n_samples = 31

    for i in range(n_samples):
        x = x_lower + (x_upper - x_lower) * i / (n_samples - 1)
        if i == 0:
            continue
        y = model(x).argmax()
        print(x)
        print('Pred:', y.item())

        decision_signatures = []
        for m in model.modules():
            if isinstance(m, CachedReLU):
                decision_signatures.append(m.decision.squeeze().tolist())
        print('Decision signatures:', decision_signatures)
        print()


if __name__ == '__main__':
    # test_concrete()
    test_range()
