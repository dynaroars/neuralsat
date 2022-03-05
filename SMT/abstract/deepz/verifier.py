import torch
import torch.nn as nn

import zonotope

class FullyConnected(nn.Module):
    def __init__(self, input_size, fc_layers):
        super().__init__()

        layers = [nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = FullyConnected(2, [2, 2, 2])
    print(net)
    # net.load_state_dict(torch.load("fc1.pt", map_location=torch.device('cpu')))
    x = torch.randn([1, 1, 2, 2])
    y = net(x)

    pred_label = y.max(dim=1)[1].item()
    print('x', x)
    print('y', y)
    print(pred_label)

    model = zonotope.Model(net, eps=0.05, x=x, true_label=pred_label)
