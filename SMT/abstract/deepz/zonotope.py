import logging
import torch
import torch.nn as nn

from layers import modLayer, ReLU

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 2
NUM_CLASSES = 2


class Model(nn.Module):
    """
    Creates a copy of provided model with `eps_terms` as new images, along batch axis.
    """

    def __init__(self, model: nn.Module, eps: float, x: torch.Tensor, true_label: int):
        super().__init__()
        layers = [modLayer(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)

        print(self.net)

        x_max, x_min = torch.clamp(x.data + eps, max=1), torch.clamp(x.data - eps, min=0)
        # print(x)
        # print(x_max)
        # print(x_min)
        self.x = (x_max + x_min) * 0.5

        eps_terms = x_max - self.x
        eps_terms = torch.diag(torch.ones(INPUT_SIZE * INPUT_SIZE) * eps_terms.flatten())
        self.eps_terms = eps_terms.reshape((INPUT_SIZE * INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE))
        print(self.eps_terms.shape)
        print(self.x.shape)
        print(torch.cat([self.x, self.eps_terms], dim=0).shape)

        self.true_label = true_label
        self.forward()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

    def forward(self) -> torch.Tensor:
        return self.net(torch.cat([self.x, self.eps_terms], dim=0))

    # def updateParams(self):
    #     # Calculates the gradient of `loss` wrt to ReLU slopes.
    #     loss = torch.clamp(-self._min_diff, min=0).sum()

    #     self.optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
        # self.optimizer.step()

    def parameters(self) -> torch.Tensor:
        # A generator to allow gradient descent on `slope` of ReLUs.
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, "slope"):
                yield layer.slope

    @staticmethod
    def getExtremum(predictions: torch.Tensor, minima: bool, label: int) -> torch.Tensor:
        # Calculates extremum values as defined by `max_config_values` and `min_config_values`
        condition = (predictions[:, label] < 0) if minima else (predictions[:, label] > 0)
        eps_config = condition.float() * 2 - 1
        eps_config[0] = 1.0
        return torch.mm(eps_config[None, :], predictions).squeeze()

    def verify(self) -> bool:
        self._zono_pred = self.forward()

        # A matrix that stores difference of activations between `true_label` and label `l` in zonotope form.
        # We check if any of these differences can obtain a negative value anytime.
        difference_matrix = self._zono_pred[:, self.true_label, None] - self._zono_pred
        self._min_diff = difference_matrix[0] - torch.abs(difference_matrix[1:]).sum(dim=0)
        if torch.any(self._min_diff < 0):
            logger.debug(f"Min difference @ `label {self._min_diff.argmin().item()}`: {self._min_diff.min():.4f}")
            return False

        min_config_values = self.getExtremum(self._zono_pred, minima=True, label=self.true_label)
        logger.debug(f"Values @ minimum of `true label: {self.true_label}`: {min_config_values}")
        if min_config_values.argmax().item() != self.true_label:
            return False

        for label in range(NUM_CLASSES):
            max_config_values = self.getExtremum(self._zono_pred, minima=False, label=label)
            logger.debug(f"Values @ maximum of `label: {label}`: {max_config_values}")
            if max_config_values.argmax().item() != self.true_label:
                return False

        return True