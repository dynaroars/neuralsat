import numpy as np
import torch


def scale_saliency_map_np(image_3d, percentile=99, dim=0):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=dim)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def diverge_saliency_map_np(image_3d, percentile=99, dim=0):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=dim)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)


class IntegratedGradient:

    "Attribution method"
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device


    def attribute(self, x, steps=20, baseline=None):
        x = x.to(self.device)
        
        # baselines
        if baseline is None:
            # baseline = torch.randn(x.shape, requires_grad=False).to(x.device)
            baseline = torch.ones_like(x, requires_grad=False, device=self.device) * x.mean()
        self._check(x, baseline)
        
        # predict label
        output = self.model(x)
        pred_label = output.max(1)[1]
        
        # inputs
        x.requires_grad_(True)
        X, delta_X = self._get_X_and_delta(x, baseline, steps)
        
        # integrated gradients
        grad = torch.autograd.grad(self.model(X)[:, pred_label].sum(), X)[0]
        # grad = grad.cpu().numpy()
        grad = delta_X * (grad[:-1] + grad[1:]) / 2.
        ig_grad = grad.sum(dim=0, keepdims=True)
        
        return ig_grad, pred_label[0]
                      
                                         
    def _get_X_and_delta(self, x, baseline, steps):
        alphas = torch.linspace(0, 1, steps + 1).view(-1, 1, 1, 1).to(x.device)
        delta = (x - baseline)
        x = baseline + alphas * delta
        return x, (delta / steps).detach()
        
        
    def _check(self, x, baseline):
        if x.shape != baseline.shape:
            raise ValueError(f'input shape should equal to baseline shape. '
                             f'Got input shape: {x.shape}, '
                             f'baseline shape: {baseline.shape}') 

