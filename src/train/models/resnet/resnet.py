import torch.nn.functional as F
import torch.nn as nn
import torch

from .resnet_utils import _weights_init, BasicBlock, BasicBlockBN

try:
    from ..models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        """
        Fallback wrapper in case timm isn't installed
        """
        return func
                

class PreLayer(nn.Module):

    def __init__(self, planes):
        super().__init__()
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        # return F.relu(self.bn1(self.conv1(x)))
        # return F.relu(self.conv1(x))
        return self.conv1(x)

class PostLayer(nn.Module):

    def __init__(self, dim, num_classes=10):
        super().__init__()
        # self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x = x.relu()
        # print(x.shape)
        x = x.mean(dim=[2, 3]) # average pooling
        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # x = self.linear2(self.linear1(x).relu())
        x = self.linear2(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='B'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.planes = 32
        pre_layer = PreLayer(self.in_planes)
        # layers = [PreLayer()]
        assert len(num_blocks)
        layers = self._make_layer(block, self.planes, num_blocks[0], stride=1, expansion=1, option=option)
        layers[0] = nn.Sequential(pre_layer, layers[0])
        if len(num_blocks) > 1:
            layers += self._make_layer(block, self.planes, num_blocks[1], stride=2, expansion=1, option=option)
        if len(num_blocks) > 2:
            layers += self._make_layer(block, self.planes, num_blocks[2], stride=2, expansion=1, option=option)
        layers[-1] = nn.Sequential(layers[-1], PostLayer(dim=self.planes, num_classes=num_classes))
        
        self.layers = nn.ModuleList(layers)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes, 
                    planes=planes, 
                    stride=stride, 
                    option=option,
                    expansion=expansion,
                )
            )
            self.in_planes = planes * expansion
        return layers

    def forward(self, x):
        for layer in self.layers:
            # print(f'{x.shape=}')
            x = layer(x)
        return x

@register_model
def resnet32B(*args, **kwargs):
    return ResNet(BasicBlock, [9, 3, 3], 10, 'B')


@register_model
def resnet3(*args, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1], 10, 'B')


@register_model
def resnet4(*args, **kwargs):
    return ResNet(BasicBlock, [2, 1, 1], 10, 'B')


@register_model
def resnet5(*args, **kwargs):
    return ResNet(BasicBlock, [3, 1, 1], 10, 'B')


@register_model
def resnet6(*args, **kwargs):
    return ResNet(BasicBlock, [4, 1, 1], 10, 'B')


@register_model
def resnet7(*args, **kwargs):
    return ResNet(BasicBlock, [5, 1, 1], 10, 'B')


@register_model
def resnet8(*args, **kwargs):
    return ResNet(BasicBlock, [6, 1, 1], 10, 'B')


@register_model
def resnet9(*args, **kwargs):
    return ResNet(BasicBlock, [7, 1, 1], 10, 'B')


@register_model
def resnet10(*args, **kwargs):
    return ResNet(BasicBlock, [8, 1, 1], 10, 'B')


@register_model
def resnet11(*args, **kwargs):
    return ResNet(BasicBlock, [9, 1, 1], 10, 'B')

@register_model
def resnet12(*args, **kwargs):
    return ResNet(BasicBlock, [9, 2, 1], 10, 'B')
    
@register_model
def resnet13(*args, **kwargs):
    return ResNet(BasicBlock, [9, 3, 1], 10, 'B')

@register_model
def resnet14(*args, **kwargs):
    return ResNet(BasicBlock, [9, 3, 2], 10, 'B')
    
@register_model
def resnet15(*args, **kwargs):
    return ResNet(BasicBlock, [9, 3, 3], 10, 'B')

@register_model
def resnet16(*args, **kwargs):
    return ResNet(BasicBlock, [9, 4, 3], 10, 'B')

@register_model
def resnet17(*args, **kwargs):
    return ResNet(BasicBlock, [9, 5, 3], 10, 'B')

@register_model
def resnet18(*args, **kwargs):
    return ResNet(BasicBlock, [9, 6, 3], 10, 'B')

@register_model
def resnet_toy(*args, **kwargs):
    return ResNet(BasicBlockBN, [12, 8, 4], 10, 'B')

@register_model
def resnet30BN(*args, **kwargs):
    return ResNet(BasicBlockBN, [10, 10, 10], 10, 'B')

@register_model
def resnet21BN(*args, **kwargs):
    return ResNet(BasicBlockBN, [12, 6, 3], 10, 'B')

@register_model
def resnet24BN(*args, **kwargs):
    return ResNet(BasicBlockBN, [12, 8, 4], 10, 'B')

@register_model
def resnet36BN(*args, **kwargs):
    return ResNet(BasicBlockBN, [12, 12, 12], 10, 'B')

if __name__ == "__main__":
    model = resnet_toy()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(model)
    print(y)
    
    torch.onnx.export(
        model,
        x,
        'test_resnet2.onnx',
        opset_version=12,
    )