import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class PaddingLayer(nn.Module):
    def __init__(self, padding):
        super(PaddingLayer, self).__init__()
        self.padding = padding
        
    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.padding, self.padding), "constant", 0)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A', expansion=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.expansion = expansion
        
        self.shortcut = nn.Identity()
        if (stride != 1 or in_planes != planes) and option == 'A':
            self.shortcut = PaddingLayer(planes // 4) # option A
        elif option == 'B':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                # nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = F.relu(self.conv1(x))
        # out = F.relu(self.conv1(x.relu()))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlockBN(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A', expansion=1):
        super(BasicBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.expansion = expansion
        
        self.shortcut = nn.Identity()
        if (stride != 1 or in_planes != planes) and option == 'A':
            self.shortcut = PaddingLayer(planes // 4) # option A
        elif option == 'B':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x.relu())))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out
