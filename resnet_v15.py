import torch
import torch.nn as nn
import torch.nn.functional as functional
from config import RetrieverVLConfig_medium

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bias=False, bn=False, active=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if active else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class Bottleneck(nn.Module):
    def __init__(self, cin, out, stride=1, expand=4):
        super().__init__()
        mid = out // expand
        self.block = nn.Sequential(
            Conv_Batch_Active(cin, mid, 1, bn=True),
            Conv_Batch_Active(mid, mid, 3, stride, 1, bn=True),
            Conv_Batch_Active(mid, out, 1, bn=True, active=False)
        )

        self.shortcut = nn.Identity() if cin == out and stride == 1 else \
                        Conv_Batch_Active(cin, out, 1, stride, bn=True, active=False)

    def forward(self, x):
        return functional.relu(self.block(x) + self.shortcut(x))

class SimpleResBlock(nn.Module):
    def __init__(self, cin, out):
        super().__init__()
        self.pre_norm = nn.LayerNorm(cin)

        self.proj = nn.Sequential(
            nn.Linear(cin, cin),
            nn.GELU(),
            nn.Linear(cin, out)
        )

        self.shortcut = nn.Linear(cin, out)
        
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x) + self.shortcut(x)

class ResNetV15(nn.Module):
    def __init__(self, block, layers, channels, config):
        super(ResNetV15, self).__init__()

        self.conv1 = Conv_Batch_Active(3, 64, 7, 2, 3, bn=True)
        self.conv2 = self._make_layer(block, layers[0], channels[0], channels[1], maxpool=True)
        self.conv3 = self._make_layer(block, layers[1], channels[1], channels[2])
        self.conv4 = self._make_layer(block, layers[2], channels[2], channels[3])
        self.conv5 = self._make_layer(block, layers[3], channels[3], channels[4])
        
        # self.positional_embedding = nn.Parameter(torch.randn(config.spacial_dim ** 2 + 1, config.img_hidden_size) / config.img_hidden_size ** 0.5)
        self.mlp = SimpleResBlock(config.img_hidden_size, config.hidden_size)

    @staticmethod
    def _make_layer(block, num_layer, cin, out, maxpool=False):
        if maxpool:
            layers = [nn.MaxPool2d(3, 2, 1),
                      block(cin, out)]
        else:
            layers = [block(cin, out, 2)]
        for i in range(num_layer-1):
            layers.append(block(out, out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.flatten(start_dim=2).permute(0, 2, 1)  # NCHW -> N(HW)C
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # N(HW+1)C
        # x = x + self.positional_embedding[None, :, :].to(x.dtype)  # N(HW+1)C
        x = self.mlp(x)

        return x

def resnet50_v15(config):
    model = ResNetV15(Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], config)
    return model

if __name__ == '__main__':
    config = RetrieverVLConfig_medium()
    x = resnet50_v15(config)
    y = torch.zeros([10,3,448,448])
    z = x(y)