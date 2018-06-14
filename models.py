import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_planes, s1x1_planes,
                 e1x1_planes, e3x3_planes):
        super().__init__()

        self.s1x1 = nn.Conv2d(in_planes, s1x1_planes, kernel_size=1)
        self.e1x1 = nn.Conv2d(s1x1_planes, e1x1_planes, kernel_size=1)
        self.e3x3 = nn.Conv2d(s1x1_planes, e3x3_planes,
                              kernel_size=3, padding=1)

        self.s1x1_activation = nn.ReLU(inplace=True)
        self.e1x1_activation = nn.ReLU(inplace=True)
        self.e3x3_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.s1x1_activation(self.s1x1(X))

        return torch.cat([
            self.e1x1_activation(self.e1x1(X)),
            self.e3x3_activation(self.e3x3(X)),
        ], dim=1)


class MiniSqueezeNet(nn.Module):

    def __init__(self, in_planes=3, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 16, 32, 32),
            Fire(64, 16, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7, stride=1),
        )

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        X = X.view(X.size(0), -1)
        return self.final_activation(X)
