import torch
import torch.nn as nn


class SimpleNeuralNet(nn.Module):
    def __init__(self, p=0):
        super(SimpleNeuralNet, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # [batch_size, 32, 96, 96]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.MaxPool2d(5)
        self.hidden = nn.Linear(32, 256)
        self.out = nn.Linear(256, 30)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.bn(x)  # [batch_size, 1, 96, 96]
        x = self.act(self.conv1(x))  # [batch_size, 32, 96, 96]
        x = self.pool(x)  # [batch_size, 32, 48, 48]

        x = self.act(self.conv2(x))  # [batch_size, 64, 48, 48]
        x = self.dropout(self.pool(x))  # [batch_size, 64, 24, 24]

        x = self.act(self.conv3(x))  # [batch_size, 128, 24, 24]
        x = self.dropout(self.pool(x))  # [batch_size, 128, 12, 12]

        x = self.act(self.conv4(x))  # [batch_size, 32, 10, 10]
        x = self.pool(x)  # [batch_size, 32, 5, 5]

        x = self.global_pool(x)  # [batch_size, 32, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 32]

        x = self.dropout(self.act(self.hidden(x)))  # [batch_size, 256]
        x = self.out(x)  # [batch_size, 30]

        return x


class InceptionNeuralNet(nn.Module):
    def __init__(self, p=0):
        super(InceptionNeuralNet, self).__init__()

        self.bn = nn.BatchNorm2d(num_features=1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3)
        self.inception_model1 = InceptionModule(32, 16, (8, 16), (8, 16), 16)
        self.inception_model2 = InceptionModule(64, 32, (16, 32), (16, 32), 32)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.MaxPool2d(5)
        self.hidden = nn.Linear(32, 256)
        self.out = nn.Linear(256, 30)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.bn(x)  # [batch_size, 1, 96, 96]
        x = self.act(self.conv1(x))  # [batch_size, 32, 96, 96]
        x = self.pool(x)  # [batch_size, 32, 48, 48]

        x = self.inception_model1(x)  # [batch_size, 64, 24, 24]
        x = self.dropout(x)

        x = self.inception_model2(x)  # [batch_size, 128, 12, 12]
        x = self.dropout(x)

        x = self.act(self.conv2(x))  # [batch_size, 32, 10, 10]
        x = self.pool(x)  # [batch_size, 32, 5, 5]

        x = self.global_pool(x)  # [batch_size, 32, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 32]

        x = self.dropout(self.act(self.hidden(x)))  # [batch_size, 256]
        x = self.out(x)  # [batch_size, 30]

        return x


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.ReLU(),
    )


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(InceptionModule, self).__init__()

        self.branch1 = conv(in_channels=in_channels, out_channels=out_channels1, kernel_size=1, stride=2)

        self.branch2 = nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels2[0], kernel_size=1),
            conv(in_channels=out_channels2[0], out_channels=out_channels2[1], kernel_size=3, stride=2, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels3[0], kernel_size=1),
            conv(in_channels=out_channels3[0], out_channels=out_channels3[1], kernel_size=3, padding=1),
            conv(in_channels=out_channels3[1], out_channels=out_channels3[1], kernel_size=3, stride=2, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        branch1, branch2, branch3, branch4 = self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return x



