# %%
import torch
from torch import nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: tuple):
        super().__init__()

        padding = (int(kernel_size[0]/2), int(kernel_size[1]/2))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        hidden_c = F.relu(self.conv1(input))
        hidden_v = F.relu(self.conv2(hidden_c))

        hidden_c = torch.cat((hidden_c, hidden_v))
        hidden_v = F.relu(self.conv3(hidden_c))
        hidden_c = torch.cat((hidden_c, hidden_v))
        hidden_v = F.relu(self.conv4(hidden_c))

        output = self.norm(hidden_v)
        return output

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Dense blocks
        self.dense_block1 = DenseBlock(6, 16, (1, 9))
        self.dense_block2 = DenseBlock(16, 16, (1, 9))
        self.dense_block3 = DenseBlock(16, 4, (1, 9))
        self.dense_block4 = DenseBlock(4, 4, (1, 9))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        
        self.output = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.prepare_input(x)
        x = self.cnn(x)
        x = self.fully_connected(x)
        x = self.output(x)
        return x

    def prepare_input(self, x):
        # x = (N x 64 x 100)
        real = x.real
        imag = x.imag

        amplitude = abs(x)
        phase = x.angle()

        ifft = torch.fft.ifft(x)

        time_input = torch.stack((ifft.real, ifft.imag), 1)
        rectangular  = torch.stack((real, imag), 1)
        polar = torch.stack((amplitude, phase), 1)

        input = torch.cat((rectangular, polar, time_input), 1)
        # input = (N x 6 x 64 x 100)
        return input.float()
        
    # TODO: 
    #   verify number of input channels
    #   check difference between nn and F
    #   check encoder (I think it just converts the output to a model)
    def cnn(self, x):
        x = self.dense_block1(x)
        x = F.avg_pool2d(x, kernel_size=(1,5))

        x = self.dense_block2(x)
        x = F.avg_pool2d(x, kernel_size=(1,2))

        x = self.dense_block3(x)
        x = self.dense_block4(x)
        x = F.avg_pool2d(x, kernel_size=(1,5))

        x = torch.flatten(x, 1)

    def fully_connected(self, x):
        dropout_rate = 0.0
        x = F.relu(self.fc1(x))
        x = F.dropout(x, dropout_rate)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, dropout_rate)
        x = F.relu(self.fc3(x))
        return x