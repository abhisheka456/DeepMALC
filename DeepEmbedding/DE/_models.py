import torch.nn as nn
import torch.nn.functional as F


class DEC(nn.Module):
    def __init__(self, image_size, final_size):
        super(DEC, self).__init__()
        self.fc0 = nn.Linear(image_size, 500)
        self.bn0 = nn.BatchNorm1d(500)
        self.fc1 = nn.Linear(500, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 2000)
        self.bn2 = nn.BatchNorm1d(2000)
        self.fc5 = nn.Linear(2000, 10)

        self.fc6 = nn.Linear(10, 2000)
        self.bn6 = nn.BatchNorm1d(2000)
        self.fc7 = nn.Linear(2000, 500)
        self.bn7 = nn.BatchNorm1d(500)
        self.fc8 = nn.Linear(500, 500)
        self.bn8 = nn.BatchNorm1d(500)
        self.fc9 = nn.Linear(500, image_size)

        self.fc10 = nn.Linear(10, 500)
        self.bn10 = nn.BatchNorm1d(500)
        self.fc11 = nn.Linear(500, 500)
        self.bn11 = nn.BatchNorm1d(500)
        self.fc12 = nn.Linear(500, 500)
        self.bn12 = nn.BatchNorm1d(500)
        self.fc15 = nn.Linear(500, 2000)
        self.bn15 = nn.BatchNorm1d(2000)
        self.fc16 = nn.Linear(2000, 500)
        self.bn16 = nn.BatchNorm1d(500)
        self.fc17 = nn.Linear(500, 100)
        self.bn17 = nn.BatchNorm1d(100)
        self.fc18 = nn.Linear(100, final_size)

        self.fc0.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc5.bias.data.fill_(0)
        self.fc6.bias.data.fill_(0)
        self.fc7.bias.data.fill_(0)
        self.fc8.bias.data.fill_(0)
        self.fc9.bias.data.fill_(0)
        self.fc10.bias.data.fill_(0)
        self.fc11.bias.data.fill_(0)
        self.fc12.bias.data.fill_(0)
        self.fc15.bias.data.fill_(0)
        self.fc16.bias.data.fill_(0)
        self.fc17.bias.data.fill_(0)
        self.fc18.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc0(x)
        out = F.relu(self.bn0(out))
        out = self.fc1(out)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))

        out1 = self.fc5(out)

        out = self.fc6(out1)
        out = F.relu(self.bn6(out))
        out = self.fc7(out)
        out = F.relu(self.bn7(out))
        out = self.fc8(out)
        out = F.relu(self.bn8(out))
        dense = self.fc9(out)

        out = self.fc10(out1)
        out = F.relu(self.bn10(out))
        out = self.fc11(out)
        out = F.relu(self.bn11(out))
        out = self.fc12(out)
        out = F.relu(self.bn12(out))

        out = self.fc15(out)
        dense1 = F.relu(self.bn15(out))
        out = self.fc16(dense1)
        dense2 = F.relu(self.bn16(out))
        out = self.fc17(dense2)
        dense3 = F.relu(self.bn17(out))
        out = self.fc18(dense3)

        return out1, dense, dense1, dense2, dense3, out



