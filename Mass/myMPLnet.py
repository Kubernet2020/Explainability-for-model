# model_urls = {
#     ('wrn28_10', 'TinyImagenet200'): 'https://drive.google.com/drive/folders/1ghrDwo8uK_B8El90Xf-1FPhqJvpneqkf?usp=sharing'
# }

import os
import torch
from torch import nn
from torch import optim
import pandas as pd
from pandas import to_numeric
import numpy as np
import torch.utils.data as Data
from torch.nn import Softmax
from sklearn import preprocessing, utils

__all__ = ('myMLP')
model_urls = {

}
class myMLP(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.training = False;
        self.layer1 = nn.Linear(inputSize, 64)
        self.dropout1 = nn.Dropout(p = 0.3)
        self.layer2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p = 0.3)
        self.layer3 = nn.Linear(64, outputSize)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = nn.functional.relu(x)

        x = nn.functional.log_softmax(x, dim=1)

        return x

class myIDS_Out_DataSet(Data.Dataset):
    def __init__(self, values, classes):
        self.values = values
        self.classes = classes

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx], self.labels[idx]

