import os
import torch
from torch import nn
from torch import optim
import pandas as pd
from pandas import to_numeric
import numpy as np
import torch.utils.data as Data
from sklearn import preprocessing, utils
from nbdt.models.myMPLnet import myMLP
from sklearn.preprocessing import LabelEncoder
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

nrows = 100000

dataFrame = pd.read_csv('IDS2017.csv', low_memory=False)#, nrows=nrows)
dataFrame = dataFrame.replace('^\s*$', np.nan, regex=True)
dataFrame = dataFrame.replace('^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$', np.nan, regex=True)

dataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
dataFrame = dataFrame.fillna(method='ffill')
dataFrame.loc[dataFrame['Label'].isin(
    ['Web Attack - Sql Injection', 'Web Attack - XSS', 'Web Attack - Brute Force']), 'Label'] = 'Web Attack'
dataFrame = dataFrame.iloc[1:,]

# IDS_classes = {
#     'BENIGN':0,
#     'DoS Hulk':1,
#     'PortScan':2,
#     'DDoS':3,
#     'DoS GoldenEye':4,
#     'FTP-Patator':5,
#     'SSH-Patator':6,
#     'DoS slowloris':7,
#     'DoS Slowhttptest':8,
#     'Web Attack - Brute Force':9,
#     'Web Attack - XSS':9,
#     'Web Attack - Sql Injection':9,
#     'Bot':10,
#     'Infiltration':11,
#     'Heartbleed':12}
# dataFrame = dataFrame.replace(to_replace = IDS_classes, value = None)

dataFrame = dataFrame.sample(frac=1)

col = dataFrame.columns.values.tolist()
col_to_pick = [18, 15, 24, 19, 12, 27, 20, 11, 14, 7, 25, 10, 22, 9, 16, 21, 23, 26, 17, 13]
x = dataFrame.iloc[:,col_to_pick].copy()
print("--->", x.shape)
names = x.columns
x = x.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=names)

split_rate = 0.6
split = int(len(x) * split_rate)

x_train = x[0:split]
x_test = x[split:len(x)]

# x = dataFrame.iloc[:,list(range(83))]
y = dataFrame.values[:,[83]]
le = LabelEncoder()
le.fit(["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator", "DoS slowloris",
        "DoS Slowhttptest", "Web Attack", "Bot", "Infiltration", "Heartbleed"])

y = le.fit_transform(y)
print(le.classes_)
# pd.DataFrame.info(y)
# print(y.describe())

y_train = y[0:split]
y_test = y[split:len(y)]

# print(x)
print(y_train)

mlp = myMLP(x.shape[1], len(le.classes_))

EPOCH = 10
BATCH_SIZE = 10000
LR = 0.001
optimizer = optim.SGD(mlp.parameters(), lr = LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()

train_dataset = Data.TensorDataset(torch.Tensor(x_train.values), torch.from_numpy(y_train))
train_loader = Data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)

correct = 0
# training and testing
for epoch in range(EPOCH):
  for step, (t_x, t_y) in enumerate(train_loader):
    mlp.training = True;
    output = mlp(t_x)
    loss = loss_func(output, t_y)
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()                                # apply gradients

    if step % 50 == 0:
        mlp.training = False;
        test_output = mlp(torch.Tensor(x_test.values))  # (samples, time_step, input_size)
        # pred_y = torch.max(test_output, 1)[1].data.numpy()
        pred_y = torch.max(test_output, 1)[1]

        # print('pred_y = torch.max(test_output, 1)[0]:', torch.max(test_output, 1)[0])
        # print('pred_y = torch.max(test_output, 1)[1]:',pred_y)
        # y_test = torch.from_numpy(y_test.iloc[:,0].values).long().data.numpy()
        target = torch.from_numpy(y_test)
        # accuracy = float((pred_y == y_test).astype(int).sum()) / float(y_test.size)
        correct = pred_y.eq(target).sum().item()

        # print('pred_y.shape:', pred_y.shape)
        # print('pred_y:', pred_y)
        # print('target:', target)
        # print('pred_y.eq(target).sum().item():', pred_y.eq(target).sum().item())
        # print('float(y_test.size):', float(y_test.size))

        accuracy = correct / float(y_test.size)
        print('Epoch: ', epoch + 1, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.8f' % accuracy)
        # print('value:', pred_y)


# print 10 predictions from test data
# test_output = mlp(torch.Tensor(x_test.values)[:10].view(-1, 840, 84))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
# print(y_test[:10], 'real number')


filepath = os.path.join('nbdt/models/', 'myMLP_model.pth.tar')
torch.save(mlp, filepath)
