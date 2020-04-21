import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import models

TRAIN_PATH = './data/training.csv'
INCP_CKPT_PATH = './ckpt/inception_model.pth'
SIMPLE_CKPT_PATH = './ckpt/simple_model.pth'


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)

    y = df[df.columns[:-1]].values
    y = (y - 48) / 48
    y = y.astype(np.float32)

    return X, y


def train(model, num_epoch, train_losses, validation_losses, train_loader, validation_loader):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_epoch_loss = 0.0
        validation_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            inputs, landmarks = data
            inputs = inputs.to(device)
            landmarks = landmarks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_epoch_loss += loss.item()

            if i % 10 == 9:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        train_losses.append(running_epoch_loss / i)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, landmarks = data
                inputs = inputs.to(device)
                landmarks = landmarks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, landmarks)
                validation_loss += loss.item()

        validation_losses.append(validation_loss / i)


if __name__ == '__main__':
    X, y = load_data(TRAIN_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)
    X_train = Variable(torch.from_numpy(X_train), requires_grad=False)
    X_test = Variable(torch.from_numpy(X_test), requires_grad=False)
    y_train = Variable(torch.from_numpy(y_train), requires_grad=False)
    y_test = Variable(torch.from_numpy(y_test), requires_grad=False)

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    validation_set = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16,
                                              shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16,
                                               shuffle=False, num_workers=2)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    model = models.SimpleNeuralNet(p=0.25) # p = 0.25
    model = model.to(device)

    train_losses = []
    validation_losses = []

    time_start = time.time()
    train(model, 140, train_losses, validation_losses, train_loader, validation_loader)
    time_end = time.time()

    print('Finished Training! Running time: ' + str(time_end - time_start) + ' seconds')
    checkpoint = {'model': models.SimpleNeuralNet(p=0.25),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, SIMPLE_CKPT_PATH)

    f = plt.figure()
    plt.plot(train_losses, linewidth=3, label='train')
    plt.plot(validation_losses, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()
    f.savefig('loss_simple.png')

    model = models.InceptionNeuralNet(p=0.25)  # p = 0.25
    model = model.to(device)

    train_losses = []
    validation_losses = []

    time_start = time.time()
    train(model, 250, train_losses, validation_losses, train_loader, validation_loader)
    time_end = time.time()

    print('Finished Training! Running time: ' + str(time_end - time_start) + ' seconds')
    checkpoint = {'model': models.InceptionNeuralNet(p=0.25),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, INCP_CKPT_PATH)

    f = plt.figure()
    plt.plot(train_losses, linewidth=3, label='train')
    plt.plot(validation_losses, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()
    f.savefig('loss_inception_with_dropout.png')







