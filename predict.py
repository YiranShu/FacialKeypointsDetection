import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

TEST_PATH = './data/test.csv'
CKPT_PATH = './ckpt/inception_model.pth'


def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)

    return X


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, color='red', marker='+', s=10)


if __name__ == '__main__':
    X_test = load_data(TEST_PATH)
    X_test = Variable(torch.from_numpy(X_test), requires_grad=False)
    model = load_checkpoint(CKPT_PATH)
    model.cpu()

    fig = plt.figure(figsize=(6, 6))
    for i in range(25):
        axis = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
        y_test = model(X_test[i].unsqueeze(0))
        plot_sample(X_test[i], y_test.flatten(), axis)
    plt.show()
    fig.savefig('predicted.png')

