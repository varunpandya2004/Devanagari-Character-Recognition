import os
import cv2
import glob
import collections
from skimage.io import imread
from tqdm import tqdm
import numpy as np
import torch
import time
import winsound
import pandas as pd
import argparse
# importing the libraries
from test_calibration import create_grouped_result, ece
# for evaluating the model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# PyTorch libraries and modules

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD


# CNN
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 8 * 8, 46)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# CNN model specified in the paper
class Net2(Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 12, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(12),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(12 * 5 * 5, 46)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# DEFINE TRAINING
def train(model, optimizer, criterion, epoch, batch_size):

    quotient = len(train_label) // batch_size
    remainder = len(train_label) % batch_size
    for u in range(quotient):
        x_train = torch.tensor(train_x[u * batch_size:(u + 1) * batch_size], dtype=torch.float)
        y_train = torch.tensor(train_label[u * batch_size:(u + 1) * batch_size], dtype=torch.long)
        model.train()
        tr_loss = 0
        # getting the training set
        # x_train, y_train = train_x, train_y

        '''if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()'''

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        train_losses.append(loss_train)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
    if remainder != 0:
        x_train = torch.tensor(train_x[u * batch_size:(u + 1) * batch_size], dtype=torch.float)
        y_train = torch.tensor(train_label[u * batch_size:(u + 1) * batch_size], dtype=torch.long)
        model.train()
        tr_loss = 0
        # getting the training set
        # x_train, y_train = train_x, train_y

        '''if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()'''

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        train_losses.append(loss_train)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_train)
    if epoch == n_epochs - 1:
        del x_train
        del y_train
    return model


def load_train_data():
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')


    train_img = []
    train_label = []
    # print(len(os.listdir('testdr/train')))
    # count=0
    # pbar=tqdm(total=len(os.listdir('testdr/train')))
    for filepath0 in tqdm(glob.glob(os.path.join('DevanagariHandwrittenCharacterDataset/Train', '*'))):
        # print(count)
        for filepath1 in os.listdir(filepath0):
            img = imread(str(filepath0 + '\\' + filepath1), as_gray=True)
            # added
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
            # img = cv2.equalizeHist(img)
            ret, img = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
            img = img.astype('float32')
            img /= 255.0
            train_img.append(img)
            if filepath0.split('\\')[1][0] == 'c': # character
                train_label.append(int(filepath0.split('_')[1])+9)
            else: # digit
                train_label.append(int(filepath0.split('_')[1]))

    train_x = np.array(train_img)
    # reshape training data
    train_x = train_x.reshape(len(train_img), 1, 32, 32)
    print("Shape of training data(before split): ", train_x.shape, len(train_label))
    return train_x,train_label


def load_test_data():
    # TESTING
    test_filepath = []
    test_img = []
    test_label = []
    # print(len(os.listdir('testdr/train')))
    # count=0
    # pbar=tqdm(total=len(os.listdir('testdr/train')))
    for filepath3 in tqdm(glob.glob(os.path.join('DevanagariHandwrittenCharacterDataset/Test', '*'))):
        # print(count)
        for filepath4 in os.listdir(filepath3):
            test_filepath.append(str(filepath4))
            img = imread(str(filepath3 + '\\' + filepath4), as_gray=True)
            # added
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
            # img = cv2.equalizeHist(img)
            ret, img = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
            # img = cv2.equalizeHist(img)
            # added
            # print(filepath1)
            # cv2.imshow(str(filepath1),img)
            img = img.astype('float32')
            img /= 255.0
            test_img.append(img)
            if filepath3.split('\\')[1][0] == 'c':  # character
                test_label.append(int(filepath3.split('_')[1]) + 9)
            else:  # digit
                test_label.append(int(filepath3.split('_')[1]))

    test_x = np.array(test_img)
    test_x = test_x.reshape(len(test_img), 1, 32, 32)
    return test_x, test_label, test_filepath


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="net")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--calibration", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="result.csv")
    parser.add_argument("--ensemble", type=int, default=1)
    hp = parser.parse_args()
    train_x, train_label = load_train_data()

    # split into training data and validation data
    train_x, valid_x, train_label, valid_label = train_test_split(train_x,train_label,test_size=0.2)
    # shape of training data and validation data
    print("Shape of training data: ", train_x.shape, len(train_label))
    print("Shape of validation data ", valid_x.shape, len(valid_label))

    model_list = []
    optimizer_list = []
    for i in range(hp.ensemble):
        # DEFINING MODEL
        if hp.model == "net":
            model = Net()
        elif hp.model == "net2":
            model = Net2()
        else:
            print("model not recognized, using net as default")
            model = Net()

        if torch.cuda.is_available():
            model = model.cuda()
        model_list.append(model)

        # defining the optimizer
        optimizer = Adam(model.parameters(), lr=hp.learning_rate)
        optimizer_list.append(optimizer)

    # defining the loss function
    criterion = CrossEntropyLoss()

    # checking if GPU is available
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # TRAIN
    # defining the number of epochs
    train_start_time = time.time()
    n_epochs = hp.n_epochs
    batch_size = hp.batch_size
    for i in range(hp.ensemble):
        model = model_list[i]
        optimizer = optimizer_list[i]
        # empty list to store training losses
        train_losses = []
        # shuffle training data to increase variance
        train_x, train_label = shuffle(train_x, train_label)
        # training the model
        for epoch in range(n_epochs):
            train(model, optimizer, criterion, epoch, batch_size)

    print('Train Time (secs)' + str(time.time() - train_start_time))

    # VALIDATION
    if hp.calibration == "temperature":
        temp_list = np.logspace(-2, 2, 100)
        min_nll = 1e9
        best_temp = 1
        x_valid = torch.tensor(valid_x, dtype=torch.float)
        label_valid = torch.tensor(valid_label, dtype=torch.long)
        for temp in temp_list:
            nll = 0
            for i in range(hp.ensemble):
                model = model_list[i]
                with torch.no_grad():
                    output_valid = model(x_valid)

                output_temp = output_valid * temp
                loss = CrossEntropyLoss()
                nll += loss(output_temp, label_valid)

            nll = nll / hp.ensemble
            if nll < min_nll:
                min_nll = nll
                best_temp = temp

        print("min_nll: ", min_nll, "temp: ", best_temp)

    # TEST
    test_x, test_label, test_filepath = load_test_data()
    x_test = torch.tensor(test_x, dtype=torch.float)
    y_test = np.array(test_label)
    print("Shape of test data: ", test_x.shape, y_test.shape)

    # generating predictions for test set
    inference_start_time = time.time()
    ensemble_prob = np.zeros((len(test_label),46))
    for i in range(hp.ensemble):
        model = model_list[i]
        with torch.no_grad():
            output = model(x_test)

        if hp.calibration == "temperature":
            output = output * best_temp

        softmax = Softmax(dim=1)
        prob = softmax(output).cpu().numpy()
        ensemble_prob += prob

    print('Inference Time (secs)' + str(time.time() - inference_start_time))

    ensemble_prob = ensemble_prob / hp.ensemble
    # check the ensemble_prob sums to 1
    prob_sum = np.sum(ensemble_prob, axis=1)
    for i in range(len(prob_sum)):
        if abs(prob_sum[i]-1) > 1e-6:
            print("the prob seems incorrect!")
            break

    predictions = np.argmax(ensemble_prob, axis=1)
    pred_prob = np.max(ensemble_prob, axis=1)
    result = {"file": test_filepath,
              "pred": predictions.tolist(),
              "prob": pred_prob.tolist(),
              "label": y_test.tolist()}

    df = pd.DataFrame(result)
    df.to_csv(hp.output_path,header=True, index=False)

    print("Test accuracy: ", accuracy_score(y_test, predictions))
    grouped_result = create_grouped_result(df)
    ece_ = ece(grouped_result)
    print("Test ECE: ", ece_)
    print('Time taken(mins)' + str((time.time() - start_time) / 60))
    # winsound.Beep(440, 2000)


