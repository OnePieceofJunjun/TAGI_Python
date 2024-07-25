import os, io, time, requests, zipfile
# import datasets as datasets
import torch
import numpy as np
import ssl
import pandas as pd
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import *
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
ssl._create_default_https_context = ssl._create_unverified_context


def create_data(dataset, data_size, feature_range):
    """Different kinds of dataset and tasks"""
    # Binary Classification Dataset
    if dataset == "dataset_moon":
        X, Y = make_moons(n_samples=data_size, noise=0.15, random_state=0)
    elif dataset == "dataset_circles":
        X, Y = make_circles(n_samples=data_size, noise=0.01, random_state=0)
    if dataset == "xor":
        data_path = 'Data_test/xor.csv'
        raw_df = pd.read_csv(data_path, sep=";", skiprows=1, header=None)#\\s = space;+/_/...
        data = raw_df.values[:, 0:2]
        target = raw_df.values[:, 2]
        X = data
        Y = target
    if dataset == "and":
        data_path = 'Data_test/and.csv'
        raw_df = pd.read_csv(data_path, sep=";", skiprows=1, header=None)#\\s = space;+/_/...
        data = raw_df.values[:, 0:2]
        target = raw_df.values[:, 2]
        X = data
        Y = target
    # Multi Classification Dataset
    elif dataset == "dataset_MNIST":
        def one_hot_encoding(labels, num_classes):
            # Create a tensor of zeros with the shape (num_samples, num_classes)
            one_hot_labels = torch.zeros(labels.size(0), num_classes)
            # Use scatter to fill the corresponding position of the one-hot tensor with 1
            one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
            return one_hot_labels
        # def hierarchical_binary_tree(labels)
        #     labels4=[0,00,0,0,0,0,0,0]
        #     l1l2ll9
        #     weights[123456789]
        #
        # def hbt(labels, num_classes):
        #     # Create a tensor of zeros with the shape (num_samples, num_classes)
        #     hbt_labels = torch.zeros(labels.size(0), num_classes-1)
        #     # Use scatter to fill the corresponding position of the one-hot tensor with 1
        #     if odds
        #         if>4
        #         ...
        #      elif
        #          if<3
        #     one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        #     return one_hot_labels
        #       make the hierarcical vector label for each numbers

        # Load the MNIST dataset
        from torchvision import datasets, transforms
        data_size = 1500
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

        # Get the first 'data_size' samples
        data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=data_size, shuffle=True)
        for images, labels in data_loader:
            X = images.view(-1, 28 * 28)  # Flatten the images into 1D arrays
            Y = one_hot_encoding(labels, num_classes=10)  # 10 classes for MNIST

        print("X shape:", X.shape)
        print("Y shape:", Y.shape)

    # Dataset for regression
    elif dataset == "dataset_1DToy_regression":
        # X_1 = np.random.uniform(-4, -2, size=int(0.5 * data_size))
        # X_2 = np.random.uniform(-2, 2, size=int(0 * data_size))
        # X_3 = np.random.uniform(2, 4, size=int(0.5 * data_size))
        # X = np.concatenate((X_1, X_2, X_3))
        X = np.random.uniform(-4, 4, size=data_size)
        Y = np.power(X, 3) + np.random.normal(0, 3, size=data_size)
        # Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)
    elif dataset == "dataset_heart":


        t = np.linspace(0, 2 * np.pi, data_size)
        X = 16 * np.sin(t) ** 3
        Y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

        X += np.random.normal(0, 0.5, data_size)
        Y += np.random.normal(0, 0.5, data_size)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)


    elif dataset == "dataset_boston":
       # X, Y = load_boston(return_X_y=True)
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        X = data
        Y = target

    elif dataset == "dataset_concrete":
        df = pd.read_excel(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
        X = df.drop(df.columns[-1], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_energy":
        zip_url = 'https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip'
        r = requests.get(zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_excel(z.open('ENB2012_data.xlsx'))
        X = df.drop(df.columns[[-1, -2]], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_wine_quality":
        df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                         sep=';')
        X = df.drop('quality', axis=1).to_numpy()
        Y = df['quality'].to_numpy()

    elif dataset == "dataset_naval_propulsion":
        zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI CBM Dataset.zip'
        r = requests.get(zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open('UCI CBM Dataset/data.txt'), sep='  ', header=None)
        X = df.iloc[:, :16].to_numpy()
        Y = df.iloc[:, 16].to_numpy()

    elif dataset == "dataset_yacht":
       # df = pd.read_fwf('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data').dropna()
       zip_url = 'https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip'
       r = requests.get(zip_url)
       z = zipfile.ZipFile(io.BytesIO(r.content))
       df = pd.read_csv(z.open('yacht_hydrodynamics.data'), sep=' +', on_bad_lines='warn', header=None)
       X = df.iloc[:, :6].to_numpy()
       Y = df.iloc[:, 6].to_numpy()

    elif dataset == "dataset_kin8nm":
        url = 'https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff'
        data = pd.read_csv(url).values
        X = data[:, :-1]
        Y = data[:, -1]

    elif dataset == "dataset_power":
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pd.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx').values
        X = data[:, :-1]
        Y = data[:, -1]

    elif dataset == "dataset_year":
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'

        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pd.read_csv('/tmp/YearPredictionMSD.txt', delimiter=',', header=None)
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols].values
        data = data[:463810, ]
        X = data[:, :-1]
        Y = data[:, -1]


    x_scaler = None
    y_scaler = None
    if not dataset == "dataset_1DToy_regression":
        x_scaler = preprocessing.MinMaxScaler(feature_range=(feature_range[0], feature_range[1]))
        # x_scaler = preprocessing.StandardScaler()
        X = x_scaler.fit_transform(X)
        output_scale = False
        if output_scale:
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            Y = (Y - Y_mean) / Y_std
            y_scaler = [Y_mean, Y_std]
    return X, Y, x_scaler, y_scaler


def accuracy_binary(y_true, y_pred, variance):
    y_pred = np.round(y_pred)
    acc = accuracy_score(y_true, y_pred)
    ave_log_likelihood = float(variance.sum() / y_pred.shape[0])
    return acc, ave_log_likelihood

    # acc = accuracy_score(y_true.numpy(), y_pred)
    # ave_log_likelihood = float(variance.sum() / y_pred.shape[0])
    # return acc, ave_log_likelihood

def accuracy_multi(y_true, y_pred, variance):
    y_true_labels = torch.argmax(y_true, dim=1)
    y_pred_labels = torch.argmax(torch.tensor(y_pred), dim=1)

#undo one hot coding
    acc = accuracy_score(y_true_labels.numpy(), y_pred_labels.numpy())
    ave_log_likelihood = float(variance.sum() / y_pred_labels.shape[0])

    classes = unique_labels(y_true_labels.numpy(), y_pred_labels.numpy())
    plot_confusion_matrix(y_true_labels.numpy(), y_pred_labels.numpy(), classes)


    # Save the plot as an image
    plt.savefig('confusion_matrix.png')
    return acc, ave_log_likelihood
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    normalize = False
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def NLL_regression(y_true, y_pred, y_cov):
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true.squeeze()
    y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred.squeeze()
    y_cov = y_cov.numpy().squeeze() if isinstance(y_cov, torch.Tensor) else y_cov.squeeze()
    # s1 =
    s1 = np.sum((y_true - y_pred) ** 2 / y_cov)
    s2_1 = y_true.shape[0] * np.log(2 * np.pi)
    s2_2 = np.sum(np.log(y_cov))
    # result = (s1 + s2_1 + s2_2) * 0.5 / y_true.shape[0]
    result = - np.mean(-0.5 * np.log(2 * np.pi * (y_cov)) - 0.5 * (y_true - y_pred) ** 2 / (y_cov)) # von HLA
    return result


def rmse_regression(y_true, y_pred, variance):
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    avg_NLL = NLL_regression(y_true, y_pred, variance)
    return rmse, avg_NLL


def evaluate(y_true, y_pred, variance, type):
    if type == "binary_classification":     # binary classification
        return accuracy_binary(y_true, y_pred, variance)
    elif type.endswith("regression"):               # regression
        return rmse_regression(y_true, y_pred, variance)
    elif type.endswith("multi_classification"):               # regression
        return accuracy_multi(y_true, y_pred, variance)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, type):
    start = time.time()
    model.train(X_train, y_train)
    end = time.time()
    train_time = end - start
    """
    Evaluation for the methods with uncertainty output
    """
    y_pred, y_cov = model.predict(X_test)
    metric, avg_NLL = evaluate(y_test, y_pred, y_cov, type)
    if type == "binary_classification":     # binary classification
        acc,NLL= accuracy_binary(y_test, y_pred, y_cov)
    elif type.endswith("regression"):               # regression
        acc,NLL = rmse_regression(y_test, y_pred, y_cov)
    elif type.endswith("multi_classification"):               # regression
        acc,NLL = accuracy_multi(y_test,y_pred,y_cov)
    return train_time, metric, avg_NLL

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def mean_std(rmse, nll, time):
    rmse_mean = np.mean(rmse, axis=0)
    rmse_std = np.std(rmse, axis=0)
    nll_mean = np.mean(nll, axis=0)
    nll_std = np.std(nll, axis=0)
    time_mean = np.mean(time, axis=0)
    time_std = np.std(time, axis=0)
    return [rmse_mean, rmse_std, nll_mean, nll_std, time_mean, time_std]


