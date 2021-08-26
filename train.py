import numpy as np
import matplotlib.pyplot as plt
import codecs, json

from preprocessing import Process
from lr_utils import load_dataset
from lrmodel import Model

def train_model():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset('datasets/train_catvnoncat.h5',
                                                                                       'datasets/test_catvnoncat.h5')

    trainset = Process(train_set_x_orig)
    trainset.getdetails()
    trainset.flattenimageset()
    trainset.standardizeimages()

    testset = Process(test_set_x_orig)
    testset.getdetails()
    testset.flattenimageset()
    testset.standardizeimages()

    mod = Model(trainset.dataset, train_set_y, num_of_iterations=1000, learning_rate=0.5)
    mod.printshapes()
    W,b = mod.optimize(print_cost=True)
    W = W.tolist()
    params = {
        'w': W,
        'b': b
    }

    Y_prediction_test = mod.predict(testset.dataset)
    Y_prediction_train = mod.predict(trainset.dataset)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))


    file_path = "parameters.json"
    json.dump(params, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)

train_model()