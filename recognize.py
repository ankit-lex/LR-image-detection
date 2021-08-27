import sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import codecs, json
from PIL import Image


def sigmoid(z):
    return 1/(1+np.exp(-z))


def predict():
    obj_text = codecs.open('parameters.json', 'r', encoding='utf-8').read()
    jsonfile = json.loads(obj_text)
    b = float(jsonfile['b'])
    w = np.array(jsonfile['w'])

    # print(w.shape)

    img = load_image()

    print("image shape : ", img.shape)
    Y_Prediction = 0

    w = w.reshape(img.shape[0], 1)
    print("w shape : ", w.shape)
    A = sigmoid(np.dot(w.T, img) + b)
    print(A)
    Y_Prediction = 1 if A > 0.6 else 0

    print(Y_Prediction)


def load_image():
    fpath = sys.argv[1]
    image = np.array(Image.open(fpath).resize((64, 64)))
    plt.imshow(image)
    image = image / 255.
    my_image = image.reshape((1, 64 * 64 * 3)).T
    return my_image

predict()