from PIL import Image
from scipy import ndimage
"""

"""

class Process:

    dataset = None
    m = None
    n = None
    dimensions = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.dimensions = self.dataset.shape

    def getdetails(self, p=True):
        self.m = self.dataset.shape[0]
        self.n = self.dataset[0].shape
        if p:
            print("Number of examples : ", self.m)
            print("Number of inputs : ", self.n)
        return self.m, self.n

    def flattenimageset(self):
        self.dataset = self.dataset.reshape(self.m, -1)

    def standardizeimages(self):
        self.dataset = self.dataset / 255

    def undoflattening(self):
        self.dataset = self.dataset.reshape(self.dimensions)

    def unstandardizeimages(self):
        self.dataset = self.dataset * 255






