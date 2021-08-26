from preprocessing import Process
from lr_utils import load_dataset
from lrmodel import Model

#loadcheck
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset('datasets/train_catvnoncat.h5',
                                                                                   'datasets/test_catvnoncat.h5')

#dimcheck
#print(train_set_x_orig[0].shape)
assert train_set_x_orig.shape == (209, 64, 64, 3)

#preproceesscheck
trainset = Process(train_set_x_orig)
trainset.getdetails()
trainset.flattenimageset()
trainset.standardizeimages()
#trainset.getdetails()
#print(trainset.dataset)

#model check
mod = Model(trainset.dataset, train_set_y, num_of_iterations=1000, learning_rate=0.5)
mod.printshapes()
mod.optimize(print_cost=True)
