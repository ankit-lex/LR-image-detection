import numpy as np


class Model:

    m = None
    W = None
    b = None
    X = None
    Y = None
    dim = None
    costs = []
    num_of_iterations = 1000
    learning_rate = 0.5

    def __init__(self, trainset_x, trainset_y, num_of_iterations=1000, learning_rate=0.5):
        self.X = trainset_x.T
        self.Y = trainset_y
        self.dim = self.X.shape[0]
        self.W, self.b = self.__initialize_weights_biases()
        self.num_of_iterations = num_of_iterations
        self.learning_rate = learning_rate
        self.m = trainset_x.shape[1]

    def __initialize_weights_biases(self):
        w = np.zeros((self.dim, 1))
        b = 0
        return w, b

    def printshapes(self):
        print('W :', self.W.shape)
        print('X :', self.X.shape)
        print('Y :', self.Y.shape)
        print(self.X.shape[1])

    def __sigmoid(self, z):
        a = 1/(1+np.exp(-z))
        return a

    def __cost(self, y_mat, a_mat):
        cost = (-1/self.m) * (np.sum((y_mat*np.log(a_mat).T) + ((1-y_mat)*(np.log(1-a_mat)).T)))
        return cost

    def propagate(self):

        A = self.__sigmoid(np.dot(self.W.T, self.X)+self.b)
        cost = -1/self.m*(np.sum(self.Y*np.log(A) + (1-self.Y)*np.log(1-A)))
        dw = (1/self.m)*np.dot(self.X, (A-self.Y).T)
        db = (1/self.m)*np.sum(A - self.Y)

        grads = {'dw': dw,
                 'db': db}

        cost = np.squeeze(cost)

        return grads, cost

    def optimize(self, print_cost=False):

        for i in range(self.num_of_iterations):

            grads, cost = self.propagate()
            self.costs.append(cost)

            dw = grads['dw']
            db = grads['db']

            self.W = self.W - (self.learning_rate*dw)
            self.b = self.b - (self.learning_rate*db)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return self.W, self.b

    def predict(self,X):
        print("test set shape : ", X.shape)
        m = X.shape[0]
        Y_Prediction = np.zeros((1, m))

        w = self.W.reshape(X.shape[1], 1)

        A = self.__sigmoid(np.dot(w.T, X.T)+self.b)

        for i in range(A.shape[1]):
            Y_Prediction[0][i] = 1 if A[0][i] > 0.5 else 0

        return Y_Prediction

