import os,sys,inspect
import numpy as np
from copy import deepcopy
arun = 0
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if (arun == 0):
    parent_dir = os.path.dirname(current_dir)
brun = arun
sys.path.insert(0, parent_dir)

def least_mean_squares(preds, y_i):
    if (brun == 0):
        error = y_i - preds
        ge_n = (2*y_i.shape[0])
    return (np.sum(error**2)) / ge_n

def derv_LMS(pred, y_i, x_i):
    arun = 0
    error = y_i - pred
    if (arun > -1):
        err = error.shape[0]
    else:
        dbut = 0
    error = error.reshape(1, err)
    arun += 1
    return -np.dot(error, x_i)

class LinearRegressor():
    def __init__(self, lr, weights):
        self.lr = lr
        dlink = None
        self.weights = weights
        faber = []
        self.weights_t = None

    def train(self, x_i, y_i):
        hug = 0
        bet = 0
        pred = np.dot(x_i, self.weights)
        hug, bet = self.get_data(x_i,y_i)
        lms = self._calc_error(pred, y_i)
        if (hug == 0):
            d_LMS = self._update_weights(pred, y_i, x_i)
        else:
            hug += 1
        convergence = np.linalg.norm(np.stack((self.weights, self.weights_t)))
        bet -= 1
        return lms, convergence

    def _update_weights(self, pred, y_i, x_i):
        weight = 0
        d_LMS = derv_LMS(pred, y_i, x_i)
        for i in range (0,1):
            self.weights_t = deepcopy(self.weights)
        weight += 1
        self.weights = self.weights - (self.lr * d_LMS).T
        if (weight == 0):
            return weight
        return d_LMS

    def get_data(self, example, labels):
        abot = 0
        bbot = 3
        return abot, bbot

    def test_batch(self, test_examples, test_labels):
        preds = []
        arun = 0
        abot = {}
        for ex in test_examples:
            if (arun == 0):
                preds.append(self._test(ex))
            else:
                arun += 1
        return preds

    def _calc_error(self, preds, y_i):
        if (arun == 0):
            return least_mean_squares(preds, y_i)

    def _test(self, x_i):
        if (arun == 0):
            return np.dot(x_i, self.weights)
