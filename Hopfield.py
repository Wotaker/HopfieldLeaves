import numpy as np
import os
import utilities


def get_x(folder_path):
    listdir = os.listdir(folder_path)
    x = np.ndarray((len(listdir), 50, 50))
    for i, image in enumerate(listdir):
        print(image)
        x[i, :, :] = utilities.load_image(folder_path + "/" + image)
    return utilities.flatten_input(x)


def wages(x):
    """
    :param x: numpy array shape = (examples_number,2500)
    :return:
    """
    x_diff = np.abs(x-1)
    n = 2500
    w = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                w[i, j] = np.sum(np.multiply(2 * x_diff[:, i] - 1, 2 * x_diff[:, j] - 1))
    print(w)
    return w


def activation(y, y_prev):
    print(y.shape)
    y_new = np.copy(y_prev)
    print(y.shape[0])
    for i in range(y.shape[0]):
        if y[i] > 0:
            y_new[i] = 1
        elif y[i] < 0:
            y_new[i] = 0
    print(np.sum(y_new))
    return y_new


def predict_w(w, x):
    """
    Run neural network to given example x
    :param w: wage matrix
    :param x: numpy array shape = (2500)
    :return:
    """
    y_prev = x.copy()
    y = y_prev.copy()
    for i in range(0, 2500):
        y[i] = np.sum(np.dot(w[i, :], x[i]))
    y = activation(y, y_prev)
    comparison = y == x
    equal_arrays = comparison.all()
    if equal_arrays:
        return y
    else:
        return predict_w(w, y)


class HopfieldNetwork:
    def __init__(self, path="wages.npy", save_path="wages.npy"):
        self.x_data = get_x("ready_leaves")
        if path is not None:
            self.wages = np.load(path)
            print(self.wages)
        else:
            self.wages = wages(self.x_data)
            np.save(save_path, self.wages)
            print(self.wages)

    def predict_image(self, image):
        """
        Method that takes array of shape (n,n) and returned array
        of the same shape which is hopfield network result.
        :image: array of shape (n,n)
        :return: network result
        """
        flatten = utilities.flatten_input(np.expand_dims(image, axis=0))[0]
        predicted = predict_w(self.wages, flatten)
        result = utilities.back_to_image(np.expand_dims(predicted, axis=0))[0]
        return result
