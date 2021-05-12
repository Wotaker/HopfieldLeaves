import numpy as np
import os
import utilities


def get_x(folder_path):
    listdir = os.listdir(folder_path)
    x = np.array((len(listdir),50,50))
    for i, image in enumerate(listdir):
        x[i, :, :] = utilities.load_image(folder_path + "/" + listdir)
    return utilities


def bin5(n):
    if n < 0 or n > 31: return None
    result = [0 for i in range(5)]
    for d in range(4, -1, -1):
        if n >= 2**d:
            result[4 - d] = 1
            n -= 2**d
    return result


def wages(x):
    n = 2500
    w = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                w[i, j] = np.sum(np.multiply(2 * x[:, i] - 1, 2 * x[:, j] - 1))
    return w

def iterate(xi, W, i):
    # x0 to wektor wejściowy do neuronu, W to macierz wag

    def activ(si, yi):
        yi1 = [0 for k in range(len(yi))]
        for k in range(len(si)):
            if si[k] > 0: yi1[k] = 1
            elif si[k] == 0: yi1[k] = yi[k]
            else: yi1[k] = 0
        return yi1

    xi1 = activ(np.matmul(W, xi), xi)
    if xi1 == xi: return xi1, i + 1
    return iterate(xi1, W, i + 1)


def activation(y, y_prev):
    y_new = np.copy(y_prev)
    for i in range(y.shape[0]):
        if y[i] > 0:
            y_new[i] = 1
        elif y[i] < 0:
            y_new[i] = 0
    return y_new


def predict_w(w, x):
    """
    Run neural network to given example x
    :param w: wage matrix
    :param x:
    :return:
    """
    y_prev = x.copy()
    y = np.matmul(x, w)
    y = activation(y, y_prev)
    comparison = y == x
    equal_arrays = comparison.all()
    if equal_arrays:
        return y
    else:
        return predict_w(w, y)


def main():
    X0 = [bin5(j) for j in range(32)]
    print("========\n")

    w = wages(X0)

    print(w)

    print("\n=== Stationary States ===")
    for i in range(31):
        result = iterate(X0[i], w, 0)
        print(str(X0[i]) + " ---> " + str(iterate(X0[i], w, 0)))


if __name__ == "__main__":
    main()
