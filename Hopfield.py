import numpy as np


def printMatrix(M):
    def formatNum(n):
        if n >= 0: return " " + str(n)
        return "-" + str(abs(n))

    for w in range(len(M)):
        print("[ ", end="")
        for k in range(len(M[0])):
            print("{x}".format(x=formatNum(M[w][k])), end=" ")
        print("]")

def bin5(n):
    if n < 0 or n > 31: return None
    result = [0 for i in range(5)]
    for d in range(4, -1, -1):
        if n >= 2**d:
            result[4 - d] = 1
            n -= 2**d
    return result



def wage5(i, j, X):
    if i == j: return 0
    wageRes = 0
    for m in [3, 24]:
        wageRes += ((2 * X[m][i] - 1) * (2 * X[m][j] - 1))
    return wageRes

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


def main():
    X0 = [bin5(j) for j in range(32)]
    print("========\n")
    W = [[0 for i in range(5)] for j in range(5)]

    for i in range(5):
        for j in range(5):
            W[i][j] = wage5(i, j, X0)

    printMatrix(W)

    print("\n=== Stationary States ===")
    for i in range(31):
        result = iterate(X0[i], W, 0)
        print(str(X0[i]) + " ---> " + str(iterate(X0[i], W, 0)))


if __name__ == "__main__":
    main()
