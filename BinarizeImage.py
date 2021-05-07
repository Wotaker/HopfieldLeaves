import cv2


def convert2binary(path, width, height, write=False, show=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error, unable to load your image!")
        return

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    th, img = cv2.threshold(img, 32, 255, cv2.THRESH_OTSU)

    if write:
        # writes converted image
        cv2.imwrite(path[:(len(path) - 4)] + "_b.png", img)

    if show:
        # Show changes
        cv2.imshow("Resized image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return matrix
    return [[img[h][w] for w in range(width)] for h in range(height)]


def linearize(matrix):
    result = []
    for row in matrix:
        result += row
    return result


def printMatrix(matrix):
    bit = lambda x: x // 255
    height = len(matrix)
    width = len(matrix[0])
    for h in range(height):
        print("[", end=" ")
        for w in range(width):
            print(bit(matrix[h][w]), end=" ")
        print("]")


def main():
    bitmap = convert2binary("fallus.png", 50, 50)
    printMatrix(bitmap)

if __name__ == "__main__":
    main()
