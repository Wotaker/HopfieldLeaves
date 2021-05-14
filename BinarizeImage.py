import cv2
import os
from utilities import change_image
from matplotlib import image

def convert2binary(path, width, height, write=None, show=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error, unable to load your image!")
        return

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    th, img = cv2.threshold(img, 75, 255, cv2.THRESH_OTSU)

    if write:
        # writes converted image
        cv2.imwrite(write, img)

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

def makeDataSet(sourceFolderPath):
    """

    :param sourceFolderPath: Path to the folder with images,
    which are to be converted into dataset
    :return: ---
    """

    # Create directory tree
    if not os.path.exists("DataSet-Kwiatki"):
        os.mkdir("DataSet-Kwiatki")
        os.mkdir("DataSet-Kwiatki/wzorcowe")
        os.mkdir("DataSet-Kwiatki/testowe")
    else:
        print("The directory already exists")

    i = 0
    # Make pattern images
    for img_path in os.listdir(sourceFolderPath):
        new_path = "DataSet-Kwiatki/wzorcowe/" + img_path
        new_path = new_path[:(len(new_path) - 4)] + ".png"
        convert2binary("ready_leaves/" + img_path, 50, 50, write=new_path)

    i = 0
    # Randomize a little bit
    for img_path in os.listdir("DataSet-Kwiatki/wzorcowe"):
        for n in range(0, 420, 40):
            new_path = "DataSet-Kwiatki/testowe/" + img_path
            new_path = new_path[:(len(new_path) - 4)] + "_c" + str(n) + ".png"
            np_img = change_image("DataSet-Kwiatki/wzorcowe/" + img_path, n)
            image.imsave(new_path, np_img)



def main():
    makeDataSet("ready_leaves")

if __name__ == "__main__":
    main()
