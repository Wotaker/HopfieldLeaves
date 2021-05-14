import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


def resize(shape, source_path, destination_path):
    i = 0
    for image_path in os.listdir(source_path):
        image = Image.open(source_path + "/" + image_path)
        image = image.convert('RGB')
        image = image.resize(shape)
        image.save(destination_path + "/" + str(i) + ".jpg")
        i = i + 1


def generate_random_images(n, shape):
    """Generates n random images of shape dimensions of values 0 or 1"""
    return np.random.randint(0, 2, (n, shape[0], shape[1]))


def change_image(path, changes):
    """Changes image get from path, in random changes location change value between 0 and 1"""
    np_image = load_image(path)

    random_positions = np.random.randint(0, 50, (changes, 2))
    for x, y in random_positions:
        np_image[x, y] = (np_image[x, y]+1) % 2
    return np_image


def load_image(path, limit=180):
    """
    :param limit: below that value map to 0, else map to 1
    :param path: path to file
    :return: load a file with 0,1 values
    """
    image = Image.open(path)
    image = ImageOps.grayscale(image)
    np_image = np.array(image)
    # np_image = np_image // 64

    cut = lambda x: x > limit
    vFun = np.vectorize(cut)


    return vFun(np_image)


def flatten_input(arr):
    """
    Flatten array.
    :param arr: array of shape = (n, height, width)
    :return: array of shape = (n, height*width)
    """
    shape = arr.shape
    return np.reshape(arr, newshape=(shape[0], shape[1]*shape[2]), order='C')


def back_to_image(arr):
    """
    Reshape array back to 3D form
    :param arr: array of shape = (n, x*x)
    :return: array of shape = (n, x, x)
    """
    shape = arr.shape
    sqrt = np.sqrt(shape[1]).astype(np.int)
    return np.reshape(arr, newshape=(shape[0], sqrt, sqrt), order='C')


if __name__ == "__main__":
    # resize((50, 50), "leaves", "ready_leaves")
    # random = generate_random_images(1, (50, 50))
    # plt.imshow(random[0])
    # plt.show()

    plt.imshow(load_image("ready_leaves/10.jpg"))
    plt.show()

    changed_image = change_image("ready_leaves/10.jpg", 250)
    plt.imshow(changed_image)
    plt.show()
    random_trans = back_to_image(flatten_input(random))
    plt.imshow(random_trans[1])
    plt.show()
