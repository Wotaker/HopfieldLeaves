import os
from PIL import Image
import numpy as np


def resize(shape, source_path, destination_path):
    i = 0
    for image_path in os.listdir(source_path):
        image = Image.open(source_path + "/" + image_path)
        image = image.convert('RGB')
        image.resize(shape)
        image.save(destination_path + "/" + str(i) + ".jpg")
        i = i + 1


def generate_random_images(n, shape):
    """Generates n random images of shape dimensions of values 0 or 1"""
    return np.random.randint(0, 2, (n, shape[0], shape[1]))


def change_image(path, changes):
    """Changes image get from path, in random changes location change value between 0 and 1"""
    image = Image.open(path)
    image = image.convert('RGB')
    np_image = np.array(image)
    random_positions = np.random.randint(0, 50, (changes, 2))
    for x, y in random_positions:
        print(x, y)
        np_image[x, y] = (np_image[x, y]+1) % 2
    return np_image


if __name__ == "__main__":
    resize((50, 50), "leaves", "ready_leaves")
    random = generate_random_images(50, (50, 50))
    # print(random)
    # print(random.shape)
    changed_image = change_image("ready_leaves/1.jpg", 50)
    # print(changed_image)
