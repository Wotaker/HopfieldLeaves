import os
from PIL import Image

def resize(shape, source_path, destination_path):
    i = 0
    for image_path in os.listdir(source_path):
        image = Image.open(source_path + "/" + image_path)
        image = image.convert('RGB')
        image.resize(shape)
        image.save(destination_path + "/" + str(i) + ".jpg")
        i = i + 1


if __name__ == "__main__":
    resize((50, 50), "leaves", "ready_leaves")
