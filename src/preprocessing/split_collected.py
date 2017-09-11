import cv2
import os
from normalization import image_utils

SOURCE = '../resource/collected'
TARGET = '../resource/splited'


def load_images():
    symbols = os.listdir(SOURCE)
    images = {}
    edge = 25
    for sym in symbols:
        image = cv2.imread(os.path.join(SOURCE, sym))
        image = image_utils.binarize3d(image)
        image[:edge, :] = 255
        image[-edge:, :] = 255
        image[:, :edge] = 255
        image[:, -edge:] = 255
        # image = cv2.threshold(
        #     image, 0, 255,
        #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # )[1]
        images[sym] = image
    return images


def split(images):
    characters = {}
    for sym, image in images.items():
        height = image.shape[0] // 12
        width = image.shape[1] // 8
        splited = []
        cur_row = 0
        for i in range(12):
            line = image[cur_row: cur_row + height, :]
            cur_col = 0
            for j in range(8):
                symbol = line[:, cur_col: cur_col + width]
                splited.append(symbol)
                cur_col += width
            cur_row += height
        characters[sym] = splited
    return characters


def save_images(images):
    for sym, ims in images.items():
        os.mkdir(os.path.join(TARGET, sym[:-4]))
        count = 0
        for im in ims:
            filename = os.path.join(TARGET, sym[:-4], str(count) + '.png')
            count += 1
            cv2.imwrite(filename=filename, img=im)


def start():
    images = load_images()
    images = split(images)
    save_images(images)
