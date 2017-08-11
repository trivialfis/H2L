import os
from skimage import io
import random
import cv2
from random import shuffle
from reform import randomReform  # , binarize


SOURCE = './pngs'
TARGET = './generated'
LIMIT = 10000


def load_images():
    symbols = os.listdir(SOURCE)
    all_images = {}
    for sym in symbols:
        path = os.path.join(SOURCE, sym)
        images_name = os.listdir(path)
        shuffle(images_name)
        images_path = [os.path.join(path, img) for img in images_name]
        images = [io.imread(img) for img in images_path]
        images = [
            cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            for img in images]
        all_images[sym] = images
    return all_images


def clean(all_images):
    result = {}
    for k, v in all_images.items():
        images = v
        if len(v) > LIMIT:
            images = v[LIMIT:]
        result[k] = images
    return result


def generate(all_images):
    result = {}
    for k, v in all_images.items():
        length = len(v)
        ori_length = length
        result[k] = v
        while length < 2*LIMIT:
            index = random.randint(0, ori_length)
            result[k].append(randomReform(v[index],
                                          binarizing=False))
            length += 1
    return result


def save_images(all_images):
    for k, v in all_images.items():
        path = os.path.join(TARGET, k)
        os.mkdir(path)
        index = 0
        for image in v:
            filename = os.path.join(path, str(index) + '.png')
            index += 1
            io.imsave(fname=filename, arr=image)


if __name__ == '__main__':
    print('Load')
    all_images = load_images()
    print('Clean')
    all_images = clean(all_images)
    print('Generate')
    all_images = generate(all_images)
    print('Save')
    save_images(all_images)
