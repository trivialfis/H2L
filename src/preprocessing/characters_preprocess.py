import os
import random
from skimage import io, exposure
# import numpy as np
import cv2
from random import shuffle
from preprocessing.reform import randomReform  # , binarize
from evaluator import h2l_debug
from tqdm import tqdm

SOURCE = '../resource/pngs'
TRAINING = '../resource/training'
VALIDATION = '../resource/validation'
LIMIT = 10000

debugger = h2l_debug.h2l_debugger()


def load_images():
    bar = tqdm(total=36, unit='symbol')
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
        bar.update(1)
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
    bar = tqdm(total=len(all_images), unit='symbol')
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
        bar.update(1)
    return result


def save_images(all_images):

    def save(data, target):
        for value in data:
            symbol = value[0]
            images = value[1]
            path = os.path.join(target, symbol)
            os.mkdir(path)
            index = 0
            for image in images:
                filename = os.path.join(path, str(index) + '.png')
                index += 1
                io.imsave(fname=filename, arr=image)

    all_images = list(all_images.items())
    shuffle(all_images)
    training = all_images[:round(len(all_images)*0.9)]
    validation = all_images[round(len(all_images)*0.9):]
    save(training, TRAINING)
    save(validation, VALIDATION)


def start():
    print('Load')
    all_images = load_images()
    low_contrast = 0
    for k, v in all_images.items():
        for image in v:
            if exposure.is_low_contrast(image):
                low_contrast += 1
    debugger.display(low_contrast)
    print('Clean')
    all_images = clean(all_images)
    print('Generate')
    all_images = generate(all_images)
    low_contrast = 0
    for k, v in all_images.items():
        for image in v:
            if exposure.is_low_contrast(image):
                low_contrast += 1
    debugger.display(low_contrast)
    print('Save')
    save_images(all_images)
