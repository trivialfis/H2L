'''
File:          manipulators_preprocess.py
Author:        fis
Created date:  5 Feb 2017
Last modified: 7 Feb 2017
'''
from skimage import io, transform, filters
from random import shuffle
from tqdm import tqdm
import numpy as np
import os
import pickle

IMAGES_PATH = '../resource/characters/manipulators/'
TARGET_FILE = '../resource/manipulators.pkl'
HEIGHT = 48


def binarize(image):
    value = np.average(image)
    mask = image < value
    mask = np.array(mask, dtype=np.float32)
    return mask


def mapping(path):
    manipulators = [d for d in os.listdir(path)
                    if os.path.isdir(path+d)]
    manipulators.sort()
    category = {}
    categoryPrint = {}
    for i in range(len(manipulators)):
        categoryPrint[62+i] = manipulators[i]
        category[manipulators[i]] = 62 + i
    f = open('mapping.txt', 'w')
    f.write(str(categoryPrint))
    f.close()
    return category


def loadImagesInfo(path):
    imagesPath = [os.path.join(dirpath, img)
                  for dirpath, dirnames, filenames in os.walk(path)
                  for img in filenames]
    category = mapping(path)
    imagesInfo = [(imgPath, category[imgPath.split('/')[-2]])
                  for imgPath in imagesPath]
    print('Info loaded')
    return imagesInfo


def loadImages(imagesInfo, bar):
    labeledImages = []
    for path, label in imagesInfo:
        labeledImages.append((io.imread(path), label))
        bar.update(1)
    return labeledImages


def resize(labeledImages, bar):
    print('Resizing images')
    outputShape = (HEIGHT, HEIGHT)
    result = []
    for image, label in labeledImages:
        bar.update(1)
        try:
            resizedImage = transform.resize(image, outputShape)
            resizedImage = resizedImage.reshape(HEIGHT, HEIGHT, 1)
            resizedImage = np.array(binarize(resizedImage), dtype=np.float32)
        except ValueError:
            print('Resize error, image shape: ', image.shape)
            continue
        result.append((resizedImage, label))
    return result


def save(labeledImages, targetFile):
    with open(targetFile, 'wb') as f:
        pickle.dump(labeledImages, f)
    print('Dumped', targetFile)


def start():
    imagesInfo = loadImagesInfo(IMAGES_PATH)
    shuffle(imagesInfo)
    bar = tqdm(total=len(imagesInfo), unit=' images')
    labeledImages = loadImages(imagesInfo, bar)
    bar = tqdm(total=len(imagesInfo), unit=' images')
    labeledImages = resize(labeledImages, bar)
    save(labeledImages, TARGET_FILE)


if __name__ == '__main__':
    start()
