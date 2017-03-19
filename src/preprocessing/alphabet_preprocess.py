'''
File:          alphabet_preprocess.py
Author:        fis
Created date:  Feb  4 2017
Last modified: Feb 26 2017
'''
from tqdm import tqdm
from random import shuffle
import numpy as np
import pickle
import os
from skimage import io, filters, transform

IMAGE_PATH = '../resource/characters/alphabet/'
TARGET_PATH = '../resource/alphabet'
TASKS_COUNT = 12
HEIGHT = 48


def binarize(image):
    value = filters.threshold_isodata(image)
    mask = image < value
    mask = np.array(mask, dtype=np.float32)
    return mask


def removeEdge(labeledImages, bar):
    '''Remove the white edges of the characters'''
    def detectRow(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.max(image[index, :])
            if filled == 0.0:
                count += 1
            else:
                break
        return count

    def detectCol(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.max(image[:, index])
            if filled == 0.0:
                count += 1
            else:
                break
        return count

    def subprocess(image):
        height, width = image.shape
        image = binarize(image)
        up = detectRow(image, height)
        left = detectCol(image, width)
        down = detectRow(image, height, True)
        right = detectCol(image, width, True)
        count = min([up, left, down, right])
        if count > 3:
            count = count - 3
        result = np.array(image[count: -count, count: -count],
                          dtype=np.float32)
        return result

    result = []
    for image, label in labeledImages:
        result.append((subprocess(image), label))
        bar.update(1)
    return result


def loadImagesInfo(imagesPath):
    imagesFilename = [img for root, subdirs, files
                      in os.walk(imagesPath)
                      for img in files]
    imagesInfo = []
    for img in imagesFilename:
        try:
            label = img.split('_')[1]
        except IndexError:
            print(img)
        path = IMAGE_PATH + label + '/' + img
        intLabel = int(label, 16)
        if intLabel in range(48, 58):
            intLabel = intLabel - 48
        elif intLabel in range(65, 91):
            intLabel = intLabel - 55
        else:
            intLabel = intLabel - 61
        imagesInfo.append({'path': path,
                           'name': img,
                           'label': intLabel})
    return imagesInfo


def resize(labeledImages, bar):
    outputShape = (HEIGHT, HEIGHT)
    result = []
    for image, label in labeledImages:
        bar.update(1)
        try:
            resizedImage = transform.resize(image, outputShape)
            resizedImage = resizedImage.reshape(HEIGHT, HEIGHT, 1)
            resizedImage = np.array(resizedImage, dtype=np.float32)
        except ValueError:
            print('Resize error, image shape: ', image.shape)
            continue
        result.append((resizedImage, label))
    return result


def loadImages(imagesInfo, bar):
    labeledImages = []
    for imginfo in imagesInfo:
        img = io.imread(imginfo['path'])
        labeledImages.append((img, imginfo['label']))
        bar.update(1)
    return labeledImages


def save(labeledImages, targetFile):
    with open(targetFile, 'wb') as target:
        pickle.dump(labeledImages, target)


def start():
    imagesInfo = loadImagesInfo(IMAGE_PATH)
    length = len(imagesInfo)
    print(length, ' images in total')
    shuffle(imagesInfo)
    tasks = [imagesInfo[length//TASKS_COUNT*i: length//TASKS_COUNT*(i+1)]
             for i in range(TASKS_COUNT)]
    bar = tqdm(total=length*3, unit=' steps')
    for t in tasks:
        images = loadImages(t, bar)
        images = removeEdge(images, bar)
        images = resize(images, bar)
        save(images, TARGET_PATH + str(tasks.index(t)) + '.pkl')


if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        print('Exit')
