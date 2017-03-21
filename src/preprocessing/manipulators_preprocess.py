'''
File:          manipulators_preprocess.py
Author:        fis
Created date:  Feb  5 2017
Last modified: Mar 14 2017
'''
from skimage import io, transform
from random import shuffle
from tqdm import tqdm
import numpy as np
import os
import pickle
from preprocessing import reform
from configuration import characterRecognizerConfig as config

IMAGES_PATH = '../resource/characters/'
TARGET_FILE = '../resource/characters_pkl/characters'
VALIDATION_FILE = '../resource/characters_pkl/validation.pkl'
TASK_NUM = config.FILES_COUNT
HEIGHT = 48
VALIDATION_SIZE = 300


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
        categoryPrint[i] = manipulators[i]
        category[manipulators[i]] = i
    f = open('../resource/character_map', 'w')
    f.write(str(categoryPrint))
    f.close()
    return category


def loadImagesInfo(path):
    print('Loading info')
    imagesPath = [os.path.join(dirpath, img)
                  for dirpath, dirnames, filenames in os.walk(path)
                  for img in filenames]
    category = mapping(path)
    # print(imagesPath[0])
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
    outputShape = (HEIGHT, HEIGHT)
    result = []
    for image, label in labeledImages:
        bar.update(1)
        try:
            resizedImage = transform.resize(image, outputShape)
            resizedImage = np.array(binarize(resizedImage), dtype=np.float32)
            resizedImage[0, :] = 0
            resizedImage[-1, :] = 0
            resizedImage[:, 0] = 0
            resizedImage[:, -1] = 0
            resizedImage = reform.removeEdge(resizedImage)
            resizedImage = reform.binarize(resizedImage, mode='greater')
            resizedImage = resizedImage.reshape(HEIGHT, HEIGHT, 1)
        except ValueError:
            print('Resize error, image shape: ', image.shape)
            continue
        result.append((resizedImage, label))
    return result


def save(labeledImages, targetFile):
    with open(targetFile, 'wb') as f:
        pickle.dump(labeledImages, f)


def start():
    imagesInfo = loadImagesInfo(IMAGES_PATH)
    shuffle(imagesInfo)
    bar = tqdm(total=2*len(imagesInfo), unit=' steps')
    validationInfo = imagesInfo[-VALIDATION_SIZE:]
    imagesInfo = imagesInfo[:-VALIDATION_SIZE]
    length = len(imagesInfo)
    tasks = [imagesInfo[(length)//TASK_NUM*i: length//TASK_NUM*(i+1)]
             for i in range(TASK_NUM)]
    for i in range(TASK_NUM):
        labeledImages = loadImages(tasks[i], bar)
        labeledImages = resize(labeledImages, bar)
        images, labels = zip(*labeledImages)
        save(labeledImages, TARGET_FILE+str(i)+'.pkl')
    print('Generating validation data')
    bar = tqdm(total=len(validationInfo), unit=' steps')
    labeledImages = loadImages(validationInfo, bar)
    labeledImages = resize(labeledImages, bar)
    images, labels = zip(*labeledImages)
    save(labeledImages, VALIDATION_FILE)
