'''
File:          dataLoader.py
Author:        fis
Created:              2016
Last modified: 08 Feb 2017
'''
import pickle
from random import shuffle
import numpy as np
from keras.utils import np_utils
from configuration import characterRecognizerConfig as config
from preprocessing import reform


def dataGenerator():
    while True:
        for i in range(config.FILES_COUNT):
            filename = config.TRAIN_DATA + str(i) + '.pkl'
            with open(filename, 'rb') as f:
                characters = pickle.load(f)
            shuffle(characters)
            images, labels = zip(*characters)
            length = len(labels)
            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            batchesCount = length // config.BATCH_SIZE
            for j in range(batchesCount):
                start = round(config.BATCH_SIZE*j)
                end = round(config.BATCH_SIZE*(j+1))
                if start >= length:
                    raise IndexError('Start is greater than the dataset size')
                elif end > length:
                    batchImages = images[start:, ...]
                    batchLabels = labels[start:, ...]
                else:
                    batchImages = images[start: end, ...]
                    batchLabels = labels[start: end, ...]
                batchLabels = np_utils.to_categorical(batchLabels,
                                                      config.CLASS_NUM)
                # reformedImages = []
                # for image in batchImages:
                #     reformed = reform.randomReform(
                #         image,
                #         binarizing=False)
                #     reformedImages.append(reformed)
                # batchImages = np.array(reformedImages, dtype=np.float32)
                batch = (batchImages, batchLabels)
                yield batch


def trainDataLoader():
    loader = dataGenerator()
    return loader


def validationDataLoader():
    with open(config.VALIDATION_DATA, 'rb') as f:
        validationData = pickle.load(f)
    images, labels = zip(*validationData)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    labels = np_utils.to_categorical(labels, config.CLASS_NUM)
    return (images, labels)
