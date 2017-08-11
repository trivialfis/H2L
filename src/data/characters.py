'''
File:          dataLoader.py
Author:        fis
Created:              2016
Last modified: 08 Feb 2017
'''
import pickle
from random import shuffle
import numpy as np
import os
from keras.utils import np_utils
from configuration import characterRecognizerConfig as config
from keras.utils import Sequence
from evaluator import h2l_debug
# from preprocessing import reform

debuger = h2l_debug.h2l_debugger()


class symbol_sequence(Sequence):
    def __init__(self):
        self.file_count = 0
        self.batch_count = 0
        self.__load_file__()

    def __len__(self):
        return config.SAMPLES_PER_EPOCH // config.BATCH_SIZE

    def __load_file__(self):
        filename = config.TRAIN_DATA + str(self.file_count) + '.pkl'
        with open(filename, 'rb') as f:
            characters = pickle.load(f)
            shuffle(characters)
            self.images, self.labels = zip(*characters)
            self.length = len(self.labels)
            self.images = np.asarray(self.images, dtype=np.float32)
            self.labels = np.asarray(self.labels, dtype=np.float32)
            self.file_count += 1

    def __getitem__(self, idx):
        if self.file_count > config.FILES_COUNT:
            self.file_count
        if self.batch_count > self.length // config.BATCH_SIZE:
            self.batch_count = 0
            self.__load_file__()

        start = round(config.BATCH_SIZE * self.batch_count)
        end = round(config.BATCH_SIZE * (self.batch_count+1))
        if start >= self.length:
            raise IndexError(
                'Start is greater than the dataset size')
        elif end > self.length:
            batchImages = self.images[start:, ...]
            batchLabels = self.labels[start:, ...]
        else:
            try:
                batchImages = self.images[start: end, ...]
                batchLabels = self.labels[start: end, ...]
            except TypeError:
                # The worker model of keras is not thread safe,
                # the np.asarray might be ignored.
                debuger.display('\ncharacters: Types:\n',
                                'images: ', type(self.images), '\n',
                                'labels: ', type(self.labels), '\n',
                                'start: ',  type(start), '\n',
                                'end: ',    type(end))
                os.abort()

        batchLabels = np_utils.to_categorical(batchLabels,
                                              config.CLASS_NUM)
        self.batch_count += 1
        batch = (batchImages, batchLabels)
        return batch

    def dataGenerator(self, idx):
        file_count = 0
        filename = config.TRAIN_DATA + str(file_count) + '.pkl'
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
                raise IndexError(
                    'Start is greater than the dataset size')
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
                        raise IndexError(
                            'Start is greater than the dataset size')
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


# def trainDataLoader():
#     loader = dataGenerator()
#     return loader


def validationDataLoader():
    with open(config.VALIDATION_DATA, 'rb') as f:
        validationData = pickle.load(f)
    images, labels = zip(*validationData)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    labels = np_utils.to_categorical(labels, config.CLASS_NUM)
    return (images, labels)
