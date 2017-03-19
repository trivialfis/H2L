'''
File: segmenter_trainer.py
Author:        fis
Created:       Jan 25 2017
Last modified: Feb 21 2017

Description:
A neural network trained to do characters segmentation
'''
from keras import models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
# from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from data.baseWords import wordsSegmentationData
from configuration import baseSegmenterConfig as config
import math


class trainer(object):
    def __init__(self):
        if config.baseModelExists():
            with open(config.BASE_ARCHITECTURE_FILE, 'r') as architecture:
                self.model = models.model_from_json(architecture.read())
            self.model.load_weights(config.BASE_WEIGHTS_FILE)
            print(config.BASE_NAME + ' initialized from existing files')
        else:
            self.model = models.Sequential()
            self.model.add(ZeroPadding2D(
                input_shape=config.INPUT_SHAPE,
                padding=(2, 2, 4, 4)
            ))
            self.model.add(Convolution2D(
                16, 5, 5, border_mode='same',
                activation='relu',
                init='uniform'
            ))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Convolution2D(
                32, 5, 5, border_mode='same',
                activation='relu',
                init='uniform'
            ))
            self.model.add(Flatten())
            '''
            self.model.add(Dense(
                128,
                activation='relu',
                W_regularizer=l2(0.01),
                activity_regularizer=activity_l2(0.01),
                init='uniform'
            ))
            '''
            self.model.add(Dropout(0.5))
            self.model.add(Dense(
                32,
                activation='sigmoid',
                # W_regularizer=l2(0.01),
                # activity_regularizer=activity_l2(0.01),
                init='uniform'
            ))
            self.model.add(Dense(
                1,
                activation='relu',
                W_regularizer=l2(0.01),
                activity_regularizer=activity_l2(0.01),
                init='uniform'
            ))
            with open(config.BASE_ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                print(config.BASE_NAME, ' architecture saved to file')
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        print(config.BASE_NAME + ' model compiled')

    def train(self):
        '''Train the network using 128 batch size'''
        print(config.BASE_NAME + ' configurating data')
        images, labels = wordsSegmentationData()
        dataCount = len(labels)
        print(images.shape, labels.shape)
        # labels = np_utils.to_categorical(labels, 2)
        trainImages, trainLabels = (
            images[:round(dataCount*0.8)],
            labels[:round(dataCount*0.8)]
        )
        validationImages, validationLabels = (
            images[round(dataCount*0.8):],
            labels[round(dataCount*0.8):]
        )
        print(config.BASE_NAME + ' start training')

        def step_decay(epoch):
            initialLearningRate = 0.01
            drop = 0.85
            epochsDrop = 10.0
            learningRate = initialLearningRate * math.pow(
                drop,
                math.floor((1+epoch)/epochsDrop))
            print('Learning rate: ', learningRate)
            return learningRate

        callbacks = [
            LearningRateScheduler(step_decay),
            # EarlyStopping(
            # monitor='val_acc',
            # verbose=True,
            # patience=5
            # ),
            ModelCheckpoint(
                config.BASE_WEIGHTS_FILE,
                monitor='val_acc',
                verbose=True,
                save_best_only=True,
                save_weights_only=True
            )]
        self.model.fit(
            trainImages, trainLabels,
            batch_size=config.BATCH_SIZE,
            nb_epoch=config.EPOCH,
            verbose=1,
            shuffle=True,
            validation_data=(validationImages, validationLabels),
            callbacks=callbacks)
