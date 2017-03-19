'''
File:          wordSegmenterTrainer.py
Author:        fis
Crated:        27 Jan 2017
Last modified:  7 Feb 2017
'''
from keras import models
from keras.layers import Dense, Flatten, Dropout
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
from data.lines import segmentationData
from configuration import wordSegmenterConfig as config


class trainer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as architecture:
                self.model = models.model_from_json(architecture.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            print(config.NAME + ' initialized from files')
        else:
            self.model = models.Sequential()
            self.model.add(ZeroPadding2D(
                input_shape=config.INPUT_SHAPE,
                padding=(2, 2, 4, 4)
            ))
            self.model.add(Conv2D(
                16, 5, 5, border_mode='same',
                activation='relu',
                init='uniform'
            ))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(
                32, 3, 3, border_mode='same',
                activation='relu',
                init='uniform'
            ))
            self.model.add(Flatten())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(
                32,
                activation='sigmoid',
                W_regularizer=l2(0.01),
                activity_regularizer=activity_l2(0.01),
                init='uniform'
            ))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(
                1,
                activation='relu',
                init='uniform',
            ))
            with open(config.ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                print(config.NAME + ' architecture saved to file')
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        print(config.NAME + ' model compiled')

    def train(self):
        print(config.NAME + ' configurating data')
        images, labels = segmentationData()
        dataCount = len(labels)
        print('images shape: ', images.shape, ' labels shape: ', labels.shape)
        dataSeparation = round(dataCount*0.8)
        trainImages, trainLabels = (
            images[:dataSeparation],
            labels[:dataSeparation],
        )
        validationImages, validationLabels = (
            images[dataSeparation:],
            labels[dataSeparation:]
        )

        def stepDecay(epoch):
            initialLearningRate = 0.01
            drop = 0.85
            epochsDrop = 4.0
            learningRate = initialLearningRate * math.pow(
                drop,
                math.floor((1+epoch)/epochsDrop))
            print(learningRate)
            return learningRate

        print(config.NAME + ' start training')
        callbacks = [
            LearningRateScheduler(stepDecay),
            ModelCheckpoint(
                config.WEIGHTS_FILE,
                monitor='val_acc',
                verbose=True,
                save_best_only=True,
                save_weights_only=True
            )
        ]
        self.model.fit(
            trainImages, trainLabels,
            batch_size=config.BATCH_SIZE,
            nb_epoch=config.EPOCH,
            verbose=True,
            validation_data=(validationImages, validationLabels),
            callbacks=callbacks
        )
