'''
File:          character_recognizer_trainer.py
Author:        fis
Created:       19 Jan 2017
Last modified: 09 Feb 2017

Description:
A convolutional network trained to recognize characters
'''
from keras.layers import Dense, Dropout, Flatten, merge, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adadelta
from keras import models
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.utils.visualize_util import plot
from data.characters import trainDataLoader, validationDataLoader
import math
from configuration import characterRecognizerConfig as config


def branchModel():
    inputLayer = Input(shape=config.INPUT_SHAPE)

    minor = Convolution2D(
        16, 5, 5, border_mode='same',
        activation='relu',
        init='uniform'
    )(inputLayer)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Convolution2D(
        32, 3, 3, border_mode='same',
        activation='relu',
        init='uniform'
    )(minor)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Flatten()(minor)

    large = Convolution2D(
        16, 8, 8, border_mode='same',
        activation='relu',
    )(inputLayer)
    large = MaxPooling2D(pool_size=(8, 8))(large)
    large = Flatten()(large)

    merged = merge([minor, large], mode='concat')
    merged = Dense(1024,
                   W_regularizer=l2(0.01),
                   init='uniform',
                   activity_regularizer=activity_l2(0.01),
                   activation='relu')(merged)
    # merged = Dropout(0.25)(merged)
    merged = Dense(512, activation='relu', init='uniform')(merged)
    merged = Dropout(0.5)(merged)
    outputLayer = Dense(
        config.CLASS_NUM,
        activation='softmax',
        init='uniform',
        W_regularizer=l2(0.01),
        activity_regularizer=activity_l2(0.01)
    )(merged)
    model = Model(input=inputLayer, output=outputLayer)
    return model


def sequentialModel():
    model = Sequential()
    model.add(ZeroPadding2D(
        input_shape=config.INPUT_SHAPE,
        padding=(2, 2, 4, 4)
    ))
    model.add(Convolution2D(
        128, 5, 5, border_mode='same',
        activation='relu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(
        64, 3, 3, border_mode='valid'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.CLASS_NUM,
                    W_regularizer=l2(0.01),
                    activity_regularizer=activity_l2(0.01),
                    init='uniform'))
    model.add(Activation('softmax'))
    return model


class trainer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as a:
                self.model = models.model_from_json(a.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            print(config.NAME + ' initialized from files')
        else:
            # self.model = branchModel()
            self.model = sequentialModel()
            with open(config.ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                # plot(self.model, to_file='model.png')
                print(config.NAME + ' saved to file')

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adadelta(),
                           metrics=['accuracy'])
        print(config.NAME + ' model compiled')

    def train(self):
        print(config.NAME + ' configuring validation data')
        validationImages, validationLabels = validationDataLoader()

        def stepDecay(epoch):
            initialLearningRate = config.INIT_LEARNING_RATE
            drop = 0.96
            epochsDrop = 1.0
            lrate = initialLearningRate * math.pow(
                drop,
                math.floor((1+epoch)/epochsDrop))
            print('Learning rate: ', lrate)
            return lrate

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
        print('Start training')
        self.model.fit_generator(
            trainDataLoader(),
            samples_per_epoch=config.SAMPLES_PER_EPOCH,
            nb_epoch=config.EPOCH,
            verbose=True,
            callbacks=callbacks,
            validation_data=(validationImages, validationLabels),
            max_q_size=80,
            nb_worker=2,
            pickle_safe=True,
            initial_epoch=0
        )
