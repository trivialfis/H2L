#!/usr/bin/env python3
#
# Copyright © 2017, 2018 Fis Trivial <ybbs.daans@hotmail.com>
#
# This file is part of H2L.
#
# H2L is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# H2L is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with H2L.  If not, see <http://www.gnu.org/licenses/>.
#

from keras.layers import Dense, Dropout, Flatten, merge, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras import models
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model

from data.characters import train_flow, validation_flow
from configuration import characterRecognizerConfig as config

from evaluator import h2l_debug

import math

debugger = h2l_debug.h2l_debugger()


def branchModel():
    inputLayer = Input(shape=config.INPUT_SHAPE)

    minor = Conv2D(
        16, 5, 5, padding='same',
        activation='relu',
        init='uniform'
    )(inputLayer)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Conv2D(
        32, 3, 3, padding='same',
        activation='relu',
        init='uniform'
    )(minor)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Flatten()(minor)

    large = Conv2D(
        16, 8, 8, padding='same',
        activation='relu',
    )(inputLayer)
    large = MaxPooling2D(pool_size=(8, 8))(large)
    large = Flatten()(large)

    merged = merge([minor, large], mode='concat')
    merged = Dense(1024,
                   W_regularizer=l2(0.01),
                   init='uniform',
                   activity_regularizer=l2(0.01),
                   activation='relu')(merged)
    # merged = Dropout(0.25)(merged)
    merged = Dense(512, activation='relu', init='uniform')(merged)
    merged = Dropout(0.5)(merged)
    outputLayer = Dense(
        config.CLASS_NUM,
        activation='softmax',
        init='uniform',
        kernel_regularizer=l2(0.01),
        activity_regularizer=l2(0.01)
    )(merged)
    model = Model(input=inputLayer, output=outputLayer)
    return model


def sequentialModel():
    model = Sequential()
    model.add(ZeroPadding2D(
        input_shape=config.INPUT_SHAPE,
        padding=((4, 4), (4, 4))
    ))
    model.add(Conv2D(
        filters=64, kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(Conv2D(
        filters=64, kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(
        filters=32, kernel_size=(2, 2),
        padding='valid',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.CLASS_NUM,
                    kernel_regularizer=l2(0.01),
                    activity_regularizer=l2(0.01),
                    kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    return model


class trainer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as a:
                self.model = models.model_from_json(a.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            debugger.display(config.NAME + ' initialized from file.')
        else:
            # self.model = branchModel()
            self.model = sequentialModel()
            with open(config.ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                plot_model(self.model, to_file=config.VISUAL_FILE,
                           show_shapes=True, show_layer_names=True)
                debugger.display(config.NAME + ' saved to file.')

        self.train_flow = train_flow()
        mapping = self.train_flow.class_indices
        mapping = dict((v, k) for k, v in mapping.items())
        with open(config.CHARACTER_MAP, 'w') as f:
            f.write(str(mapping))
        samples_per_epoch = self.train_flow.samples
        self.steps_per_epoch = samples_per_epoch // config.BATCH_SIZE

        self.validation_flow = validation_flow()
        validation_samples = self.validation_flow.samples
        batch_size_validation = config.VALIDATION_BATCH_SIZE
        self.validation_steps = validation_samples // batch_size_validation

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adadelta(),
                           metrics=['accuracy'])
        debugger.display(config.NAME + ' model compiled.')

    def train(self):

        def stepDecay(epoch):
            initialLearningRate = config.INIT_LEARNING_RATE
            drop = 0.96
            epochsDrop = 1.0
            lrate = initialLearningRate * math.pow(
                drop,
                math.floor((1+epoch)/epochsDrop))
            debugger.display('Learning rate: ', str(lrate))
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

        debugger.display('Start training.')
        self.model.fit_generator(
            generator=self.train_flow,
            # steps_per_epoch=config.SAMPLES_PER_EPOCH // config.BATCH_SIZE,
            steps_per_epoch=self.steps_per_epoch,
            epochs=config.EPOCH,
            verbose=1,
            callbacks=callbacks,
            validation_data=self.validation_flow,
            validation_steps=self.validation_steps,
            max_queue_size=10,
            workers=2,          # Not thread safe
            # multiprocessing may disable the gpu
            # use_multiprocessing=True,
            initial_epoch=0
        )
