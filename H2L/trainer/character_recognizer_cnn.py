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

from keras.optimizers import Adadelta
from keras import models

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model

from ..configuration import characterRecognizerConfig as config
from ..evaluator import h2l_debug

import math

debugger = h2l_debug.h2l_debugger()


class trainer(object):

    def __init__(self, data_flow):

        self.train_flow = data_flow['train']
        mapping = self.train_flow.class_indices
        mapping = dict((v, k) for k, v in mapping.items())
        with open(config.CHARACTER_MAP, 'w') as f:
            f.write(str(mapping))
        samples_per_epoch = self.train_flow.samples

        self.validation_flow = data_flow['valid']

        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as a:
                self.model = models.model_from_json(a.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            debugger.display(config.NAME + ' initialized from file.')
        else:
            if config.ALGORITHM == 'cnn':
                from .cnns import sequentialModel
                self.model, self.paras = sequentialModel(
                    self.train_flow.num_classes)
            elif config.ALGORITHM == 'res':
                from .resnet import res32
                self.model, self.paras = res32(self.train_flow.num_classes)

            with open(config.ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                plot_model(self.model, to_file=config.VISUAL_FILE,
                           show_shapes=True, show_layer_names=True)
                debugger.display(config.NAME + ' saved to file.')

        validation_samples = self.validation_flow.samples
        batch_size_validation = self.paras['valid_batch_size']
        self.steps_per_epoch = samples_per_epoch // self.paras['batch_size']
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
