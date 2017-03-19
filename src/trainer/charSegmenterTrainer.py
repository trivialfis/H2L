'''
File:          charSegmenterTrainer.py
Author:        fis
Created:       Feb 2  2017
Last modified: Feb 12 2017
'''
import charSegmenterConfig as config
from wordsData import dataLoader

from keras import models
from keras.layers import Dense, Dropout
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import math


class charSegmenterTrainer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE) as wa:
                self.model = models.model_from_json(wa.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            print(config.NAME + ' initialized from file')
        else:
            self.model = models.Sequential()
            self.model.add(Dense(4, input_shape=(2, ),
                                 init='uniform', activation='sigmoid'))
            self.model.add(Dense(8, init='uniform', activation='sigmoid'))
            self.model.add(Dense(8, init='uniform', activation='sigmoid'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, init='uniform', activation='softmax'))
            with open(config.ARCHITECTURE_FILE, 'w') as jsonFile:
                architecture = self.model.to_json()
                print(architecture, file=jsonFile)
                print(config.NAME, ' architecture saved to file')
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
        )
        print(config.NAME + ' model compiled')

    def train(self):
        print(config.NAME + ' configuring data')
        loader = dataLoader()
        validationParas, validationLabels = loader.getValidationData()
        trainingParas, trainingLabels = loader.getTrainingData()

        def stepDecay(epoch):
            initialLearningRate = 1e-32
            drop = 0.90
            epochsDrop = 1.0
            learningRate = initialLearningRate * math.pow(
                drop,
                math.floor((1+epoch)/epochsDrop)
                )
            return learningRate

        callbacks = [
            ModelCheckpoint(
                config.WEIGHTS_FILE,
                monitor='val_acc',
                verbose=True,
                save_best_only=True,
                save_weights_only=True
            ),
            LearningRateScheduler(stepDecay)
        ]
        print('Start training')
        self.model.fit(
            trainingParas, trainingLabels,
            batch_size=config.BATCH_SIZE,
            nb_epoch=config.EPOCH,
            verbose=True,
            validation_data=(validationParas, validationLabels),
            callbacks=callbacks
        )


if __name__ == '__main__':
    kevin = charSegmenterTrainer()
    try:
        kevin.train()
    except KeyboardInterrupt:
        print('\nExit')
