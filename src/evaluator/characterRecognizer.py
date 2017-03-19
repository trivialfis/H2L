'''
Author:        fis
Created:             2016
Last modified: Feb 9 2017
Description:
A character_recognizer using trained convolutional net
'''
import sys
from keras import models
import numpy as np
from configuration import characterRecognizerConfig as config


class recognizer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as a:
                self.model = models.model_from_json(a.read())
                self.model.load_weights(config.WEIGHTS_FILE)
                print(config.NAME + ' initialized from file')
            with open(config.CHARACTER_MAP, 'r') as mapFile:
                mapString = mapFile.read()
                self.charactersMap = eval(mapString)
        else:
            sys.exit('No weight file or json file found')

    def predictCharacter(self, img):
        result = self.model.predict(img, batch_size=1, verbose=False)
        result = self.charactersMap[np.argmax(result)]
        return result

    def predict(self, images):
        classCode = list(self.model.predict_classes(images,
                                                    batch_size=len(images),
                                                    verbose=False))
        characters = [self.charactersMap[code] for code in classCode]
        return characters

    def predictProbability(self, images):
        probabilities = list(self.model.predict(images,
                                                batch_size=len(images),
                                                verbose=False))
        probabilities = [np.max(probas) for probas in probabilities]
        return probabilities
