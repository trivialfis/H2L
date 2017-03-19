'''
File:          lines_segmenter.py
Author:        fis
Created date:  28 Jan 2017
Last modified: 30 Jan 2017

Description:

'''
from keras import models
import numpy as np
from configuration import wordSegmenterConfig as config


class segmenter(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as architecture:
                self.model = models.model_from_json(architecture.read())
            self.model.load_weights(config.WEIGHTS_FILE)
            print(config.NAME + ' initialized from files')
        else:
            raise IOError('Model not found')

    def __extract(self, image):
        fragments = []
        for k in range(0, image.shape[1]):
            if k < config.SPACE:
                segment = np.zeros(
                    (config.HEIGHT, config.FRAGMENT_LENGTH),
                    dtype=np.float32)
                segment[:, config.SPACE-k:] = image[:, 0:config.SPACE+k+1]
            elif image.shape[1]-k-1 < config.SPACE:
                segment = np.zeros(
                    (config.HEIGHT, config.FRAGMENT_LENGTH), dtype=np.float32)
                temp = config.FRAGMENT_LENGTH - config.SPACE
                temp = temp + image.shape[1] - k - 1
                segment[:, 0:temp] = image[:, k-config.SPACE:]
            else:
                segment = np.array(image[:, k-config.SPACE:k+config.SPACE+1],
                                   dtype=np.float32)
            segment = segment.reshape(
                (config.HEIGHT, config.FRAGMENT_LENGTH, 1))
            fragments.append(segment)
        fragments = np.array(fragments, dtype=np.float32)
        return fragments

    def segment(self, line):
        '''Segment a line image and return a list of word images'''
        segmentationPoints = []
        fragments = self.__extract(line)
        prediction = self.model.predict(fragments, batch_size=len(line), verbose=False)
        # for l in line:
        #     isSeg = self.model.predict(l, batch_size=1, verbose=False)[0] >= 0.5
        #     print('is seg: ', isSeg)
        #     if isSeg[0]:
        #         segmentationPoints.append(l)
        for i in range(prediction.shape[0]):
            if prediction[i] >= 0.5:
                segmentationPoints.append(i)
        if 0 not in segmentationPoints:
            segmentationPoints.insert(0, 0)
        if line.shape[1] - 1 not in segmentationPoints:
            segmentationPoints.append(line.shape[1] - 1)
        words = []
        if len(segmentationPoints) != 0:
            lastPoint = segmentationPoints[0]
            for i in range(1, len(segmentationPoints)):
                if segmentationPoints[i] - lastPoint < 3:
                    lastPoint = segmentationPoints[i]
                    continue
                else:
                    words.append(line[:, lastPoint:segmentationPoints[i]])
                    lastPoint = segmentationPoints[i]
        return words
