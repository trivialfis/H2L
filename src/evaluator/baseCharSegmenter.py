'''
File:          baseCharSegmenter.py
Author:        fis
Created:       Feb 9  2017
Last modified: Feb 26 2017
'''
from keras import models
import numpy as np
from configuration import baseSegmenterConfig as config
from normalization.slantCorrect import correctSlant
from preprocessing import reform
from skimage import transform


class segmenter(object):
    def __init__(self):
        if config.baseModelExists():
            with open(config.BASE_ARCHITECTURE_FILE, 'r') as architecture:
                self.model = models.model_from_json(architecture.read())
            self.model.load_weights(config.BASE_WEIGHTS_FILE)
            print(config.BASE_NAME + ' initialized from files')
        else:
            raise IOError('Model not found')

    def __extractFragments(self, image):
        fragments = []
        for k in range(0, image.shape[1]):
            if image.shape[1] < config.FRAGMENT_LENGTH:
                segment = np.zeros((config.HEIGHT, config.FRAGMENT_LENGTH),
                                   dtype=np.float32)
                segment[:, 0:image.shape[1]] = image
            elif k < config.SPACE:
                segment = np.zeros((config.HEIGHT, config.FRAGMENT_LENGTH),
                                   dtype=np.float32)
                segment[:, config.SPACE-k:] = image[:, 0:config.SPACE+k+1]
            elif image.shape[1]-k-1 < config.SPACE:
                segment = np.zeros((config.HEIGHT, config.FRAGMENT_LENGTH),
                                   dtype=np.float32)
                temp = config.FRAGMENT_LENGTH - config.SPACE + image.shape[1]
                temp = temp - k - 1
                segment[:, 0:temp] = image[:, k-config.SPACE:]
            else:
                segment = np.array(image[:, k-config.SPACE:k+config.SPACE+1],
                                   dtype=np.float32)
            fragments.append(
                segment.reshape((config.HEIGHT, config.FRAGMENT_LENGTH, 1)))
        return np.array(fragments)

    def __extractCharacters(self, image, segmentationPoints):
        lastPoint = 0
        characterList = []
        '''
        for i in range(1, len(segmentationPoints)):
            character = np.zeros((config.HEIGHT, config.HEIGHT))
            rows, cols = image.shape
            length = segmentationPoints[i] - segmentationPoints[lastPoint]
            if length <= 0:
                raise ValueError(
                    'Negative length, sp[i]:'+str(
                        segmentationPoints[i]
                    )+' lp:'+str(
                        segmentationPoints[lastPoint]
                    )
                )
            if length > config.HEIGHT:
                ratio = config.HEIGHT / length
            else:
                ratio = 1
            resized = transform.rescale(
                image[:,
                      segmentationPoints[lastPoint]:
                      segmentationPoints[lastPoint]+length],
                ratio)
            colStart = (config.HEIGHT - resized.shape[1]) // 2
            rowStart = (config.HEIGHT - resized.shape[0]) // 2
            character[
                rowStart:rowStart+resized.shape[0],
                colStart:colStart+resized.shape[1]
            ] = resized
            lastPoint = i
            characterList.append(character)
        '''
        for point in segmentationPoints:
            if point+round(config.HEIGHT*2/3) < image.shape[1]:
                character = image[:, point:point+round(config.HEIGHT*2/3)]
            else:
                character = np.zeros((config.HEIGHT, config.HEIGHT))
                character[:, 0:image.shape[1]-point] = image[:, point:]
            characterList.append(character)

        # print('in extract characters: ', len(characterList))
        return characterList

    def __segment(self, word, isSub=False):
        '''
        return: the initial segmentation points
        '''
        isVerbose = False
        fragments = self.__extractFragments(word)
        if not isSub:
            predictions = self.model.predict_classes(
                fragments, batch_size=len(fragments), verbose=isVerbose)
            tempPoints = []
            for i in range(len(predictions)):
                if predictions[i]:
                    tempPoints.append(i)
        else:
            predictions = self.model.predict_proba(
                fragments, batch_size=len(fragments), verbose=isVerbose
            )[2:-2]
            tempPoints = [np.argmax(predictions)]
            tempPoints.insert(0, 0)
            tempPoints.append(len(predictions)-1)
        points = tempPoints.copy()
        index = [0]
        if len(tempPoints) > 0:
            for i in range(0, len(tempPoints)-1):
                if tempPoints[i+1] - tempPoints[i] > config.AW:
                    frag = word[:, tempPoints[i]:tempPoints[i+1]]
                    temp = self.__segment(frag, isSub=True)
                    newPoints = [p for p in temp if p not in points]
                    newPoints = [p + tempPoints[i] for p in newPoints]
                    for j in range(len(newPoints)):
                        points.insert(index[0]+i+j+1, newPoints[j])
                    index[0] = index[0] + len(newPoints)
        elif word.shape[1] < config.AW:
            points = [0, word.shape[1]-1]
        else:
            points = self.__segment(word, isSub=True)
        return points

    def segment(self, image):
        image = correctSlant(image)
        fragments = self.__extractFragments(image)
        predictions = list(self.model.predict_classes(
            fragments, batch_size=len(fragments), verbose=False
        ))
        probabilities = list(self.model.predict_proba(
            fragments, batch_size=len(fragments), verbose=False
        ))
        if len(predictions) != len(probabilities):
            raise ValueError('length of predictions is not equal to the length of probabilities')
        for i in range(len(fragments)):
            if np.sum(fragments[i][:, config.SPACE]) == 0:
                predictions[i] = True
                probabilities[i] = 1.0
        predictions = [i for i in range(len(predictions)) if predictions[i]]
        if 0 not in predictions:
            predictions.insert(0, 0)
        if image.shape[1]-1 not in predictions:
            predictions.append(image.shape[1]-1)
        i = 0
        filtered = []
        while i < len(predictions):
            start, end = i, i
            for j in range(start+1, len(predictions)):
                if predictions[j] - predictions[start] <= config.HEIGHT * 2/3:
                    end = j
                else:
                    break
            if start != end:
                i += (end - start)
                if end+1 == len(predictions):
                    end -= 1
                maxlikilihood = max(
                    probabilities[predictions[start]:predictions[end+1]]
                )
                index = probabilities.index(maxlikilihood,
                                            predictions[start],
                                            predictions[end+1])
                filtered.append(index)
            else:
                filtered.append(predictions[start])
            i += 1
        characterImages = self.__extractCharacters(image, filtered)
        # print('In baseCharSegmenter:\n',
        #       'character images: ', len(characterImages),
        #       '\n filtered: ', len(filtered), filtered)
        # return filtered
        return characterImages

    def overSegment(self, image):
        '''Get the segmentation points and tries to clean it'''
        image = correctSlant(image)
        segmentationPoints = self.__segment(image)
        segmentationPoints.sort()
        temp = [[point, True] for point in segmentationPoints]
        i = 0
        while i < len(segmentationPoints)-1:
            if segmentationPoints[i+1] - segmentationPoints[i] < 3:
                sumPoints = segmentationPoints[i] + segmentationPoints[i+1]
                count = 2
                for j in range(i+1, len(segmentationPoints)-1):
                    if segmentationPoints[j+1] - segmentationPoints[j] < 3:
                        sumPoints += segmentationPoints[j+1]
                        count += 1
                    else:
                        break
                average = round(sumPoints // count)
                for point in temp[i:i+count]:
                    point[1] = False
                temp[i+count-1] = (average, True)
                i = i + count - 2
            i = i + 1
        segmentationPoints = []
        for point in temp:
            if point[1]:
                segmentationPoints.append(point[0])
        characterImages = self.__extractCharacters(image, segmentationPoints)
        return characterImages

    def predictProbability(self, word):
        fragments = self.__extractFragments(word)
        probabilities = self.model.predict_proba(
            fragments,
            batch_size=len(fragments),
            verbose=False,
        )
        return probabilities
