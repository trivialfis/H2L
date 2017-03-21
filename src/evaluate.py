'''
File:          evaluate.py
Author:        fis
Created:       Feb 17 2017
Last modified: Mar 20 2017
'''
from evaluator import baseCharSegmenter, wordSegmenter, heuristicSegmenter
from evaluator import characterRecognizer
from evaluator import toLaTeX
from evaluator import extractDocument
from evaluator.LineSegment import LineSegment
from configuration import characterRecognizerConfig as crconfig
from preprocessing import reform
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from subprocess import call
import os

from preprocessing.reform import saveImages

ws = wordSegmenter.segmenter()
bcs = baseCharSegmenter.segmenter()
hs = heuristicSegmenter.segmenter()
cr = characterRecognizer.recognizer()

from matplotlib import pyplot as plt


def generate(image, preprocess=False):
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    image = reform.rescale(image, 64)
    wordImages = ws.segment(image)
    charactersList = []
    saveImages.counter = 0
    print('Word num ', len(wordImages))
    for word in wordImages:
        # characterImages = bcs.segment(word)  # [2:-2]
        characterImages = hs.segment(word)
        if len(characterImages) == 0:
            continue
        characterImages = [reform.resize(
            char, (crconfig.IMG_COLS, crconfig.IMG_COLS))
                           for char in characterImages]
        saveImages(characterImages, prefix='bef')
        characterImages = [reform.removeEdge(char)
                           for char in characterImages]
        saveImages(characterImages, prefix='aft')
        characterImages = [char.reshape(char.shape + (1, ))
                           for char in characterImages]
        characterImages = np.array(characterImages, dtype=np.float32)
        characters = cr.predict(characterImages)
        print('number of prediction', len(characters))
        characters.append(' ')
        charactersList += characters
        print('number of character images', len(charactersList))
        print(charactersList)
        characters = ''.join(charactersList)
    return characters


def heursiticGenerate(image):

    def findMid(line):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(line.shape[0]):
            if np.max(line[i, :]) != 0:
                top = i
                break
        for i in range(line.shape[0]):
            if np.max(line[-i, :]) != 0:
                bottom = line.shape[0] - i + 1
                break
        middle = (top + bottom) / 2
        return middle

    def isSuper(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(character.shape[0]):
            if np.max(character[-i, :]) != 0:
                bottom = character.shape[1] - i + 1
                break
        if bottom < middle:
            return True
        else:
            return False

    def segmentCharacters(line, root):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        characterImages = hs.segment(line)
        characterImages = [reform.removeEdge(char)
                           for char in characterImages]
        characterImages = [reform.resize(
            char, (crconfig.IMG_ROWS, crconfig.IMG_COLS))
                           for char in characterImages]
        characterImages = [reform.binarize(char, mode='greater')
                           for char in characterImages]
        characterImages = [char.reshape(char.shape+(1, ))
                           for char in characterImages]
        characterImages = np.array(characterImages, dtype=np.float32)
        characters = cr.predict(characterImages)
        for char in characters:
            item = ET.SubElement(root, 'item', name='oper_or_num')
            ET.SubElement(item, 'oper_or_num').text = char

    image = extractDocument.drawContours(image)
    print('Document extracted')
    image = Image.fromarray(image)
    image = np.array(image.convert('L'), dtype=np.float32)
    plt.imshow(image, cmap='gray')
    plt.show()
    image = reform.binarize(image, mode='less', threshold='max')
    plt.imshow(image, cmap='gray')
    plt.show()
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    lineImages = LineSegment.segment(image)
    lineImages = [reform.rescale(line, 64) for line in lineImages]
    root = ET.Element('formula', shelf='math')
    for line in lineImages:
        segmentCharacters(line, root)
    xmlFilename = 'temp.xml'
    tree = ET.ElementTree(root)
    tree.write(xmlFilename)
    converter = toLaTeX.Formula(xmlFilename)
    # texStr = converter.expection()
    texFileName = 'texFile.tex'
    converter.write_tex_file(texFileName)
    call(['pdflatex', texFileName])
    with open(os.devnull, 'wb') as dump:
        call(['./clean.sh'], stdout=dump, stderr=dump)


if __name__ == '__main__':
    heursiticGenerate()
    # generate()
