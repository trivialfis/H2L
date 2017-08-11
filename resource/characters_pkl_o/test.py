#!/bin/python3
import pickle
from matplotlib import pyplot as plt


def run():
    with open('./characters4.pkl', 'rb') as f:
        characters = pickle.load(f)
    images, labels = zip(*characters)
    print(len(images), len(labels))
    for im in images:
        image = im.reshape(im.shape[:-1])
        print(image.shape)
        plt.imshow(image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print('\nExit')
