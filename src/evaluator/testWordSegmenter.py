from matplotlib import pyplot as plt
from lines_segmenter import linesSegmenter
from PIL import Image
import numpy as np
import lines_segmenter_config as config

line = Image.open('./a01-030-02.png')
image = np.array(line)
rows, cols = image.shape
print(image.shape)
ratio = config.HEIGHT / rows
outputShape = (round(ratio*cols), config.HEIGHT)

line = np.array(
    Image.fromarray(image).resize(
        outputShape,
        Image.ANTIALIAS
    )
)
print('line.shape: ', line.shape)
kevin = linesSegmenter()
words = kevin.segment(line)
for w in words:
    plt.imshow(w)
    plt.show()
