from slantCorrect import correctSlant
from skimage import io

for i in range(1, 5):
    image = io.imread(str(i) + '.png')
    image = correctSlant(image)
    io.imsave(arr=image, fname='r'+str(i)+'.png')
