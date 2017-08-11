from matplotlib import pyplot as plt
from skimage import io

H2L_DEBUG = False
ORANGE = '\033[38;2;255;185;0m'
RESET = '\033[0m\n'


class h2l_debugger():
    def plot(self, images, caption=None):
        if H2L_DEBUG:
            if caption is not None:
                print(caption)
            if type(images) is not list and type(images) is not tuple:
                images = [images]
            for img in images:
                plt.imshow(img, cmap='gray')
                plt.show()

    def save_img(self, image, caption):
        if H2L_DEBUG:
            io.imsave(fname=caption + '.png', arr=image)

    def display(self, *strings):
        for s in strings:
            print(ORANGE, s, RESET, end=' ')

    def image_info(self, prefix, image):
        print(prefix, '\n',
              '  type : ', ORANGE, type(image), RESET,
              '  dtype: ', ORANGE, image.dtype, RESET,
              '  shape: ', ORANGE, image.shape, RESET)

    def log(self, data):
        if H2L_DEBUG:
            with open('h2l.log', 'a') as f:
                f.write(data)
