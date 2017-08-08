from matplotlib import pyplot as plt
from skimage import io

H2L_DEBUG = False


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
        print(*strings)

    def log(self, data):
        if H2L_DEBUG:
            with open('h2l.log', 'a') as f:
                f.write(data)
