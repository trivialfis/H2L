import cv2
from evaluator import h2l_debug


debugging = h2l_debug.h2l_debugger()


def binarize3d(image):
    '''
    Binarize IMAGE, return the binarized version.
    '''
    img = cv2.fastNlMeansDenoisingColored(image, None, 3, 10, 7, 21)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray
    imgray = cv2.fastNlMeansDenoising(imgray, None, h=2,
                                      templateWindowSize=5,
                                      searchWindowSize=7)

    mask = cv2.adaptiveThreshold(
        imgray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 5, 2)
    debugging.display('binarize_image: type:', type(mask),
                      'dtype: ', mask.dtype,
                      'shape: ', mask.shape)
    return mask


def binarize2d(image):
    result = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return result


def binarize2d_inv(image):
    result = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return result
