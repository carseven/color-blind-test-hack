import skimage.morphology as morphology
import numpy as np


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def apply_kmeans(image, K):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    vectorized = image.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    return result_image


def procesar_img(image):
    img = apply_brightness_contrast(image, 10, 70)
    img = apply_kmeans(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = apply_brightness_contrast(img, 10, 70)
    img = morphology.dilation(image=img, selem=morphology.disk(3))
    img = morphology.erosion(image=img, selem=morphology.disk(1))
    img = cv2.medianBlur(img, 15)
    return img
