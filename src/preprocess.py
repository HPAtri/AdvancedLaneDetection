import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def warp(image):
    w = image.shape[1]
    h = image.shape[0]

    src = np.float32([[545, 465], [w - 500, 465], [400, h], [w, h]])  # 原始图像中 4点坐标
    dst = np.float32([[200, 0], [w, 0], [200, h ], [w , h]])  # 变换图像中 4点坐标

    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, invM

def threshold(image):
    ret, image = cv2.threshold(image, 170, 230, cv2.THRESH_BINARY)
    if(ret == False):
        print('Error in thresholding')
    else:
        return image