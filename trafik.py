import cv2
import numpy as np
from scipy.stats import itemfreq

def d_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]


cv2.waitKey(1)
frame = cv2.imread('4.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 37)
rows = img.shape[0]
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.4, rows/8)

if not circles is None:
    circles = np.uint16(np.around(circles))
    max_r, max_i = 0, 0
    for i in range(len(circles[:, :, 2][0])):
        if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
            max_i = i
            max_r = circles[:, :, 2][0][i]
    x, y, r = circles[:, :, :][0][max_i]
    if y > r and x > r:
        square = frame[y-r:y+r, x-r:x+r]
        cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0, 255, 255), 2)
        dcolor = d_color(square, 2)
        # print(dcolor)
        if dcolor[2] > 100:
            if(dcolor[0]<150 and dcolor[1]<150):
                print('TASIT GIREMEZ')
            else:
                print('HIZ LIMITI')
        elif dcolor[0] > 80:
            z0 = square[y-r+square.shape[0]//4:square.shape[0]*2//4, x-r+square.shape[0]//4:square.shape[0]*2//4]
            z0color = d_color(z0, 1)

            z1 = square[square.shape[0]*2//4:square.shape[0]*3//4, square.shape[0]*2//4:square.shape[0]*3//4]
            z1color = d_color(z1, 1)

            # print(zone_0_color)
            # print(zone_1_color)
            # print(zone_2_color)
            cv2.rectangle(frame, (x-r+square.shape[0]//4, y-r+square.shape[0]//4),(square.shape[0]*2//4, square.shape[0]*2//4), (0, 255, 255), 2)
            cv2.rectangle(frame, (square.shape[0]*2//4, square.shape[0]*2//4),(square.shape[0]*3//4, square.shape[0]*3//4), (0, 0, 255), 2)
            # cv2.rectangle(frame, (square.shape[0]//8, square.shape[0]*3//8), (square.shape[1]//8, square.shape[1]*3//8), (0, 0, 255), 2)
            # cv2.rectangle(frame, (square.shape[0]*1//8, square.shape[0]*3//8), (square.shape[1]*3//8, square.shape[1]*5//8), (0, 0, 0), 2)
            # cv2.rectangle(frame, (square.shape[0]*3//8, square.shape[0]*5//8), (square.shape[1]*5//8, square.shape[1]*7//8), (0, 255, 0), 2)
            # cv2.rectangle(frame, (square.shape[0]*0//8, square.shape[0]*1//8), (square.shape[1]*1//8, square.shape[1]*2//8), (0, 255, 255), 2)
            if z1color[2] < 40:
                print("SAGA DON")
            else:
                print("DUZ GIT")
else:
    print("TRAFIK LEVHASI DEGIL")

cv2.imshow('resim', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()