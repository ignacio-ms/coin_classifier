import cv2
import numpy as np

img = cv2.imread('data/test_3.jpg', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (21, 21), cv2.BORDER_DEFAULT)

circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=30)
circles_rounded = np.uint16(np.round(circles))
print(f'{circles_rounded.shape[1]} coins found')

for i in circles_rounded[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

    # cv2.rectangle(img, (i[0] - i[2] - 10, i[1] - i[2] - 10), (i[0] + i[2] + 10, i[1] + i[2] + 10), (0, 0, 255), 2)

cv2.imshow('Find circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
