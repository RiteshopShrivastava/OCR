import easyocr
# import urllib.request
import cv2
# import sys
# import imutils
import matplotlib.pyplot as plt
import numpy as np
# from IPython.display import Image
# import math
import re

# •	Resizing it.
# •	Converting the image to grayscale.
# •	Applying Gaussian blurring with a 5×5 kernel to reduce high-frequency noise.
# •	Computing the edge map via the Canny edge detector.

# image=cv2.imread("Scale_Reading.png")
# image = cv2.imread("D:\OCR\paint4_processed.png")
image = cv2.imread(r'D:\OCR\pos_new_images\2468000020_crop.PNG')
# print(type(image))
# image = imutils.resize(image, height=150)
# window_name = 'image'
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image
# blur = cv2.GaussianBlur(gray, (5, 5), 0)

# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
# plt.imshow(thresh)
# plt.show()

# Apply morphological operations to the image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#
# # window_name = 'image'
# # cv2_imshow(thresh)
#
# Join the fragmented digit parts
# kernel = np.ones((3, 3), np.uint8)
# dilation = cv2.dilate(thresh, kernel, iterations=1)
# erosion = cv2.erode(dilation, kernel, iterations=1)

# Inverting Image
inverted_image = np.invert(thresh)
# plt.imshow(inverted_image)
# plt.show()
# cv2.imshow('inverted_image', inverted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# saving inverted image
# cv2.imwrite('D:\OCR\meterimage3_gray_new.png',gray)
# cv2.imwrite('D:\OCR\test_processed.png', inverted_image)

reader = easyocr.Reader(['en'], gpu=False)
# result = reader.readtext('D:\OCR\paint4_inverted.png', contrast_ths = 0.05, adjust_contrast = 0.7, width_ths =0.7, decoder = 'beamsearch')
result = reader.readtext(inverted_image, detail=0, batch_size=50, adjust_contrast=0.5, allowlist='0123456789.')
# result = reader.readtext(inverted_image, detail=0, paragraph=False, slope_ths=0.1, ycenter_ths=0.5,
#                          height_ths=0.5, width_ths=0.5, contrast_ths=0.01, adjust_contrast=0.7,
#                          decoder='beamsearch', beamWidth=15, batch_size=5, allowlist='0123456789.', min_size=15,
#                          text_threshold=0.5)
print(result)
print(result[1])
# print((re.findall('\d+', result[0][1])))

# top_left = tuple(result[0][0][0])
# bottom_right = tuple(result[0][0][2])
# text = result[0][1]
# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
# img = cv2.putText(img, text, bottom_right, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

# result = reader.readtext(inverted_image)
# paint4- erossion & inverted images working fine
# metereimage3_preprocessed working fie
#
# for i in range(len(result)):
#     # print(i)
#     # print(*([int(x) for x in (re.findall('\d+',result[i][0]))]))
#     # print(*(re.findall('\d+\.\d+',result[i][1])))
#     # print(result[i][1])
#     print(*(re.findall('\d+', result[i][1])))

# iterate on all results
# cv2.imwrite('D:\OCR\nightmeter_processed.png',inverted_image)
# pro_image = cv2.imread(r'D:\OCR\paint4_inverted.png')
# for res in result:
#     top_left = tuple(res[0][0]) # top left coordinates as tuple
#     bottom_right = tuple(res[0][2]) # bottom right coordinates as tuple
#     # draw rectangle on image
#     cv2.rectangle(pro_image, top_left, bottom_right, (0, 255, 0), 2)
#     # write recognized text on image (top_left) minus 10 pixel on y
#     cv2.putText(pro_image, res[1], (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #result = [val for val in my_list if val.isdigit()]
# cv2.imwrite(r'D:\OCR\paint4_inverted_rectangle.png',pro_image)