import glob

import cv2
import numpy as np

files = glob.glob("images/*.jpg")
latest_number = 0
for file in files:
    ratio = 1.42
    scaled_width = 1920 * ratio
    scaled_margin_width = (scaled_width - 1024) / 2

    m = np.float32([[1, 0, -600], [0, 1, -180]])

    im = cv2.imread(file)
    h, w, ch = im.shape
    im_trasformed = cv2.warpAffine(im, m, (w, h))
    im_resized = cv2.resize(im_trasformed, dsize=None, fx=ratio, fy=ratio)
    dst = im_resized[0:1024, 0:1024]

    cv2.imwrite(f"renumber_images/{latest_number}.jpg", im)
    print(f"renumbering {file} to renumber_images/{latest_number}.jpg")
    latest_number = latest_number + 1
