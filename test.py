from PIL import Image
import cv2 as cv
import numpy as np


img = Image.open("./imges/fingerprint.jpg");

img2 = img.convert('L')

img_np = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)

