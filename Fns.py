import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from fastapi import UploadFile
from io import BytesIO
import base64


def read_image_Fn(file: UploadFile):
    extension = file.filename.split(".")[-1]
    extension = extension.upper()
    supportedTypes = ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'WEBP']

    if extension not in supportedTypes:
        return None
    image = Image.open(file.file)
    if image.format in supportedTypes:
        return image
    return None


def convertToGrayScale_Fn(image: Image):
    gray = ImageOps.grayscale(image)
    return gray


def convertToRGB_Fn(image: Image):
    return image.convert("RGB")


def gaussianBlur_Fn(image: Image, ksize: int = 5):
    image = np.array(image)
    image = cv.GaussianBlur(image, (ksize, ksize), cv.BORDER_DEFAULT)
    return Image.fromarray(image)


def encodeImage_Fn(image: Image):
    # Convert the grayscale image to bytes
    with BytesIO() as buffer:
        image.save(buffer, "PNG")
        img_bytes = buffer.getvalue()

    # Encode the bytes as base64
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64
