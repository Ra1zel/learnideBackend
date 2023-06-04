from PIL import Image
import cv2 as cv
import numpy as np
from fastapi import UploadFile


def read_image(file: UploadFile):
    image = Image.open(file.file)
    image = np.array(image)
    return image


def createResponse(type: str, valueType: str, value: any):
    return {"type": type, "valueType": valueType, "value": value}


def pilToNumpy(img: Image):
    if img.mode == '1':
        return np.array(img)
    elif img.mode == 'L':
        return np.array(img)
    # elif img.mode == 'P':
    #     return cv.cvtColor(np.array(img.convert('RGB')), cv.COLOR_RGB2BGR)
    elif img.mode == 'RGB':
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    elif img.mode == 'RGBA':
        return cv.cvtColor(np.array(img), cv.COLOR_RGBA2BGRA)
    elif img.mode == 'CMYK':
        return cv.cvtColor(np.array(img), cv.COLOR_CMYK2BGR)
    elif img.mode == 'YCbCr':
        return cv.cvtColor(np.array(img), cv.COLOR_YCrCb2BGR)
    elif img.mode == 'LAB':
        return cv.cvtColor(np.array(img), cv.COLOR_LAB2BGR)
    elif img.mode == 'HSV':
        return cv.cvtColor(np.array(img), cv.COLOR_HSV2BGR)
    elif img.mode == 'I':
        return np.array(img)
    elif img.mode == 'F':
        return np.array(img)
    else:
        raise ValueError(f"Unsupported image mode: {img.mode}")

def numpyToPil(img: np.ndarray):
    if img.ndim == 2:
        return Image.fromarray(img)
    elif img.ndim == 3:
        if img.shape[2] == 3:
            return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        elif img.shape[2] == 4:
            return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")