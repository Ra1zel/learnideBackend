import math
from typing import Optional
from enum import Enum
from fastapi import FastAPI, Form, Response, status

from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import base64
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from utils import createResponse, pilToNumpy, numpyToPil

app = FastAPI()

# Allow CORS
# origins = [
#     "http://localhost:3000",
#     "http://localhost:5173",
#     "http://localhost:5500",
#     "http://localhost:5501",
#     "http://localhost:5502",
#     "http://127.0.0.1:3000",
#     "http://127.0.0.1:5173",
#     "http://127.0.0.1:5500",
#     "http://127.0.0.1:5501",
#     "http://127.0.0.1:5502",
#     "https://learnide.el.r.appspot.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


class ConvertType(str, Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"


@app.get("/")
def home():
    return {"message": "FUCK YOU"}


@app.post("/api/convert/image/{convert_type}")
async def convert(
    response: Response, convert_type: ConvertType, image_base64: str = Form(...)
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        converted_img: Image.Image
        if convert_type == ConvertType.GRAYSCALE:
            # Convert the image to grayscale
            converted_img = img.convert("L")
            # Convert the grayscale image back to bytes
            gray_bytes = io.BytesIO()
            converted_img.save(gray_bytes, format="JPEG")
            gray_bytes = gray_bytes.getvalue()

            # Encode the grayscale image as base64 and send the response
            jsonRes = createResponse(
                "base64", "imagec1", base64.b64encode(gray_bytes).decode("utf-8")
            )
            # this response is sent for the single outclip that has the payload key of "data"
            return {"data": jsonRes}
        elif convert_type == ConvertType.RGB:
            # Convert the image to RGB
            if img.mode != "RGB":
                converted_img = img.convert("RGB")
            else:
                converted_img = img

            # Convert the RGB image and individual channels back to bytes
            rgb_bytes = io.BytesIO()
            converted_img.save(rgb_bytes, format=img.format)
            rgb_bytes = rgb_bytes.getvalue()

            red_bytes = io.BytesIO()
            converted_img.getchannel(0).save(red_bytes, format=img.format)
            red_bytes = red_bytes.getvalue()

            green_bytes = io.BytesIO()
            converted_img.getchannel(1).save(green_bytes, format=img.format)
            green_bytes = green_bytes.getvalue()

            blue_bytes = io.BytesIO()
            converted_img.getchannel(2).save(blue_bytes, format=img.format)
            blue_bytes = blue_bytes.getvalue()

            # Encode the RGB image as base64 and send the response
            color_data = createResponse(
                "base64", "imagec3", base64.b64encode(rgb_bytes).decode("utf-8")
            )
            red_data = createResponse(
                "base64", "imagec1", base64.b64encode(red_bytes).decode("utf-8")
            )
            green_data = createResponse(
                "base64", "imagec1", base64.b64encode(green_bytes).decode("utf-8")
            )
            blue_data = createResponse(
                "base64", "imagec1", base64.b64encode(blue_bytes).decode("utf-8")
            )

            # this response is sent for the single outclip that has the payload key of "data"
            return {
                "color": color_data,
                "red": red_data,
                "green": green_data,
                "blue": blue_data,
            }
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}


class SmoothType(str, Enum):
    BOX = "box"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


@app.post("/api/process/image/smooth/{smooth_type}")
async def smooth(
    response: Response,
    smooth_type: SmoothType,
    image_base64: str = Form(...),
    size: Optional[int] = Form(3),
    sigma: Optional[float] = Form(0),
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        # Convert the Pillow image to a NumPy array
        img_np = pilToNumpy(img)

        filtered_img: np.ndarray
        if smooth_type == SmoothType.BOX:
            # Apply the box filter using OpenCV
            filtered_img = cv.boxFilter(img_np, -1, (size, size))

        elif smooth_type == SmoothType.GAUSSIAN:
            # Apply the Gaussian filter using OpenCV
            filtered_img = cv.GaussianBlur(img_np, (size, size), sigma)

        elif smooth_type == SmoothType.MEDIAN:
            # Apply the median filter using OpenCV
            filtered_img = cv.medianBlur(img_np, size)

        # Convert the filtered image back to a Pillow image
        filtered_pil = numpyToPil(filtered_img)

        # Determine the number of channels in the filtered image
        num_channels = len(filtered_pil.getbands())

        # Convert the filtered image to bytes
        filtered_bytes = io.BytesIO()
        filtered_pil.save(filtered_bytes, format=img.format)
        filtered_bytes = filtered_bytes.getvalue()

        # Encode the filtered image as base64 and set the value type in the response
        value_type = "imagec" + str(num_channels)
        json_res = createResponse(
            "base64", value_type, base64.b64encode(filtered_bytes).decode("utf-8")
        )
        # this response is sent for the single outclip that has the payload key of "data"
        return {"data": json_res}
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}


class ThresholdType(str, Enum):
    BINARY = "binary"
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"


@app.post("/api/process/image/threshold/{threshold_type}")
async def threshold(
    response: Response,
    threshold_type: ThresholdType,
    image_base64: str = Form(...),
    thresh: Optional[int] = Form(127),
    max_value: Optional[int] = Form(255),
    window: Optional[int] = Form(15),
    c: Optional[int] = Form(2),
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        # Convert the Pillow image to a NumPy array
        img_np = pilToNumpy(img)

        filtered_img: np.ndarray
        if threshold_type == ThresholdType.BINARY:
            # Apply the binary threshold using OpenCV
            filtered_img = cv.threshold(img_np, thresh, max_value, cv.THRESH_BINARY)[1]
        # check if image is single channe;
        if len(img.getbands()) != 1:
            raise Exception("Image must be single channel")

        if threshold_type == ThresholdType.OTSU:
            # Apply the otsu threshold using OpenCV
            filtered_img = cv.threshold(
                img_np, 0, max_value, cv.THRESH_BINARY + cv.THRESH_OTSU
            )[1]

        elif threshold_type == ThresholdType.ADAPTIVE_MEAN:
            # Apply the adaptive mean threshold using OpenCV
            filtered_img = cv.adaptiveThreshold(
                img_np,
                max_value,
                cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,
                window,
                c,
            )

        elif threshold_type == ThresholdType.ADAPTIVE_GAUSSIAN:
            # Apply the adaptive gaussian threshold using OpenCV
            filtered_img = cv.adaptiveThreshold(
                img_np,
                max_value,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                window,
                c,
            )

        # Convert the filtered image back to a Pillow image
        filtered_pil = numpyToPil(filtered_img)

        # Determine the number of channels in the filtered image
        num_channels = len(filtered_pil.getbands())

        # Convert the filtered image to bytes
        filtered_bytes = io.BytesIO()
        filtered_pil.save(filtered_bytes, format=img.format)
        filtered_bytes = filtered_bytes.getvalue()

        # Encode the filtered image as base64 and set the value type in the response
        value_type = "imagec" + str(num_channels)
        json_res = createResponse(
            "base64", value_type, base64.b64encode(filtered_bytes).decode("utf-8")
        )
        # this response is sent for the single outclip that has the payload key of "data"
        return {"data": json_res}
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}


class ArithmeticType(str, Enum):
    MULTIPLY_CONSTANT = "multiply_constant"
    ADD_IMAGES = "add_images"
    SUBTRACT_IMAGES = "subtract_images"
    MAX_OF_TWO_IMAGES = "max_pick_of_two_images"
    MIN_OF_TWO_IMAGES = "min_pick_of_two_images"
    CONVOVLE = "convolution"


@app.post("/api/process/image/arithmetic/{arithmetic_type}")
async def arithmetic(
    response: Response,
    arithmetic_type: ArithmeticType,
    image_base64: Optional[str] = Form(...),
    image_base64_2: Optional[str] = Form(None),
    k: Optional[float] = Form(1),
    k_2: Optional[float] = Form(1),
    kernel: Optional[str] = Form(None),
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        # Convert the Pillow image to a NumPy array
        img_np = pilToNumpy(img)

        filtered_img: np.ndarray
        if arithmetic_type == ArithmeticType.MULTIPLY_CONSTANT:
            # Apply the multiply constant using OpenCV
            filtered_img = np.multiply(img_np, k)

        elif arithmetic_type != ArithmeticType.CONVOVLE:
            image2 = base64.b64decode(image_base64_2)
            img2 = Image.open(io.BytesIO(image2))
            img_np2 = pilToNumpy(img2)
            # check if images are of same size and depth
            if img_np.shape[:2] != img_np2.shape[:2]:
                raise Exception("Images must be of same size")

            if len(img.getbands()) != len(img2.getbands()):
                raise Exception("Images must be of same depth")

            if arithmetic_type == ArithmeticType.ADD_IMAGES:
                filtered_img = np.add(np.multiply(img_np, k), np.multiply(img_np2, k_2))

            elif arithmetic_type == ArithmeticType.SUBTRACT_IMAGES:
                filtered_img = np.subtract(
                    np.multiply(img_np, k), np.multiply(img_np2, k_2)
                )

            elif arithmetic_type == ArithmeticType.MAX_OF_TWO_IMAGES:
                filtered_img = np.maximum(
                    np.multiply(img_np, k), np.multiply(img_np2, k_2)
                )

            elif arithmetic_type == ArithmeticType.MIN_OF_TWO_IMAGES:
                filtered_img = np.minimum(
                    np.multiply(img_np, k), np.multiply(img_np2, k_2)
                )

        elif arithmetic_type == ArithmeticType.CONVOVLE:
            kernel2D = kernel.split(",")
            kernel2D = [float(i) for i in kernel2D]
            kernel2D = np.array(kernel2D)
            kernel_size = int(math.sqrt(len(kernel2D)))
            kernel2D = np.reshape(kernel2D, (kernel_size, kernel_size))
            filtered_img = cv.filter2D(img_np, -1, kernel2D)

        # cap the values to 255
        filtered_img[filtered_img > 255] = 255
        filtered_img = filtered_img.astype(np.uint8)

        # Convert the filtered image back to a Pillow image
        filtered_pil = numpyToPil(filtered_img)

        # Determine the number of channels in the filtered image
        num_channels = len(filtered_pil.getbands())

        # Convert the filtered image to bytes
        filtered_bytes = io.BytesIO()
        filtered_pil.save(filtered_bytes, format=img.format)
        filtered_bytes = filtered_bytes.getvalue()

        # Encode the filtered image as base64 and set the value type in the response
        value_type = "imagec" + str(num_channels)
        json_res = createResponse(
            "base64", value_type, base64.b64encode(filtered_bytes).decode("utf-8")
        )

        # this response is sent for the single outclip that has the payload key of "data"
        return {"data": json_res}
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}


class MorphType(str, Enum):
    ERODE = "erode"
    DILATE = "dilate"
    OPEN = "open"
    CLOSE = "close"


class ShapeType(str, Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    CROSS = "cross"


@app.post("/api/process/image/morph/{morph_type}")
async def morph(
    response: Response,
    morph_type: MorphType,
    image_base64: Optional[str] = Form(...),
    iterations: Optional[int] = Form(1),
    shape: Optional[ShapeType] = Form(ShapeType.RECTANGLE),
    size: Optional[int] = Form(3),
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        # Convert the Pillow image to a NumPy array
        img_np = pilToNumpy(img)

        filtered_img: np.ndarray

        kernel: np.ndarray
        if shape == ShapeType.RECTANGLE:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
        elif shape == ShapeType.ELLIPSE:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        elif shape == ShapeType.CROSS:
            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (size, size))

        if morph_type == MorphType.ERODE:
            filtered_img = cv.erode(img_np, kernel, iterations=iterations)
        elif morph_type == MorphType.DILATE:
            filtered_img = cv.dilate(img_np, kernel, iterations=iterations)
        elif morph_type == MorphType.OPEN:
            filtered_img = cv.morphologyEx(
                img_np, cv.MORPH_OPEN, kernel, iterations=iterations
            )
        elif morph_type == MorphType.CLOSE:
            filtered_img = cv.morphologyEx(
                img_np, cv.MORPH_CLOSE, kernel, iterations=iterations
            )

        # cap the values to 255
        filtered_img[filtered_img > 255] = 255
        filtered_img = filtered_img.astype(np.uint8)

        # Convert the filtered image back to a Pillow image
        filtered_pil = numpyToPil(filtered_img)

        # Determine the number of channels in the filtered image
        num_channels = len(filtered_pil.getbands())

        # Convert the filtered image to bytes
        filtered_bytes = io.BytesIO()
        filtered_pil.save(filtered_bytes, format=img.format)
        filtered_bytes = filtered_bytes.getvalue()

        # Encode the filtered image as base64 and set the value type in the response
        value_type = "imagec" + str(num_channels)
        json_res = createResponse(
            "base64", value_type, base64.b64encode(filtered_bytes).decode("utf-8")
        )

        # this response is sent for the single outclip that has the payload key of "data"
        return {"data": json_res}
    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}


class UtilityType(str, Enum):
    FFT = "fft"
    HISTOGRAM = "histogram"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    POLAR = "polar"
    NON_MAXIMAL_SUPPRESSION = "non_maximal_suppression"


@app.post("/api/process/image/utility/{utility_type}")
async def freq(
    response: Response,
    utility_type: UtilityType,
    image_base64: Optional[str] = Form(...),
    image_base64_2: Optional[str] = Form(None),
):
    try:
        # Decode the base64 encoded image
        image = base64.b64decode(image_base64)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image))

        # Convert the Pillow image to a NumPy array
        img_np = pilToNumpy(img)

        processed_img: np.ndarray

        if utility_type == UtilityType.FFT:
            if len(img_np.shape) == 3:
                img_np = np.mean(img_np, axis=-1)
            if img_np.shape[0] > 512 or img_np.shape[1] > 512:
                img_np = np.resize(img_np, (512, 512))
            processed_img = np.fft.fft2(img_np)
            processed_img = np.fft.fftshift(processed_img)
            processed_img = np.log(np.abs(processed_img) + 1)
            processed_img = processed_img / np.max(processed_img)
            processed_img = processed_img * 255

        elif utility_type == UtilityType.HISTOGRAM_EQUALIZATION:
            if len(img.getbands()) != 1:
                raise Exception("Image must be grayscale")
            processed_img = cv.equalizeHist(img_np)

        elif utility_type == UtilityType.HISTOGRAM:
            if len(img.getbands()) == 1:
                # Create a histogram of pixel values in the image
                plt.hist(img_np.ravel(), bins=256, range=(0, 255))
                plt.xlim([-0.5, 255.5])
                plt.xlabel("Pixel value")
                plt.ylabel("Frequency")

                # Convert the Matplotlib figure to a PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.cla()
                buf.seek(0)
                hist_img = Image.open(buf)

                # Convert the histogram image to a NumPy array
                processed_img = pilToNumpy(hist_img)
            elif len(img.getbands()) == 3:
                # Create a histogram of pixel values in the image
                color = ("b", "g", "r")
                for i, col in enumerate(color):
                    histr = cv.calcHist([img_np], [i], None, [256], [0, 256])
                    # plt.plot(histr, color=col)
                    plt.fill_between(
                        np.arange(256), histr.flatten(), color=col, alpha=0.3
                    )
                    plt.xlim([0, 256])
                plt.xlabel("Pixel value")
                plt.ylabel("Frequency")

                # Convert the Matplotlib figure to a PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.cla()
                buf.seek(0)
                hist_img = Image.open(buf)

                # Convert the histogram image to a NumPy array
                processed_img = pilToNumpy(hist_img)

        elif utility_type == UtilityType.POLAR:
            image2 = base64.b64decode(image_base64_2)
            img2 = Image.open(io.BytesIO(image2))
            grad_y = pilToNumpy(img2).astype(np.float32)
            grad_x = img_np.astype(np.float32)

            if len(img.getbands()) != 1 or len(img2.getbands()) != 1:
                raise Exception("Images must be Single Channel")

            if grad_x.shape != grad_y.shape:
                raise Exception("Images must be the same size")

            mag_img, phase_img = cv.cartToPolar(grad_x, grad_y, angleInDegrees=False)
            phase_img = (phase_img + 180) * 180 / np.pi

            mag_img = mag_img.astype(np.uint8)
            mag_img_bytes = io.BytesIO()
            numpyToPil(mag_img).save(mag_img_bytes, format="png")
            mag_img_bytes = mag_img_bytes.getvalue()

            phase_img = phase_img.astype(np.uint8)
            phase_img_bytes = io.BytesIO()
            numpyToPil(phase_img).save(phase_img_bytes, format="png")
            phase_img_bytes = phase_img_bytes.getvalue()

            json_res1 = createResponse(
                "base64", "imagec1", base64.b64encode(mag_img_bytes).decode("utf-8")
            )
            json_res2 = createResponse(
                "base64", "imagec1", base64.b64encode(phase_img_bytes).decode("utf-8")
            )

            return {"magnitude": json_res1, "phase": json_res2}

        elif utility_type == UtilityType.NON_MAXIMAL_SUPPRESSION:
            image2 = base64.b64decode(image_base64_2)
            img2 = Image.open(io.BytesIO(image2))
            grad_ori = pilToNumpy(img2)
            grad_mag = img_np

            if len(img.getbands()) != 1 or len(img2.getbands()) != 1:
                raise Exception("Image must be Single Channel")
            if grad_mag.shape != grad_ori.shape:
                raise Exception("Images must be the same size")

            # write code for non maximal suppression here
            processed_img = np.zeros_like(grad_mag)
            for i in range(1, grad_mag.shape[0] - 1):
                for j in range(1, grad_mag.shape[1] - 1):
                    if grad_ori[i, j] < 22.5 or grad_ori[i, j] >= 157.5:
                        # Edge is horizontal
                        if (
                            grad_mag[i, j] >= grad_mag[i, j - 1]
                            and grad_mag[i, j] >= grad_mag[i, j + 1]
                        ):
                            processed_img[i, j] = grad_mag[i, j]
                    elif grad_ori[i, j] < 67.5:
                        # Edge is diagonal with slope -1
                        if (
                            grad_mag[i, j] >= grad_mag[i - 1, j - 1]
                            and grad_mag[i, j] >= grad_mag[i + 1, j + 1]
                        ):
                            processed_img[i, j] = grad_mag[i, j]
                    elif grad_ori[i, j] < 112.5:
                        # Edge is vertical
                        if (
                            grad_mag[i, j] >= grad_mag[i - 1, j]
                            and grad_mag[i, j] >= grad_mag[i + 1, j]
                        ):
                            processed_img[i, j] = grad_mag[i, j]
                    else:
                        # Edge is diagonal with slope 1
                        if (
                            grad_mag[i, j] >= grad_mag[i - 1, j + 1]
                            and grad_mag[i, j] >= grad_mag[i + 1, j - 1]
                        ):
                            processed_img[i, j] = grad_mag[i, j]

        # cap the values to 255
        processed_img[processed_img > 255] = 255
        processed_img = processed_img.astype(np.uint8)

        # Convert the filtered image back to a Pillow image
        filtered_pil = numpyToPil(processed_img)

        # Determine the number of channels in the filtered image
        num_channels = len(filtered_pil.getbands())

        # Convert the filtered image to bytes
        filtered_bytes = io.BytesIO()
        filtered_pil.save(filtered_bytes, format="PNG")
        filtered_bytes = filtered_bytes.getvalue()

        # Encode the filtered image as base64 and set the value type in the response
        value_type = "imagec" + str(num_channels)
        json_res = createResponse(
            "base64", value_type, base64.b64encode(filtered_bytes).decode("utf-8")
        )

        # this response is sent for the single outclip that has the payload key of "data"
        return {"data": json_res}

    except Exception as e:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": True, "message": e.__str__()}
