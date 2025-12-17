import math
import random

import cv2
import numpy as np


def rotate(image: np.ndarray, angle_range=(0, 360)) -> np.ndarray:
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def translate(image: np.ndarray, max_shift=(0.2, 0.2)) -> np.ndarray:
    h, w = image.shape[:2]
    max_dx = max_shift[0] * w
    max_dy = max_shift[1] * h
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def shear(image: np.ndarray, shear_range=(-0.3, 0.3)) -> np.ndarray:
    h, w = image.shape[:2]
    sh = random.uniform(*shear_range)
    M = np.array([[1, sh, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def flip(image: np.ndarray) -> np.ndarray:
    if random.choice([True, False]):
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def crop_resize(image: np.ndarray, scale_range=(0.6, 1.0)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    crop = image[top:top+new_h, left:left+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def inject_noise(image: np.ndarray, mean=0, var=10) -> np.ndarray:
    sigma = math.sqrt(var)
    gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def adjust_brightness(image: np.ndarray, brightness_range=(-50, 50)) -> np.ndarray:
    value = random.uniform(*brightness_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jitter(image: np.ndarray, brightness_range=(-30, 30), contrast_range=(0.7, 1.3),
           saturation_range=(0.7, 1.3), hue_range=(-10, 10)) -> np.ndarray:
    img = image.astype(np.float32)

    b = random.uniform(*brightness_range)
    img += b

    c = random.uniform(*contrast_range)
    img *= c

    img = np.clip(img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(*saturation_range)
    hsv[:, :, 0] += random.uniform(*hue_range)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def kernel_filter(image: np.ndarray, kernel_size_options=(3, 5, 7)) -> np.ndarray:
    if random.choice([True, False]):
        k = random.choice(kernel_size_options)
        return cv2.GaussianBlur(image, (k, k), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)


def random_erasing(image: np.ndarray, min_size=0.02, max_size=0.4) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    max_erase_h = max_size * h
    max_erase_w = max_size * w
    erase_h = int(round(random.uniform(min_size, max_size) * max_erase_h))
    erase_w = int(round(random.uniform(min_size, max_size) * max_erase_w))

    top = random.randint(0, h - erase_h)
    left = random.randint(0, w - erase_w)
    img[top:top+erase_h, left:left+erase_w] = np.random.randint(0, 255, (erase_h, erase_w, 3))
    return img


def hide_and_seek(image: np.ndarray, grid_sizes=(2, 20), max_deleted_squares=10) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    grid_size = random.randint(*grid_sizes)
    square_h = h // grid_size
    square_w = w // grid_size

    deleted_squares = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < 0.5:
                y1, y2 = i * square_h, (i + 1) * square_h
                x1, x2 = j * square_w, (j + 1) * square_w
                img[y1:y2, x1:x2] = 0
                deleted_squares += 1
                if deleted_squares >= max_deleted_squares:
                    return img

    return img


def augment_image(image: np.ndarray) -> np.ndarray:
    functions = [
        rotate,
        translate,
        shear,

        flip,
        crop_resize,
        inject_noise,
        adjust_brightness,
        jitter,
        kernel_filter,

        random_erasing,
        hide_and_seek
    ]
    func = random.choice(functions)
    return func(image)


def augment_images(batch: list) -> list:
    return [augment_image(img) for img in batch]
