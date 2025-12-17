import json
import os
import shutil
from random import *

import cv2
import matplotlib.pyplot as plt
import numpy as np

import augmentation


def pop_images_by_ratio(stack_images, initial_count, ratio):
    result_list = []

    count_to_pop = round(initial_count * ratio)

    for i in range(count_to_pop):
        if len(stack_images) == 0:
            break
        result_list.append(stack_images.pop())

    return result_list


def read_all_classes(root):
    return list(os.listdir(root))


def validate_ratios(ratios: dict):
    if sum(ratios.values()) != 1:
        raise Exception('Сумма соотношений не равна единице!')


def clear_result_dir():
    global RESULT_PATH
    shutil.rmtree(RESULT_PATH, ignore_errors=True)


def get_result_dir_and_filename(division: str, clazz: str, source_image_path: str):
    global RESULT_PATH
    return f'{RESULT_PATH}/{division}/{clazz}/', os.path.split(source_image_path)[1]


def copy_sources_to_divided(divided_images):
    global RESULT_PATH

    for clazz, divisions in divided_images.items():
        for division, images in divisions.items():
            for source_image_path in images:
                result_dir, filename = get_result_dir_and_filename(division, clazz, source_image_path)
                os.makedirs(result_dir, exist_ok=True)
                shutil.copy(source_image_path, f'{result_dir}/{filename}')


def create_augmented_image_by_path(image_path, augmentation_function=augmentation.augment_image):
    img = cv2.imread(image_path)
    if img is not None:
        augmented_img = augmentation_function(img)

        directory, full_filename = os.path.split(image_path)
        filename, extension = full_filename.rsplit('.', 1)
        augmented_filename = f'{directory}/{filename}_augmented.{extension}'

        cv2.imwrite(augmented_filename, augmented_img)
        return augmented_img


def read_all_class_images(class_path: str):
    global ALLOWED_EXTENSIONS
    result = []

    for filename in os.listdir(class_path):
        _, file_extension = os.path.splitext(filename)
        if any(extension == file_extension for extension in ALLOWED_EXTENSIONS):
            result.append(f'{class_path}/{filename}')

    return result


def divide_images(images: list, ratios: dict):
    divided_images = dict.fromkeys(ratios.keys())

    shuffled_images = images.copy()
    shuffle(shuffled_images)

    initial_count = len(shuffled_images)
    for division, ratio in ratios.items():
        divided_images[division] = pop_images_by_ratio(shuffled_images, initial_count, ratio)

    if len(shuffled_images) != 0:
        divided_images[list(divided_images.keys())[-1]].append(shuffled_images)

    return divided_images


def get_divided_images(source_path: str, ratios: dict) -> dict:
    classes = read_all_classes(SOURCE_PATH)
    print(f'Обнаруженные классы: {classes}\n')

    divided_images = {}
    for clazz in classes:
        class_images = read_all_class_images(f'{source_path}/{clazz}')
        divided_images[clazz] = divide_images(class_images, ratios)

    print(json.dumps(divided_images, indent=4, ensure_ascii=False))

    return divided_images


def augment_images_by_division(divided_images):
    augmented_images = []
    for clazz, divisions in divided_images.items():
        for division, images in divisions.items():
            if division in ['val', 'test']:
                continue
            for source_image_path in images:
                result_dir, filename = get_result_dir_and_filename(division, clazz, source_image_path)
                augmented_image = create_augmented_image_by_path(f'{result_dir}/{filename}')
                augmented_images.append(augmented_image)
    return augmented_images


def visualize_batch(images: list, cols: int = 5):
    rows = int(np.ceil(len(images) / cols))
    figsize = 10
    plt.figure(figsize=(figsize, figsize))
    for idx, img in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.show()


SOURCE_PATH = 'source_images'
RESULT_PATH = 'result'

ALLOWED_EXTENSIONS = ['.jpg']

RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def main():
    global SOURCE_PATH, RESULT_PATH, RATIOS

    validate_ratios(RATIOS)
    divided_images = get_divided_images(SOURCE_PATH, RATIOS)

    clear_result_dir()
    copy_sources_to_divided(divided_images)

    augmented_images = augment_images_by_division(divided_images)

    visualize_batch(augmented_images[:20])


if __name__ == '__main__':
    main()
