import numpy as np
import cv2

image_dir = './data/'
lebels_dir = './data/XML Metadata'

def pre_process_image (image_path, target_height):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    image_resized = cv2.resize(image, (new_width, target_height))
    image_normalized = image_resized / 255.0
    return image_normalized

def data_augmentor(image):
    ...

def data_generator(image_paths, target_height, target_width, batch_size, augmentation_probability=0.2):
    ...
