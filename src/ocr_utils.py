import cv2
import numpy as np
import tensorflow as tf
import os

# fLoad the trained model from disk
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'CustomCnn_model.keras')
    model = tf.keras.models.load_model(model_path)
    return model

# Convert image to grayscale
def convert_2_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Binarize image using Otsu thresholding
def binarization(image):
    img, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return img, thresh

# Dilation to connect components
def dilate(image, words=False):
    img = image.copy()
    m = 3
    n = m - 2
    itrs = 4
    if words:
        m = 6
        n = m
        itrs = 3
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    dilation = cv2.dilate(img, rect_kernel, iterations=itrs)
    return dilation

# Find bounding boxes for characters or words
def find_rect(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    return sorted(rects, key=lambda x: x[0])

# Character mapping for lowercase letters
def get_mapping():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

# Extract characters and run OCR
def extract(image):
    model = load_model()
    mapping, mapping_inverse = get_mapping()
    chars = []

    image_cpy = image.copy()
    _, bin_img = binarization(convert_2_gray(image_cpy))
    full_dil_img = dilate(bin_img, words=True)
    words = find_rect(full_dil_img)

    for word in words:
        x, y, w, h = word
        img = image_cpy[y:y+h, x:x+w]

        _, bin_img = binarization(convert_2_gray(img))
        dil_img = dilate(bin_img)
        char_parts = find_rect(dil_img)

        for char in char_parts:
            cx, cy, cw, ch = char
            ch_img = img[cy:cy+ch, cx:cx+cw]

            white_img = np.full((32, 32, 1), 255, dtype=np.uint8)
            resized = cv2.resize(ch_img, (16, 22), interpolation=cv2.INTER_CUBIC)
            gray = convert_2_gray(resized)
            white_img[3:25, 3:19, 0] = gray
            gray_rgb = cv2.cvtColor(white_img, cv2.COLOR_GRAY2RGB)
            gray_rgb = gray_rgb.astype(np.int32)

            prediction = model.predict(np.array([gray_rgb]), verbose=0)
            predicted = mapping_inverse[np.argmax(prediction)]
            chars.append(predicted)

        chars.append(' ')

    return ''.join(chars[:-1])
