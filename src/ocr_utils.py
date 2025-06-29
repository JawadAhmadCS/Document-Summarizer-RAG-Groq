from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# Load trained model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'CustomCnn_model.keras')
    model = tf.keras.models.load_model(model_path)
    return model

# Convert image to grayscale
def convert_2_gray(image):
    return ImageOps.grayscale(image)

# Binarize image using simple threshold (PIL version of Otsu)
def binarization(image):
    gray_np = np.array(image)
    threshold = gray_np.mean()
    binary = (gray_np < threshold).astype(np.uint8) * 255
    return Image.fromarray(binary), binary

# Dummy dilation (PIL doesn’t have direct morphological ops)
def dilate(image_array, words=False):
    from scipy.ndimage import binary_dilation
    structure = np.ones((3, 2)) if not words else np.ones((6, 6))
    dilated = binary_dilation(image_array, structure=structure).astype(np.uint8) * 255
    return Image.fromarray(dilated)

# Find rectangular bounding boxes
def find_rect(image_array):
    from scipy.ndimage import label, find_objects
    labeled, num = label(image_array > 0)
    boxes = find_objects(labeled)
    rects = []
    for box in boxes:
        if box is not None:
            y1, y2 = box[0].start, box[0].stop
            x1, x2 = box[1].start, box[1].stop
            rects.append((x1, y1, x2 - x1, y2 - y1))
    return sorted(rects, key=lambda x: x[0])

# Mapping for lowercase characters
def get_mapping():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

# Extract characters from image
def extract(pil_image):
    model = load_model()
    mapping, mapping_inverse = get_mapping()
    chars = []

    img_gray = convert_2_gray(pil_image)
    bin_img, bin_arr = binarization(img_gray)
    full_dil_img = dilate(bin_arr, words=True)
    word_boxes = find_rect(np.array(full_dil_img))

    for word in word_boxes:
        x, y, w, h = word
        word_img = pil_image.crop((x, y, x + w, y + h))
        gray_word = convert_2_gray(word_img)
        bin_word, bin_arr = binarization(gray_word)
        dil_word = dilate(bin_arr)
        char_boxes = find_rect(np.array(dil_word))

        for char in char_boxes:
            cx, cy, cw, ch = char
            char_img = gray_word.crop((cx, cy, cx + cw, cy + ch))
            white_bg = Image.new("L", (32, 32), 255)
            char_resized = char_img.resize((16, 22))
            white_bg.paste(char_resized, (3, 3))
            rgb = Image.merge("RGB", (white_bg, white_bg, white_bg))
            gray_rgb = np.array(rgb).astype(np.int32)

            prediction = model.predict(np.array([gray_rgb]), verbose=0)
            predicted = mapping_inverse[np.argmax(prediction)]
            chars.append(predicted)

        chars.append(' ')

    return ''.join(chars[:-1])
