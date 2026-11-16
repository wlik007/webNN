from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input, decode_predictions)
from PIL import Image
import numpy as np
 
IMG_SIZE = (224, 224)  # розмір входу EfficientNetB0
 
def load_model():
    # Попередньо навчений ImageNet-класифікатор (1000 класів)
    return EfficientNetB0(weights="imagenet")
 
def prepare_image(image: Image.Image):
    image = image.convert("RGB")   # RGBA/сірі -> 3 канали
    image = image.resize(IMG_SIZE)
    array = keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)  # (1, 224, 224, 3)
    return preprocess_input(array)
 
def predict(model, image: Image.Image, top=3):
    processed = prepare_image(image)
    preds = model.predict(processed, verbose=0)
    return decode_predictions(preds, top=top)[0]  # top-3 класи

