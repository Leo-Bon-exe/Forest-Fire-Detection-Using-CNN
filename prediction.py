import tensorflow as tf
import numpy as np
import cv2
import os

data_dir = "dataset"
img_size = 128
categories = os.listdir(data_dir)

loaded_model = tf.keras.models.load_model("image_classification_model.keras")


def load_and_predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = loaded_model.predict(img)

    prediction_value = prediction[0][0]

    if prediction_value >= 0.5:
        predicted_class = categories[1]
    else:
        predicted_class = categories[0]

    print(f"Tahmin edilen sınıf: {predicted_class} ({prediction_value})")
    return predicted_class


load_and_predict_image("test-image.jpg")
