import tensorflow as tf
import numpy as np
import os


class Model:

    saved_model = 'DenseNet169'

    def __init__(self):
        self.model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
        self.model.summary()


    def preprocess(self, image):
        img = tf.keras.preprocessing.image.load_img(image, target_size=(224,224))
        # 将图片转换为 NumPy 数组并缩放像素值到 [0, 1] 区间
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        # 将图片数组调整为模型的输入格式
        image = tf.expand_dims(img_array, axis=0)
        return image

    def predict(self, image):
        image = self.preprocess(image)
        prediction = self.model.predict(image)
        # Training data was normalised so convert back to car units
        angle = prediction[0][0][0]  # 从二维数组中提取值
        speed = prediction[1][0][0]
        if speed >= 0.6:
                    speed = 1
        elif speed <= 0.4:
                    speed = 0
        else:
                    speed = speed
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed

