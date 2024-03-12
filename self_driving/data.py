import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def Data(root_path, img_path, csv_path):

    img = os.path.join(root_path, img_path)
    labels_df = pd.read_csv(os.path.join(root_path, csv_path))

    
    image_data = []
    for image_id in labels_df['image_id']:
        image_path = f'{img}/{image_id}.png'
        if os.path.exists(image_path):
            image = load_img(image_path, target_size=(150, 150))
            image_array = img_to_array(image)
            image_data.append(image_array)
        else:
            print(f"Warning: Image {image_path} not found, skipping...")

    image_data = np.array(image_data)

   #divided into train dataset and validation dataset with 7:3
    X_train, X_val, y_train_angle, y_val_angle, y_train_speed, y_val_speed = train_test_split(
        image_data, labels_df['angle'], labels_df['speed'], test_size=0.3, random_state=42)

  
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_angle, y_train_speed))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_angle, y_val_speed))

  
    def preprocess_image(image, label_angle, label_speed):
  
        image = tf.image.random_brightness(image, max_delta=0.2) 
        #image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  
        image = tf.image.random_crop(image, size=[140, 140, 3])  #
        #image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  

 
        image = tf.image.resize(image, [150, 150]) / 255.0 
        return image, label_angle * 80 + 50, label_speed * 35

    train_dataset = train_dataset.map(preprocess_image)
    val_dataset = val_dataset.map(preprocess_image)

    
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    val_dataset = val_dataset

    return train_dataset, val_dataset

def pre_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   
    _, lane_mask = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY)

    
    _, object_mask = cv2.threshold(gray, 100, 150, cv2.THRESH_BINARY)

    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    
    result = cv2.merge([lane_mask, object_mask, gray])

    return result

def data_preprocess(input_folder, output_folder):
  
    os.makedirs(output_folder, exist_ok=True)

    
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

  
    for image_file in image_files:
      
        input_path = os.path.join(input_folder, image_file)
        
   
        image = cv2.imread(input_path)
        
     
        if image is None:
            print(f"fali to load img: {input_path}")
            continue

       
        result = pre_image(image)
        
        
        # 构造输出文件路径
        output_path = os.path.join(output_folder, image_file)
        
        # 保存灰度图像
        cv2.imwrite(output_path, result)

    print("Finish!")
