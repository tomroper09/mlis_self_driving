import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, History
import time
import csv
from data import Data
from datetime import datetime


class Restnet(tf.keras.Model):
    def __init__(self):
        super(Restnet, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(150, 150, 3))
        #self.base_model = tf.keras.applications.efficientnet.EfficientNetB0(
        #                                                include_top=False,
        #                                                weights='imagenet',
        #                                                input_tensor=None,
        #                                                input_shape=None,
        #                                                pooling=None,)
        self.base_model = tf.keras.applications.resnet50.ResNet50(
                                      include_top=False,
                                      weights='imagenet',
                                      input_tensor=None)
        self.base_model.trainable = False
        
        #for layer in self.base_model.layers[-3:]:
        #    layer.trainable = True
        print('ALL layers: ', len(self.base_model.layers))
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        #self.dense = tf.keras.layers.Dense(500, activation='relu')
        #self.dropout = tf.keras.layers.Dropout(0.2)
        #angle branch
        self.angle_branch = tf.keras.Sequential([
            tf.keras.layers.Dense(250, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # speed branch
        self.speed_branch = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_average_layer(x)
        #x = self.dense(x)
        #x = self.dropout(x)
        
        output_1 = self.angle_branch(x)
        output_2 = self.speed_branch(x)
        
        return output_1, output_2


    def training(self, train_dataset, val_dataset, epochs, batch_size, model_save_path,load_model=None):
        if load_model != None:
            print('Loda trained model!')
            model = tf.keras.models.load_model(load_model)
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={'output_1': 'mse', 'output_2': 'mse'},
              metrics={'output_1': 'mse', 'output_2': 'mse'}) 
        self.build([None, 150, 150, 3])
        self.summary()


        log_dir = "/home/alyjf10/self_driving/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',save_best_only=True)
        #earlystopping_callback = EarlyStopping(patience=30) 
        history_callback = History()
        start = time.time()
        history = self.fit(train_dataset.batch(batch_size),
                          epochs=epochs,
                          validation_data=val_dataset.batch(batch_size),
                          callbacks=[checkpoint_callback, tensorboard_callback, history_callback])

        self.save(model_save_path)
        end = time.time()
        print('Time: {:.2f} minutes'.format((end - start) / 60))
        # plot the figure of loss and mse
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(history.history['output_1_loss'], label='Training Angle Loss')
        plt.plot(history.history['val_output_1_loss'], label='Validation Angle Loss')
        plt.title('Angle Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history.history['output_2_loss'], label='Training Speed Loss')
        plt.plot(history.history['val_output_2_loss'], label='Validation Speed Loss')
        plt.title('Speed Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        #plt.savefig('/home/alyjf10/self_driving/loss_plot.png')#if wang to save, you need to change the path
        plt.show()

def predict_model(model_path, image_path, output_path):
    
    model = tf.keras.models.load_model(model_path)
    
 
    with open(output_path, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(['image_id','angle','speed'])

        for image_name in sorted(os.listdir(image_path)):
            img_path = os.path.join(image_path, image_name)

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150,150))
           
            img_array = tf.keras.preprocessing.image.img_to_array(img)
           
            img_array = img_array / 255.0
          
            img_input = tf.expand_dims(img_array, axis=0)
            
          
            angle, speed = model.predict(img_input)
           
            angle = angle[0][0]
            if speed >= 0.5:
                speed = 1
            else:
                speed = 0
            image_id = image_name.split('.')[0]

            
            writer.writerow([image_id, angle, speed])
    print('Finish!')


        
    


    


