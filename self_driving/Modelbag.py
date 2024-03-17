import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, History
from sklearn.model_selection import train_test_split
import time
import csv
from datetime import datetime

log_path="/home/alyjf10/self_driving/logs/fit/"
loss_img='/home/alyjf10/self_driving/loss_plot.png'

class NN(tf.keras.Model):
    def __init__(self):
        '''
        Init the structure,when you init the structure in main.py,it will plot a summary of structure.
        my initial structure is that
                Input_img
                base_model
            GlobalAveragePooling2D
                Flatten
               Droup out 0.2
                 Dense 50
               Droup out 0.2
        
        Dense 10            Dense 10
        Dense 1(angle)     Dense 1(speed)

        '''
        super(NN, self).__init__()
        self.base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                                      include_top=False,
                                      weights='imagenet',
                                      input_shape=(224, 224, 3),
                                      input_tensor=None)
        '''
        This base_model has 195 layers, I frozen all of them, means will not train the base layer.
        If you want to fine-tune the pre-trained model, you can use the follow code, 
        that means fine-tune the layers which is over 190, others will be frozen.
        for i, layer in enumerate(self.base_model.layers):
            if i > 190:
                layer.trainable = True
            else:
                layer.trainable = False
        '''
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = False
        print('ALL layers: ', len(self.base_model.layers))#print the number of layers in base model.

        self.AveragePooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.Dropout_2 = tf.keras.layers.Dropout(0.2)
        self.Dense = tf.keras.layers.Dense(50,activation='relu')
        self.Dropout_3 = tf.keras.layers.Dropout(0.2)
        #angle branch
        self.angle_branch = tf.keras.Sequential([                              
            tf.keras.layers.Dense(10, activation='relu'),   
            tf.keras.layers.Dense(1, activation='linear')])                                                                                                                                      
        # speed branch
        self.speed_branch = tf.keras.Sequential([             
            tf.keras.layers.Dense(10, activation='relu'), 
            tf.keras.layers.Dense(1, activation='linear') ])
        self.build([None, 224, 224, 3])
        self.summary()

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.AveragePooling(x)
        x = self.flatten(x)
        x = self.Dropout_2(x)
        x = self.Dense(x)
        x = self.Dropout_3(x)
        output_1 = self.angle_branch(x)
        output_2 = self.speed_branch(x)   
        return output_1, output_2
    #Read img and preprocess
    def preprocess_image(self, image_path, augment=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        if augment:
            image = tf.image.random_brightness(image, max_delta=0.1)  # 随机亮度调整
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1) 
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  

        image = tf.image.resize(image, [224, 224]) / 255.0
        print("Data Process!")
        return image

    def load_data(self, root_path, img_path, csv_path,batch_size=16):
        # load csv
        labels_df = pd.read_csv(os.path.join(root_path, csv_path))

        # path process
        labels_df['image_path'] = labels_df['image_id'].apply(lambda x: os.path.join(root_path, img_path, f"{x}.png"))
        labels_df = labels_df[labels_df['image_path'].apply(os.path.exists)]
        # split into train dataset and validation dataset
        train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=6)
        # to tensor type
        train_img_dataset = tf.data.Dataset.from_tensor_slices((
            train_df['image_path'].values
        ))
        val_img_dataset = tf.data.Dataset.from_tensor_slices((
            val_df['image_path'].values
        ))
        train_label_dataset = tf.data.Dataset.from_tensor_slices((
            train_df['angle'].values,
            train_df['speed'].values
        ))
        val_label_dataset = tf.data.Dataset.from_tensor_slices((
            val_df['angle'].values,
            val_df['speed'].values
        ))

        
        train_img_dataset = train_img_dataset.map(lambda x: self.preprocess_image(x, augment=True),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        val_img_dataset = val_img_dataset.map(lambda x: self.preprocess_image(x, augment=False),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #zip img and label
        train_dataset = tf.data.Dataset.zip((train_img_dataset,train_label_dataset))
        val_dataset = tf.data.Dataset.zip((val_img_dataset, val_label_dataset))
        return train_dataset.batch(batch_size), val_dataset.batch(batch_size)

    def training(self, train_dataset, val_dataset, epochs, trained_model,model_save_path):
        if trained_model != None:
            self.load_weights(trained_model)        
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={'output_1': 'mse', 'output_2': 'mse'},
            metrics={'output_1': 'mse', 'output_2': 'mse'})        
        self.build([None, 224, 224, 3])
        
        log_dir = log_path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',save_best_only=True)
        earlystopping_callback = EarlyStopping(patience=60,restore_best_weights=True) 
        history_callback = History()
        start = time.time()
        history = self.fit(train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        callbacks=[checkpoint_callback, tensorboard_callback, earlystopping_callback,history_callback])

        end = time.time()
        print('Time: {:.2f} minutes'.format((end - start) / 60))
        
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
        plt.savefig(loss_img)
        plt.show()

    def predict_model(self, trained_model_path, image_path, output_path):
        # 加载已经训练好的模型
        self.load_weights(trained_model_path)


        # 打开 CSV 文件，准备写入数据
        with open(output_path, 'w', newline='') as csvfile:
            # 定义 CSV 写入器
            writer = csv.writer(csvfile)
            # 写入 CSV 表头
            writer.writerow(['image_id','angle','speed'])

            # 遍历文件夹中的图片，对每张图片进行预测
            for image_name in sorted(os.listdir(image_path)):
                img_path = os.path.join(image_path, image_name)
                # 加载并预处理图片
                # 读取图片文件并调整大小
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
                # 将图片转换为 NumPy 数组并缩放像素值到 [0, 1] 区间
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                # 将图片数组调整为模型的输入格式
                img_input = tf.expand_dims(img_array, axis=0)

                # 使用训练好的模型进行预测
                prediction = self.predict(img_input)
                
                # 对预测结果进行处理
                angle = prediction[0][0][0]  # 从二维数组中提取值
                speed = prediction[1][0][0] 

                
                # 四舍五入到合适的精度
                angle = np.round(angle, 4)
                speed = np.round(speed, 2)
                image_id = image_name.split('.')[0]

                writer.writerow([image_id, angle, speed])
        print('Finish!')


        
    

class Customer(tf.keras.Model):
    def __init__(self):
        super(Customer, self).__init__()
        self.conv_layers = [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
 
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten()
        ]
        
        self.dense_layers = [
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
        ]
        self.out1 = tf.keras.layers.Dense(1,activation='linear')
        self.out2 = tf.keras.layers.Dense(1,activation='linear')
    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        
        output_1 = self.out1(x)
        output_2 = self.out2(x)
                
        return output_1, output_2
              

    def training(self, train_dataset, val_dataset, epochs, batch_size, model_save_path,load_model=None):
        if load_model != None:
            print('Loda trained model!')
            model = tf.keras.models.load_model(load_model)
        self.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'output_1': 'mse', 'output_2': 'mse'},
              metrics={'output_1': 'mse', 'output_2': 'mse'}) 
        self.build([None, 128, 128, 3])
        self.summary()


        log_dir = "/home/alyjf10/self_driving/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',save_best_only=True)
        earlystopping_callback = EarlyStopping(patience=50) 
        history_callback = History()
        start = time.time()
        history = self.fit(train_dataset.batch(batch_size),
                          epochs=epochs,
                          validation_data=val_dataset.batch(batch_size),
                          callbacks=[checkpoint_callback, tensorboard_callback, earlystopping_callback,history_callback])

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
        plt.savefig('/home/alyjf10/self_driving/loss_plot.png')#if wang to save, you need to change the path
 

 
 
