import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, History
from sklearn.model_selection import train_test_split,KFold
import time
import csv
from datetime import datetime

log_path="/home/alyjf10/self_driving_car/logs/fit/"
loss_img='/home/alyjf10/self_driving_car/loss_plot.png'
output_file = '/home/alyjf10/self_driving_car/model/Kfold/'

class PilotNet(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3)):
        super(PilotNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(36, (3, 3), strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(48, (3, 3), strides=(2, 2), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(100, activation='relu')
        self.fc2 = tf.keras.layers.Dense(50, activation='relu')
        self.fc3 = tf.keras.layers.Dense(10, activation='relu')
        self.fc4 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.fc4(x)
        return output
class SENet(tf.keras.Model):
    def __init__(self, channels, reduction_ratio=16):
        super(SENet, self).__init__()
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.global_pooling(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        return x

class NN(tf.keras.Model):
    def __init__(self,model_name):
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
        print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model: ',model_name)
        if model_name == 'ResNet50V2':
            self.base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                                        include_top=False,
                                        weights='imagenet',
                                        input_shape=(224, 224, 3),
                                        input_tensor=None)
        elif model_name == 'ResNet101V2':
            self.base_model=tf.keras.applications.resnet_v2.ResNet101V2(
                                        include_top=False,
                                        weights='imagenet',
                                        input_shape=(224, 224, 3)

                                    )
        elif model_name == 'ResNet50':
            self.base_model = tf.keras.applications.resnet50.ResNet50(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224, 224, 3)
                                    )
        elif model_name == 'VGG19':
            self.base_model = tf.keras.applications.vgg19.VGG19(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224, 224, 3)
                                    )
        elif model_name == 'ResNet152V2':
            self.base_model = tf.keras.applications.resnet_v2.ResNet152V2(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224, 224, 3)
                                    )
        elif model_name == 'VGG16':
            self.base_model = tf.keras.applications.vgg16.VGG16(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224, 224, 3))
        elif model_name == 'DenseNet201':
            self.base_model = tf.keras.applications.densenet.DenseNet201(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3),
                                    )
        elif model_name == 'DenseNet169':
            self.base_model = tf.keras.applications.densenet.DenseNet169(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3),
                                    )
        elif model_name == 'DenseNet121':
            self.base_model = tf.keras.applications.densenet.DenseNet121(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3),
                                    )
        elif model_name == 'EfficientNetV2S':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3))
        elif model_name == 'EfficientNetV2M':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3))
        elif model_name == 'EfficientNetV2L':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(224,224,3))

        
        else:
            raise ValueError("Invalid model_name. Supported options are 'ResNet50V2'.")


            
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
            layer.trainable = True



        print('ALL layers: ', len(self.base_model.layers))#print the number of layers in base model.
        self.conv_reduce_channels = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.bn_reduce_channels = tf.keras.layers.BatchNormalization()
        self.relu_reduce_channels = tf.keras.layers.Activation('relu')
        self.conv_reduce_channels2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        #self.AveragePooling = tf.keras.layers.GlobalAveragePooling2D()
        #self.flatten = tf.keras.layers.Flatten()
        #self.Dropout_2 = tf.keras.layers.Dropout(0.5)
        #angle branch
        self.angle_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),                   
            tf.keras.layers.Dense(1, activation='linear')])                                                                                                                                      
        # speed branch
        self.speed_branch = tf.keras.Sequential([ 
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),          
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dropout(0.3),                    
            tf.keras.layers.Dense(1, activation='linear')])
        self.build([None, 224, 224, 3])
        self.summary()

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.conv_reduce_channels(x)
        x = self.bn_reduce_channels(x)
        x = self.relu_reduce_channels(x)
        x = self.conv_reduce_channels2(x)
        output_1 = self.angle_branch(x)
        output_2 = self.speed_branch(x)
        return output_1, output_2

    #Read img and preprocess
    def preprocess_image(self, image_path, augment=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        if augment:
            image = tf.image.random_brightness(image, max_delta=0.2)  # 随机亮度调整
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  
            #image = tf.image.random_crop(image,(220,220,3))

        image = tf.image.resize(image, [224, 224]) / 255.0
        print("Data Process!")
        return image

    def load_data(self, root_path, img_path, csv_path,flod_number ):
        # load csv
        train_img = []
        train_label = []
        val_img = []
        val_label = []
        labels_df = pd.read_csv(os.path.join(root_path, csv_path))

        # path process
        labels_df['image_path'] = labels_df['image_id'].apply(lambda x: os.path.join(root_path, img_path, f"{x}.png"))
        labels_df = labels_df[labels_df['image_path'].apply(os.path.exists)]
        print('Amount of path: ',len(labels_df))
        # split into train dataset and validation dataset

        # Split data into two parts based on index
        #flip_data = labels_df[labels_df.index > 13794]
        #train_df, val_df = train_test_split(original_datasets, test_size=0.2, random_state=6)


        kfold = KFold(n_splits=flod_number, shuffle=True, random_state=42)
        for train_indices, val_indices in kfold.split(labels_df):
            train_df = labels_df.iloc[train_indices]
            val_df = labels_df.iloc[val_indices]
            # Combine remaining data with training set
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

            train_img.append(train_img_dataset)
            train_label.append(train_label_dataset)
            val_img.append(val_img_dataset)
            val_label.append(val_label_dataset)


        return train_img,train_label,val_img,val_label




        


    def training(self,train_img,train_label,val_img,val_label, epochs,batch_size, trained_model,model_save_path):
        if trained_model != None:
            self.load_weights(trained_model)        
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={'output_1': 'mse', 'output_2': 'mse'},#binary_crossentropy
            metrics=['mse'])        
        self.build([None, 224, 224, 3])

        log_dir = log_path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',save_best_only=True)
        earlystopping_callback = EarlyStopping(patience=10,restore_best_weights=True) 
        history_callback = History()
        fold_metrics = []  # 用于存储每个 fold 的指标结果

        for i in range(len(train_img)):
            print(f"**********************************************************************FOLD{i+1}**********************************************************************************")
            train_img_dataset = train_img[i]
            train_label_dataset = train_label[i]
            val_img_dataset = val_img[i]
            val_label_dataset = val_label[i]

            train_img_dataset = train_img_dataset.map(lambda x: self.preprocess_image(x, augment=True),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_img_dataset = val_img_dataset.map(lambda x: self.preprocess_image(x, augment=False),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #zip img and label
            train_dataset = tf.data.Dataset.zip((train_img_dataset,train_label_dataset))
            val_dataset = tf.data.Dataset.zip((val_img_dataset, val_label_dataset))
            train_dataset = train_dataset.shuffle(buffer_size=len(train_img_dataset)*2) 
            train_dataset=train_dataset.batch(batch_size)
            val_dataset=val_dataset.batch(batch_size)


            start = time.time()


            self.fit(train_dataset,
                            epochs=epochs,
                            validation_data=val_dataset,
                            validation_freq = 2,
                            callbacks=[checkpoint_callback, tensorboard_callback, earlystopping_callback,history_callback])

            val_loss, output_1_loss, output_2_loss, output_1_mse, output_2_mse = self.evaluate(val_dataset)

            fold_metrics.append({'val_mse': val_loss,
                             'Angle': output_1_mse,
                             'Speed': output_2_mse})
            

            end = time.time()
            #self.save(model_save_path, save_format='tf')
            print('Time: {:.2f} minutes'.format((end - start) / 60))
            print(f"Fold {i+1} Metrics -  Validation MSE: val_mse:{val_loss},angle:{output_1_mse},speed:{output_2_mse}")

        

    def predict_model(self, trained_model_path, image_path, output_path):
        # 加载已经训练好的模型
        self.load_weights(trained_model_path)
        label_interval = 0.0625
        num_labels = 17
        # 标签的最小值
        min_label = 0.0
        
        # 预测值所在的标签范围
        label_range = [min_label + i * label_interval for i in range(num_labels)]

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
                if speed >= 0.6:
                    speed = 1
                elif speed <= 0.4:
                    speed = 0
                else:
                    speed = speed


                # 计算预测值与每个标签的差值的绝对值
                #differences = [abs(prediction - label) for label in label_range]
                #
                ## 找到最小差值对应的标签值
                #predicted_label_value = label_range[differences.index(min(differences))]
                #angle = predicted_label_value

                
                # 四舍五入到合适的精度
                angle = np.round(angle, 6)
                speed = np.round(speed, 6)
                image_id = image_name.split('.')[0]

                writer.writerow([image_id, angle, speed])
        print('Finish!')







        
 
 
