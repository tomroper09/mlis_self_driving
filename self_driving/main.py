from data import Data, data_preprocess
import Modelbag
import os
import tensorflow as tf
#if you don't have this library, use this code
#!pip freeze > requirements.txt
#mean use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0' 
#load the dataset and split into train dataset and vals dataset,you need to change the path
trains, vals = Data(root_path='/home/alyjf10/self_driving/data/',img_path='training_data/training_data',csv_path='training_norm.csv')
#construct the model(resnet)
model = Modelbag.Restnet()
#train the model [model.training(train data, val data, epoch num, batch size, path to save model,load the trained model for continus training(if not, it should be None))]
model.training(trains,vals,epochs=250,batch_size=32,model_save_path='/home/alyjf10/self_driving/model/resnet_500',load_model='/home/alyjf10/self_driving/model/resnet_lastest')
#after training, can use this one to predict
#Modelbag.predict_model(model_path='/home/alyjf10/self_driving/model/resnet', image_path='/home/alyjf10/self_driving/data/test_data/test_data',output_path = '/home/alyjf10/self_driving/data/test_data/predict.csv')

