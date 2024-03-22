import Modelbag
import os
import json


#choose gpu
os.environ['CUDA_VISIBLE_DEVICES'] ='1' 
#
root_path = "/home/alyjf10/self_driving/data"                          #root path
img_path = "training_data/training_data"                                #train dataset name
csv_path = "combined_training_data.csv"                                          #train dataset csv name
test_data = '/home/alyjf10/self_driving/data/test_data/test_data'       #test data path
test_data_csv = '/home/alyjf10/self_driving/data/test_data/test.csv' #where to store the prediction result
#init Model
all_model = {'DenseNet169','VGG16'}#VGG16,'DenseNet121','ResNetRS50','ResNetRS101','ResNetRS152'}#{'ResNet50','ResNet50V2','ResNet101V2','ResNet152V2','VGG19','VGG16','DenseNet121','DenseNet169','DenseNet201'}
#best 'DenseNet169' 'DenseNet121'
mse_results = {} 
for name in all_model:
    Model = Modelbag.NN(model_name=name)
    #loda data
    train_dataset, val_dataset = Model.load_data( root_path, img_path, csv_path,batch_size=16)
    #model save path must be the .h5 type!
    path = os.path.join('/home/alyjf10/self_driving/model/',name)
    history = Model.training(train_dataset, val_dataset, epochs=500, trained_model=None, model_save_path=path)

#print('Finish!')
#Model = Modelbag.NN(model_name='DenseNet169')
#Model.predict_model(trained_model_path='/home/alyjf10/self_driving/model/DenseNet169_0.0120/',image_path=test_data , output_path=test_data_csv)

