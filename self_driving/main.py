import Modelbag_copy
import os



#choose gpu
os.environ['CUDA_VISIBLE_DEVICES'] ='1' 
#
root_path = "/home/alyjf10/self_driving_car/"                          #root path
img_path = "training_data/training_data"                                #train dataset name
csv_path = "combined_training_data.csv"                                          #train dataset csv name
test_data = '/home/alyjf10/self_driving_car/data/test_data/test_data'       #test data path
test_data_csv = '/home/alyjf10/self_driving_car/data/test_data/test.csv' #where to store the prediction result
#init Model
all_model = {'DenseNet169'}#VGG16,'DenseNet121','ResNetRS50','ResNetRS101','ResNetRS152'}#{'ResNet50','ResNet50V2','ResNet101V2','ResNet152V2','VGG19','VGG16','DenseNet121','DenseNet169','DenseNet201'}
#best 'DenseNet169' 'DenseNet121'
mse_results = {} 
for name in all_model:
    Model = Modelbag_copy.NN(model_name=name)
    #loda data
    #model save path must be the .h5 type!
    path = os.path.join('/home/alyjf10/self_driving_car/model',name)
    train_data,val_data = Model.load_data( root_path, img_path, csv_path,batch_size=32)
    Model.training(train_data,val_data, epochs=500, trained_model=None,model_save_path=path)
#Model.predict_model(trained_model_path='/home/alyjf10/self_driving/model/norm_1.h5',image_path=test_data , output_path=test_data_csv)

#after base_model

#训练三个模型，选出最好的一个，然后再将原始数据集全部给到训练一边
