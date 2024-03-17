import Modelbag
import os


#choose gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
#
root_path = "/home/alyjf10/self_driving/data/"                          #root path
img_path = "training_data/training_data"                                #train dataset name
csv_path = "training_norm.csv"                                          #train dataset csv name
test_data = '/home/alyjf10/self_driving/data/test_data/test_data'       #test data path
test_data_csv = '/home/alyjf10/self_driving/data/test_data/test.csv' #where to store the prediction result
#init Model
Model = Modelbag.NN()
#loda data
train_dataset, val_dataset = Model.load_data( root_path, img_path, csv_path,batch_size=16)
#model save path must be the .h5 type!
Model.training(train_dataset,val_dataset,epochs=2,trained_model=None,model_save_path='/home/alyjf10/self_driving/model/test.h5')
#Model.predict_model(trained_model_path='/home/alyjf10/self_driving/model/test.h5',image_path=test_data , output_path=test_data_csv)

