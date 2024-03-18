import pandas as pd
import numpy as np
import cv2

# read csv file
data = pd.read_csv("/home/alyjf10/self_driving/data/combined_training_data.csv")

# match angle and speed
angle = data['angle']
speed = data['speed']


# count 
angle_label_counts = data.groupby('angle').size().reset_index(name='count')
speed_label_counts = data.groupby('speed').size().reset_index(name='count')
# angle=0.5
angle_labels = [0.5]

# find all of speed=0 and angle=0.5 
selected_data = data[(data['speed'] == 0) & (data['angle'].isin(angle_labels))]

# get the image_id with speed=0 and angle=0.5 
selected_image_ids = selected_data['image_id']

image_folder = "/home/alyjf10/self_driving/data/training_data/training_data/"
new_data=[]
i = len(data['image_id'])

for index, row in selected_data.iterrows():
   
    image_path = image_folder + str(selected_image_ids[index]) + '.png'

    image = cv2.imread(image_path)
    if image is None:
        print("fali to read img:", image_path)
        continue
    
    # flip
    flipped_image = cv2.flip(image, 1)


    new_angle = 1 - row['angle']
    speed = row['speed']

    cv2.imwrite('/home/alyjf10/self_driving/data/training_data/training_data/'+ str(i) + '.png', flipped_image)
    new_data.append({'image_id': str(i) , 'angle': new_angle, 'speed': speed})  
    i += 1

#add new data to original data
new_data_df = pd.DataFrame(new_data)
data = pd.concat([data, new_data_df], ignore_index=True)

# save file
data.to_csv("/home/alyjf10/self_driving/data/combined_training_data.csv",index=False)
