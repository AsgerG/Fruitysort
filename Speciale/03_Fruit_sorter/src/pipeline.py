import serial

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


from model import Net
from data_loader import create_single_dataloader, create_dataloader


# Opening device connection
s = serial.Serial('COM6',9600)
s.write(b'\x03')
s.write(b'import hub\r\n')

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
default_device = 'cpu' #config['training']['default_device']

folder_path = config['files']['folder_path']

#model_name = 'categorical_model_2022-03-02_1844_6_128_64/version_2022-03-02_1844_61'
model_name = 'binary_model_2022-03-17_1047_2_128_64/version_2022-03-17_1047_294' # model_folder/model_version
model_path = folder_path + 'models/' + model_name


# Init fields   - should match tested model
model_data = model_name.split("_")

image_size = int(model_data[5])
batch_size = int(model_data[6].split("/")[0])
num_classes = int(model_data[4])
csv_tag = model_data[0]

csv_train_file = 'data_csv/train_' + csv_tag + '.csv'
csv_test_file = 'data_csv/test_' + csv_tag +'.csv'

# Load model
saved_model = Net(image_size=image_size, num_classes=num_classes)



saved_model.load_state_dict(torch.load(model_path))


saved_model = saved_model.to(default_device)

_ , test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)

saved_model.eval()
with torch.no_grad():
    

    inputs, labels = next(iter(test_dataloader))

    # iterate over test data

    img = inputs[0].squeeze()
    label = labels[0]

    predict_img = img[None, :, :, :]
    img.size()

    prediction = saved_model(predict_img)
    predicted_class = int(np.argmax(prediction))

    


#COM 4 = USB

#COM6 = bluetooth

if(predicted_class==0):
    s.write(b'hub.display.show("0")\r\n')
elif(predicted_class==1):
    s.write(b'hub.display.show("1")\r\n')
else:
    print('ERROR')



os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
img = torch.permute(img, (1, 2, 0)) 
plt.imshow(img)
plt.show()
print(f"Label: {label}")
print(f"Prediction: {predicted_class}")
print(max(prediction))    
# print(predicted_class)
