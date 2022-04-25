import serial

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
import signal
import time

from model import Net
from data_loader import create_single_dataloader, create_dataloader

from lego import Lego



url = "http://192.168.87.146:8080/video"



LEGO = Lego(port="COM6")
LEGO.command("from mindstorms import Motor")
LEGO.command('from mindstorms import DistanceSensor')

LEGO.command("motor_a = Motor('A')")
LEGO.command("motor_a.start(20)")
# Measure the distance between the Distance Sensor and object in centimeters and inches.
LEGO.command("wall_detector = DistanceSensor('B')")
distance = 11
while(distance>10):
    LEGO.command("dist_cm = wall_detector.get_distance_cm()")
# LEGO.test_read(12)

    LEGO.command("print(dist_cm, 'stop')")
    txt = LEGO.read_print('stop\r\n>>>')
# txt[-11]
    txt=txt.split(" stop")[-2].split("\r\n")[-1]
    if not txt == "None":
        distance = int(txt)

LEGO.command("motor_a.stop()") 



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


cam = cv2.VideoCapture(url)
image_size= 128
result, image = cam.read()

cv2.imwrite('WebCamCapture.png',image)
image=read_image('WebCamCapture.png')


Transformations = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size)])
image = torch.cat((torch.split(image,1)[0],torch.split(image,1)[1],torch.split(image,1)[2]),0)/255
image=Transformations(image)

with torch.no_grad():
    

    # inputs, labels = next(iter(test_dataloader))

    # iterate over test data

    img = image
    # label = labels[0]

    predict_img = img[None, :, :, :]
    img.size()

    prediction = saved_model(predict_img)
    predicted_class = int(np.argmax(prediction))

    


#COM 4 = USB

#COM6 = bluetooth

if(predicted_class==0):
    LEGO.command("hub.display.show('0')")
elif(predicted_class==1):
    LEGO.command("hub.display.show('1')")
else:
    print('ERROR')



os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
img = torch.permute(img, (1, 2, 0)) 
plt.imshow(img)
plt.show()
# print(f"Label: {label}")
print(f"Prediction: {predicted_class}")
print(max(prediction))    
# print(predicted_class)

