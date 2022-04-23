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


from model import Net
from data_loader import create_single_dataloader, create_dataloader


# Opening device connection
def getCoordinates(image):
    img=image
    img_h, img_w, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, 4, cv2.CV_32S
    )

    # iterate over components
    best_match, best_dist = None, 1000000
    for i in range(numlabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        # reject if too small
        if h < img_h * 0.1 or w < img_w * 0.1:
            continue

        # reject if too large
        if h > img_h * 0.9 or w > img_w * 0.9:
            continue

        # pick component that is close to center
        dist = (img_h / 2 - cx) * 2 + (img_w / 2 - cy) * 2
        if dist < best_dist:
            best_match = i
            best_dist = dist

    if best_match is not None:
        x = stats[best_match, cv2.CC_STAT_LEFT]
        y = stats[best_match, cv2.CC_STAT_TOP]
        w = stats[best_match, cv2.CC_STAT_WIDTH]
        h = stats[best_match, cv2.CC_STAT_HEIGHT]
        
        
        
        
        
        # cx, cy = centroids[best_match]
        output = img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        
        plt.imshow(output, cmap="gray")
        plt.plot(x,y,'ro') 
        plt.show()
        print(f"x:{x}, y:{y}, w:{w}, h:{h}")
               

        return x,y,w,h








def cropMetrics(x,y,width,height):
    center_x = x+(width/2) 
    center_y = y-(height/2)

    if width >= height:
        y_new = center_y + (width/2)
        return x, y_new, width 

    elif height > width:
        x_new = center_x - (height/2)
        return x_new, y, height 





cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
cam.get(cv2.CAP_PROP_FRAME_WIDTH)
cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

# s = serial.Serial('COM6',9600)
# s.write(b'\x03')
# s.write(b'import hub\r\n')

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



url = "http://192.168.87.146:8080/video"
cam = cv2.VideoCapture(url)

image_size= 128
result, image = cam.read()

# cv2.imshow("frame",image)
cv2.imwrite('WebCamCapture.png',image)
image=cv2.imread('WebCamCapture.png')

# image = torch.permute(image, (1, 2, 0)) 
# plt.imshow(image)



x, y, w, h = getCoordinates(image)
x, y, l = cropMetrics(x, y, w, h)
channels , img_h, img_w = image.shape


image=read_image('WebCamCapture.png')

image = transforms.functional.crop(image,top=int(y),left=int(x),height=int(l),width=int(l))

Transformations = transforms.Compose([transforms.Resize(image_size)])
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

# if(predicted_class==0):
#     s.write(b'hub.display.show("0")\r\n')
# elif(predicted_class==1):
#     s.write(b'hub.display.show("1")\r\n')
# else:
#     print('ERROR')



os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
img = torch.permute(img, (1, 2, 0)) 
plt.imshow(img)
plt.show()
# print(f"Label: {label}")
print(f"Prediction: {predicted_class}")
print(max(prediction))    
# print(predicted_class)
