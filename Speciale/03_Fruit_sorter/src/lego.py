import serial
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from model import Net
from data_loader import create_single_dataloader, create_dataloader
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

class Lego():
    def __init__(self,port="COM6"):

        self.ser = serial.Serial(port,9600)
        self.ser.write(b'\x03')

    def command(self, py_command):
        py_command = py_command + '\r\n'
        py_command = py_command.encode('utf-8')
        self.ser.write(py_command)

    def read_print(self,stop_string):
        return self.ser.read_until(stop_string.encode('utf-8')).decode('utf-8')

    def read_from_hub_terminal(self,num_char):
        return self.ser.read(num_char)

    def read_sensor_data(self, sensor, None_value, print_it=False):
        self.command("print(" + sensor + ", 'stop')")
        txt = self.read_print('stop\r\n>>>')
        txt=txt.split(" stop")[-2].split("\r\n")[-1]
        if print_it: print(sensor + " value: " + txt)
        if not txt == "None":
            return int(txt)
        else:
            return None_value

    def pop_queue(self,push_queue,distance_push, time_saved):
        time_diff = int((datetime.now() - time_saved).seconds)
        if(distance_push<10 and len(push_queue)>=1 and time_diff > 1):
            if(push_queue[0]==1):
                self.command("push_motor.run_for_degrees(620, 100)")
                self.command("push_motor.run_for_degrees(-620, 100)")
            push_queue.pop(0)
            return datetime.now()
        return time_saved

def take_picture(url):
    cam = cv2.VideoCapture(url)
    result, image = cam.read()
    timestamp =datetime.now().strftime('%Y-%m-%d_%H%M%S')
    data_path = config['files']['folder_path'] + 'data/generated_data/setup_images/image'+timestamp + '.png'
    cv2.imwrite(data_path,image)
    return image


def largest_connected_component(image):
    img_h, img_w, channels = image.shape

    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # blur = cv2.GaussianBlur(image, (7, 7), 0)
    # thresh = cv2.inRange(blur, (0, 0, 0), (255, 255, 200))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    (T, thresh) = cv2.threshold(blurred, 253, 255, cv2.THRESH_BINARY_INV)
    
    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

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
        cx, cy = centroids[best_match]
        output = image.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        plt.imshow(output, cmap="gray"); plt.plot(x,y,'ro'); plt.show() ; print(f"x:{x}, y:{y}, w:{w}, h:{h}")

        if(w>=h):
            y=y-(w-h)/2
            if y<0:y=0
            return x,y,w
        elif(h>w):
            x=x-(h-w)/2
            if x<0:x=0
            return x, y, h


def process_image(image,background, threshold):
    image_size= 128
    segmentor = SelfiSegmentation()


    img_Out = segmentor.removeBG(image, (255,255,255), threshold=threshold)
    cv2.imwrite('WebCamCapture_R_BG.png',img_Out)
    x,y,l =largest_connected_component(img_Out)
    if(background=='default'):
        cv2.imwrite('WebCamCapture.png',image)
    elif(background=='black'):
        img = segmentor.removeBG(image, (0,0,0), threshold=threshold)
        cv2.imwrite('WebCamCapture.png',img)
    elif(background=='white'):
        cv2.imwrite('WebCamCapture.png',img_Out)
    else:
        print('invalid input')

    image=read_image('WebCamCapture.png')
    image = transforms.functional.crop(image,top=int(y),left=int(x),height=int(l),width=int(l))
    Transformations = transforms.Compose([transforms.Resize(image_size)])
    image = torch.cat((torch.split(image,1)[0],torch.split(image,1)[1],torch.split(image,1)[2]),0)/255
    image=Transformations(image)
    return image




def init_model():
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
    return saved_model

def predict_image(saved_model,image):
    with torch.no_grad():

        img = image
        predict_img = img[None, :, :, :]
        img.size()

        prediction_values = saved_model(predict_img)
        predicted_class = int(np.argmax(prediction_values))
        return predicted_class, prediction_values

def print_to_terminal(image,predicted_class,prediction_values):
    #Printing results for extra checs
    os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
    img = torch.permute(image, (1, 2, 0))
    plt.imshow(img)
    plt.show()
    # print(f"Label: {label}")
    print(f"Prediction: {predicted_class}")
    print(max(prediction_values))






