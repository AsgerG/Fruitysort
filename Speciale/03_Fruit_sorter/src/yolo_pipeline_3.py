from datetime import datetime
import time
from lego import Lego, predict_image, take_picture, init_model, print_to_terminal ,process_image
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#load model
path_to_yolov5 = r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\yolov5"
path_to_custom_model = r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\yolov5\runs\train\exp4\weights\best.pt"
model = torch.hub.load(path_to_yolov5, 'custom', path=path_to_custom_model, source='local')


# url = "http://192.168.1.5:8080/video"
url = "http://192.168.1.164:8080/video"
LEGO = Lego(port="COM5")

LEGO.command("from mindstorms import Motor")
LEGO.command('from mindstorms import DistanceSensor')
LEGO.command('from mindstorms.control import wait_for_seconds')
LEGO.command("conveyor_motor = Motor('B')")
LEGO.command("image_detector = DistanceSensor('D')")
LEGO.command("push_detector = DistanceSensor('A')")
LEGO.command("push_motor = Motor('C')")

#Start Conveyorbelt process
LEGO.command("conveyor_motor.start(20)")

running_pred = np.array([])
running_cord_thres_x_min = []

running_pred_2 = np.array([])
running_cord_thres_x_min_2 = []

# Run forever
while(True):

    #Take picture    
    image = take_picture(url)
    # image  = process_image(image, 'default', 0.9)
    
    #Prediction
    results=model(image)
    # results = model(r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\WebCamCapture.png")
    # results.xyxyn[0].cpu()
    labels, cord_thres = results.xyxyn[0].cpu()[:, -1].numpy(), results.xyxyn[0].cpu()[:, :-1].numpy()
    # results.save(save_dir="C:/Users/andri/OneDrive/Documents/DTU/Fruitysort/Speciale/03_Fruit_sorter/src/")
    
    ## Reaction
    if len(labels) == 1:

        if not len(running_pred_2) == 0:
            print("flushing")
            running_pred_ = running_pred_2
            running_pred_2 = np.array([])

        running_cord_thres_x_min.append(cord_thres[0][0])
        #print(results.pandas().xyxy[0])
        predicted_class = results.pandas().xyxy[0]['name'][0]
        
        if(predicted_class=='fresh_oranges' or predicted_class=='fresh_apples'):
            # LEGO.command("hub.display.show('0')")
            running_pred = np.append(running_pred,0)
        elif(predicted_class=='rotten_apples' or predicted_class=='rotten_oranges'):
            # LEGO.command("hub.display.show('1')")
            running_pred = np.append(running_pred,1)
        
        if running_cord_thres_x_min[-1] > 0.8:
            if running_pred.mean() > 0.5:
                #push - can push twice?
                LEGO.command("push_motor.run_for_degrees(620, 100)")
                LEGO.command("push_motor.run_for_degrees(-620, 100)")
            print(f"prediction value: {running_pred.mean()}")
            running_pred = np.array([])

    elif len(labels) == 2:
        print("two")
        running_cord_thres_x_min.append(cord_thres[1][0])
        running_cord_thres_x_min_2.append(cord_thres[0][0])
        #print(results.pandas().xyxy[0])
        predicted_class = results.pandas().xyxy[0]['name'][1]
        predicted_class_2 = results.pandas().xyxy[0]['name'][0]

        if(predicted_class=='fresh_oranges' or predicted_class=='fresh_apples'):
            # LEGO.command("hub.display.show('0')")
            running_pred = np.append(running_pred,0) 
        elif(predicted_class=='rotten_apples' or predicted_class=='rotten_oranges'):
            # LEGO.command("hub.display.show('1')")
            running_pred = np.append(running_pred,1)

        if(predicted_class_2=='fresh_oranges' or predicted_class_2=='fresh_apples'):
            running_pred = np.append(running_pred,0)
        elif(predicted_class_2=='rotten_apples' or predicted_class_2=='rotten_oranges'):
            running_pred = np.append(running_pred,1)
        
        if running_cord_thres_x_min[-1] > 0.8:
            if running_pred.mean() > 0.5:
                LEGO.command("hub.display.show('1')")
                #push - can push twice?
                LEGO.command("push_motor.run_for_degrees(620, 100)")
                LEGO.command("push_motor.run_for_degrees(-620, 100)")
            else:
                LEGO.command("hub.display.show('0')")

            print(f"prediction value: {running_pred.mean()}")
            running_pred = np.array([])

    
print_to_terminal(image,predicted_class,prediction_values)
LEGO.command("conveyor_motor.stop()") 



