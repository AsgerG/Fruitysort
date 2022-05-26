from datetime import datetime
import time
from lego import Lego, predict_image, take_picture, init_model, print_to_terminal ,process_image
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

#load model
path_to_yolov5 = r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\yolov5"
path_to_custom_model = r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\yolov5\runs\train\exp4\weights\best.pt"
model = torch.hub.load(path_to_yolov5, 'custom', path=path_to_custom_model, source='local')
model.conf = 0.35


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
LEGO.command("conveyor_motor.start(5)")

running_pred = np.array([])
running_cord_thres_x_min = []

running_pred_2 = np.array([])
running_cord_thres_x_min_2 = []

running_labels_length = np.array([])

# Run forever
while(True):

    #Take picture    
    image = take_picture(url,True)
    # image  = process_image(image, 'default', 0.9)
    
    #Prediction
    results=model(image)
    results_df = results.xyxyn[0].cpu().numpy()
    
    
    results.show()
    # results = model(r"C:\Users\andri\OneDrive\Documents\DTU\Fruitysort\Speciale\03_Fruit_sorter\src\WebCamCapture.png")
    # results.xyxyn[0].cpu()
    # labels, cord_thres = results.xyxyn[0].cpu()[:, -1].numpy(), results.xyxyn[0].cpu()[:, :-1].numpy()
    # results.save(save_dir="C:/Users/andri/OneDrive/Documents/DTU/Fruitysort/Speciale/03_Fruit_sorter/src/",)
    # running_labels_length = np.append(running_labels_length,len(labels))
    
    if len(results_df) > 1:
        #check x_min coords
        index_to_remove = []
        for i in range(len(results_df)-1):
            if abs(results_df[i+1,0]-results_df[i,0])<0.05:               

                if(results_df[i+1,4]<results_df[i,4]):
                    index_to_remove.append(i+1)
                else:
                    index_to_remove.append(i)

                print("fejl! - skal fjernes")
                
                
        results_df = np.delete(results_df,index_to_remove,0)
        # print(results_df)
        results.show()
    labels, cord_thres = results_df[:,-1], results_df[:,:-2]


    ## Reaction
    if len(labels) == 1:
        results.show()
        
        
        # If labels has been 2 but now is 1
        if not len(running_pred_2) == 0:
            # print(f"flushing {running_pred_2}")
            running_pred = running_pred_2
            running_pred_2 = np.array([])

        running_cord_thres_x_min.append(cord_thres[0][0])
        #print(results.pandas().xyxy[0])
        predicted_class = labels[0]
        
        if(predicted_class%2==0):
            # LEGO.command("hub.display.show('0')")
            running_pred = np.append(running_pred,0)
        else:
            # LEGO.command("hub.display.show('1')")
            running_pred = np.append(running_pred,1)
        
        if running_cord_thres_x_min[-1] > 0.8:
            if running_pred.mean() > 0.5:
                LEGO.command("hub.display.show('1')")
                #push - can push twice?
                # LEGO.command("push_motor.run_for_degrees(620, 100)")
                # LEGO.command("push_motor.run_for_degrees(-620, 100)")
            else:
                LEGO.command("hub.display.show('0')")
            # print(f"prediction value: {running_pred.mean()}")
            # running_pred = np.array([])
        

    elif len(labels) == 2:
        # print("two")
        running_cord_thres_x_min.append(cord_thres[1][0])
        running_cord_thres_x_min_2.append(cord_thres[0][0])
        #print(results.pandas().xyxy[0])
        predicted_class = labels[1]
        predicted_class_2 = labels[0]

        if(predicted_class%2==0):
            # LEGO.command("hub.display.show('0')")
            running_pred = np.append(running_pred,0) 
        else:
            # LEGO.command("hub.display.show('1')")
            running_pred = np.append(running_pred,1)

        if(predicted_class_2%2==0):
            running_pred_2 = np.append(running_pred_2,0)
        else:
            running_pred_2 = np.append(running_pred_2,1)
        
        if running_cord_thres_x_min[-1] > 0.8:
            if running_pred.mean() > 0.5:
                LEGO.command("hub.display.show('1')")
                #push - can push twice?
                # LEGO.command("push_motor.run_for_degrees(620, 100)")
                # LEGO.command("push_motor.run_for_degrees(-620, 100)")
            else:
                LEGO.command("hub.display.show('0')")

            
            # print(f"prediction value: {running_pred.mean()}")
            # # running_pred = np.array([])
    # elif len(labels) > 2:
        # results.show()

    
print_to_terminal(image,predicted_class,prediction_values)
LEGO.command("conveyor_motor.stop()") 



