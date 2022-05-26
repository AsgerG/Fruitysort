from datetime import datetime
import time
from lego import Lego, predict_image, take_picture, init_model, print_to_terminal ,process_image
import csv
import pandas as pd


## Initalization of Hub and camera
#url = "http://10.209.243.249:8080/video"
# url = "http://192.168.1.5:8080/video"
url = "http://192.168.1.164:8080/video"
push_queue = []
LEGO = Lego(port="COM5")

LEGO.command("from mindstorms import Motor")
LEGO.command('from mindstorms import DistanceSensor')
LEGO.command('from mindstorms.control import wait_for_seconds')
LEGO.command("conveyor_motor = Motor('B')")
LEGO.command("image_detector = DistanceSensor('D')")
LEGO.command("push_detector = DistanceSensor('A')")
LEGO.command("push_motor = Motor('C')")

saved_model = init_model()
sensor_list1 = []
sensor_list2 = []
running_time_list = []


#Start Conveyorbelt process
LEGO.command("conveyor_motor.start(20)")
# LEGO.command("push_motor.run_for_degrees(620, 100)")
# LEGO.command("push_motor.run_for_degrees(-620, 100)")

distance_cam = 12.
distance_push = 11.
time_saved = datetime.now()
running_time_start = datetime.now()
#time_diff_tracking = datetime.now()
# Run forever
while(True):

    #State1 waiting for a fruit under the camera
    #print(distance_cam)
    while(distance_cam>6):
        LEGO.command("dist_cam = image_detector.get_distance_cm(short_range = True)")
        LEGO.command("dist_push = push_detector.get_distance_cm(short_range = True)")
        
        time_diff_tracking = (datetime.now()-running_time_start)
        running_time_list.append(float(str(time_diff_tracking.seconds)+"."+str(time_diff_tracking.microseconds)))

        distance_cam = LEGO.read_sensor_data("dist_cam", distance_cam)
        sensor_list1.append(distance_cam)
        distance_push = LEGO.read_sensor_data("dist_push", 12)
        
        sensor_list2.append(distance_push)
        time_saved = LEGO.pop_queue(push_queue, distance_push, time_saved, threshold=7) 
        


    LEGO.command("conveyor_motor.stop()") 
    #time.sleep(1)
    image = take_picture(url)
    LEGO.command("conveyor_motor.start(20)") 
    image  = process_image(image, 'default', 0.9)
    #Prediction
    predicted_class, prediction_values = predict_image(saved_model,image)
    # print_to_terminal(image, predicted_class, prediction_values)
    print_to_terminal(image,predicted_class,prediction_values)
    ## Reaction
    if(predicted_class==0):
        LEGO.command("hub.display.show('0')")
    elif(predicted_class==1):
        LEGO.command("hub.display.show('1')") 
    else:
        print('ERROR')
    
    push_queue.append(predicted_class)
    # Measure the distance between the Distance Sensor and object in centimeters and inches.

    #print(distance_cam)
    while(distance_cam<10):
        LEGO.command("dist_cam = image_detector.get_distance_cm(short_range = True)")
        LEGO.command("dist_push = push_detector.get_distance_cm(short_range = True)")

        time_diff_tracking = (datetime.now()-running_time_start)
        running_time_list.append(float(str(time_diff_tracking.seconds)+"."+str(time_diff_tracking.microseconds)))

        distance_cam = LEGO.read_sensor_data("dist_cam", distance_cam)
        sensor_list1.append(distance_cam)
               
        distance_push = LEGO.read_sensor_data("dist_push", 12)
        sensor_list2.append(distance_push)
        time_saved = LEGO.pop_queue(push_queue, distance_push, time_saved, threshold=7) 

sprint_to_terminal(image,predicted_class,prediction_values)
LEGO.command("conveyor_motor.stop()") 


time = datetime.now().strftime('%Y-%m-%d_%H%M')
test_name = "stop_problem_timer"
csv_path = 'C:/Users/andri/OneDrive/Documents/DTU/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_'+test_name+time+'.csv'
df = pd.DataFrame(data=zip(sensor_list1,sensor_list2,running_time_list))
df.to_csv(csv_path)

running_time_list
df = pd.read_csv(csv_path,index_col=0)



df.reset_index().plot.scatter(y='0',x='index',title=("Video sensor: " + test_name))
df.reset_index().plot.scatter(y='1',x='index', title=("Push sensor: " + test_name))

#df.plot(y='0', use_index=True)

#df.reset_index().plot.scatter(y='0',x='index',title=("Video sensor: " + test_name))

df.reset_index().plot(y='0',x='index',title=("Video sensor: " + test_name))
df.reset_index().plot(y='1',x='index', title=("Push sensor: " + test_name))
