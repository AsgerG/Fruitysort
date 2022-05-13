from datetime import datetime
import time
from lego import Lego, predict_image, take_picture, init_model, print_to_terminal ,process_image



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


#Start Conveyorbelt process
LEGO.command("conveyor_motor.start(20)")

distance_cam = 12
distance_push = 11
time_saved = datetime.now()
# Run forever
while(True):

    #State1 waiting for a fruit under the camera
    #print(distance_cam)
    while(distance_cam>8):
        LEGO.command("dist_cam = image_detector.get_distance_cm()")
        LEGO.command("dist_push = push_detector.get_distance_cm()")

        distance_cam = LEGO.read_sensor_data("dist_cam", distance_cam)
        
        distance_push = LEGO.read_sensor_data("dist_push", 12)
        time_saved = LEGO.pop_queue(push_queue, distance_push, time_saved,threshold=5) 
        

    LEGO.command("conveyor_motor.stop()") 
    time.sleep(1)
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
        LEGO.command("dist_cam = image_detector.get_distance_cm()")
        LEGO.command("dist_push = push_detector.get_distance_cm()")
        
        distance_cam = LEGO.read_sensor_data("dist_cam", distance_cam)
        distance_push = LEGO.read_sensor_data("dist_push", 12)
        time_saved = LEGO.pop_queue(push_queue, distance_push, time_saved, threshold=7) 

print_to_terminal(image,predicted_class,prediction_values)
LEGO.command("conveyor_motor.stop()") 



