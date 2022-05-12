import torch
import os

label = 4
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
path =r"C:\Users\Asger\OneDrive\Skrivebord\ModifiedOpenLabelling"
imagepath = path + '\\images'
textfilepath = path + '\\bbox_txt'

for k, file in enumerate(os.listdir(imagepath)):
    if k%10 == 0:
        print(k)
    try:
        filename = imagepath + '\\' + file
        textfilename = textfilepath + '\\' + file
        textfilename = textfilename.replace('.png','.txt')

        # print(filename)




        results = model(filename)
        results.xyxyn[0].cpu()
        labels, cord_thres = results.xyxyn[0].cpu()[:, -1].numpy(), results.xyxyn[0].cpu()[:, :-1].numpy()

        write_string = ''
        for i in range(len(cord_thres)):
            
            x_min = cord_thres[i][0]
            y_min = cord_thres[i][1]
            x_max = cord_thres[i][2]
            y_max = cord_thres[i][3]

            x_center = (x_min + x_max)/2
            y_center = (y_min + y_max)/2
            width = x_max - x_min
            height = y_max - y_min

            write_string = write_string + f"{label} {x_center} {y_center} {width} {height}\n"
        
        # print(write_string)

        with open(textfilename, 'w') as f:
             f.write(write_string)

    except:
        continue




