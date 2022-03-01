import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

#os.chdir("src")
#os.getcwd()

from model import Net
from data_loader import create_dataloader

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
default_device = config['training']['default_device']

folder_path = config['files']['folder_path']

model_name = 'model_128_32_categorical_feb21/model_20220221_132450_42' # model_folder/model_version
model_path = folder_path + 'models/' + model_name


# Init fields   - should match tested model
model_data = model_name.split("_")

image_size = int(model_data[1])
batch_size = int(model_data[2])
num_classes = int(model_data[3])
csv_tag = model_data[4]

csv_train_file = 'data_csv/train_' + csv_tag + '.csv'
csv_test_file = 'data_csv/test_' + csv_tag +'.csv'

# Load model
saved_model = Net(image_size=image_size, num_classes=num_classes)
saved_model.load_state_dict(torch.load(model_path))


# Init dataloaders
train_dataloader, test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)


# Disable grad
with torch.no_grad():
    test_features, test_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_labels.size()}")
    img = test_features[11].squeeze()
    label = test_labels[11]

    predict_img = img[None, :, :, :]
    img.size()

    prediction = saved_model(predict_img)
    predicted_class = np.argmax(prediction)


    os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
    img = torch.permute(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
    print(f"Prediction: {predicted_class}")
    


