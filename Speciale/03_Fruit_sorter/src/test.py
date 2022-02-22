import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Net
from data_loader import create_dataloader


data_path = "C:/Users/Asger/OneDrive/Dokumenter/Speciale/03_Fruit_sorter/data/"
model_path = "C:/Users/Asger/OneDrive/Dokumenter/Speciale/03_Fruit_sorter/models/model_128_32_feb21/model_20220221_132450_42"


# Load model
saved_model = Net()
saved_model.load_state_dict(torch.load(model_path))

# Init fields
image_size = 128
batch_size = 32 

# Init dataloaders
train_dataloader, test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size)


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
    
    

