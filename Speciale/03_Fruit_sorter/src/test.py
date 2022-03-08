import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

#os.chdir("src")
#os.getcwd()

from model import Net
from data_loader import create_single_dataloader, create_dataloader

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
default_device = 'cpu'

folder_path = config['files']['folder_path']

model_name = 'categorical_model_2022-03-02_1844_6_128_64/version_2022-03-02_1844_61' # model_folder/model_version
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

# Init dataloaders

#generated data:
#csv_test_file = 'data_csv/test_generated_data_' + csv_tag +'.csv'
#test_dataloader = create_single_dataloader(data_path, "test", "generated_data/", batch_size = 32, image_size = 128, csv_test_file="data_csv/test_generated_data_" + csv_tag +".csv")

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
    




    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_dataloader:
        output = saved_model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
    
    # constant for classes
    classes = ('freshapples', 'rottenapples', 'freshbananas', 'rottenbananas', 'freshoranges','rottenoranges')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    #cf_matrix = cf_matrix.astype(float)
    
    df_percentage = np.empty(shape=(len(classes),len(classes))).astype(float)

    for i in range(len(classes)):
        df_percentage[:,i] = (cf_matrix[:,i]/(cf_matrix[:,i].sum())*100).round(2)

    df_cm = pd.DataFrame(df_percentage, index = [i for i in classes],
                        columns = [i for i in classes])


    labels = (np.asarray([f"{value} % \n ({string}) "
                      for string, value in zip(cf_matrix.flatten(),
                                               df_percentage.flatten())])).reshape(len(classes), len(classes))
    
    test_set_name = csv_test_file.split('/')[1]

    fig, ax = plt.subplots(figsize = (12,7))

    sn.heatmap(df_cm, annot=labels, fmt="", cmap='RdYlGn', ax=ax, vmin=0, vmax=100)
    plt.xlabel("True Class")    
    plt.ylabel("Predicted Class")
    plt.title(f"Confusion matrix for : {test_set_name} \n model: {model_name}")
    plt.show()
    plt.savefig('output.png')

