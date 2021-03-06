import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#os.chdir("src")
#os.getcwd()

from model import Net
from data_loader import create_single_dataloader, create_dataloader

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
default_device = 'cuda'

folder_path = config['files']['folder_path']

#model_name = 'categorical_model_2022-03-02_1844_6_128_64/version_2022-03-02_1844_61'
model_name = 'categorical_model_2022-06-01_1642_6_224_32/version_2022-06-01_1642_1' # model_folder/model_version
model_path = folder_path + 'models/' + model_name


# Init fields   - should match tested model
model_data = model_name.split("_")

image_size = int(model_data[5])
batch_size = int(model_data[6].split("/")[0])
num_classes = int(model_data[4])
csv_tag = model_data[0]

csv_train_file = 'data_csv/train_no_bananas_' + csv_tag + '.csv'
csv_test_file = 'data_csv/train_no_bananas_' + csv_tag +'.csv'

# Load model
saved_model = Net(image_size=image_size, num_classes=num_classes)

# For cpu model saving
# model_path = model_path + '_cpu'
# saved_model = saved_model.to('cpu')
# torch.save(saved_model.state_dict(), model_path)

saved_model.load_state_dict(torch.load(model_path))
# scripted_model = torch.jit.script(saved_model)

# torch.jit.save(scripted_model, 'model_to_nicki.pt')

saved_model = saved_model.to(default_device)

# Init dataloaders

#generated data:
# csv_test_file = 'data_csv/test_generated_data_' + csv_tag +'.csv'

#test_dataloader = create_single_dataloader(data_path, "test", "generated_data_cropped/", batch_size = batch_size, image_size = image_size, csv_test_file="data_csv/test_generated_cropped_data_" + csv_tag +".csv", device=default_device)
train_dataloader, test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)
#test_dataloader = create_single_dataloader(data_path, "test", "generated_data/island_cropped", batch_size = batch_size, image_size = image_size, csv_test_file="data_csv/test_iceland_" + csv_tag +".csv", device=default_device)
#test_dataloader = create_single_dataloader(data_path, "test", "generated_data/setup_images_cropped/", batch_size = batch_size, image_size = image_size, csv_test_file="data_csv/test_prototype_" + csv_tag +".csv", device=default_device)
    
"""
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
    """




saved_model.eval()
with torch.no_grad():
    y_pred = []
    y_pred_raw = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_dataloader:
        output = saved_model(inputs) # Feed Network
        
        output_raw = output[:,1]
        output_onehot = (torch.max(output, 1)[1]).data.cpu().numpy()        
        
        y_pred_raw.extend(output_raw)
        y_pred.extend(output_onehot) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        
    cf_matrix = confusion_matrix(y_true, y_pred)

    """
    # GET ACCURACY and PRECISION
    true_values = np.array(y_true)
    predictions = np.array(y_pred)
    
    N = len(true_values)
    accuracy = (true_values == predictions).sum() / N
    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    precision = TP / (TP+FP)
    """

    # constant for classes
    if num_classes == 6:
        classes = ('freshapples', 'rottenapples', 'freshbananas', 'rottenbananas', 'freshoranges','rottenoranges')
    if num_classes == 2:
        classes = ('fresh', 'rotten')

    if num_classes == 4:
        classes = ('freshapples', 'rottenapples', 'freshoranges','rottenoranges')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    #cf_matrix = cf_matrix.astype(float)
    

    df_percentage = np.empty(shape=(len(classes),len(classes))).astype(float)

    for i in range(len(classes)):
        #df_percentage[:,i] = (cf_matrix[:,i]/(cf_matrix[:,i].sum())*100).round(2)
        df_percentage[i,:] = (cf_matrix[i,:]/(cf_matrix[i,:].sum())*100).round(2)



    df_cm = pd.DataFrame(df_percentage, index = [i for i in classes],
                        columns = [i for i in classes])


    labels = (np.asarray([f"{value} % \n ({string}) "
                      for string, value in zip(cf_matrix.flatten(),
                                               df_percentage.flatten())])).reshape(len(classes), len(classes))
    
    test_set_name = csv_test_file.split('/')[1]

    fig, ax = plt.subplots(figsize = (12,7))

    sn.heatmap(df_cm, annot=labels, fmt="", cmap='RdYlGn', ax=ax, vmin=0, vmax=100)
    plt.xlabel("Predicted Class")    
    plt.ylabel("True Class")
    #plt.title(f"Confusion matrix for : {test_set_name} \n model: {model_name}")
    plt.show()
    plt.savefig('output.png')


    if num_classes == 2:
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_raw)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
        display.plot(color="darkorange")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.show()





# Disable grad
with torch.no_grad():
    test_features, test_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_labels.size()}")
    for i in range(63):
        img = test_features[i].squeeze()
        label = test_labels[i]

        predict_img = img[None, :, :, :]
        img.size()

        prediction = saved_model(predict_img)
        predicted_class = np.argmax(prediction)


        if(label!=predicted_class):
            os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
            img = torch.permute(img, (1, 2, 0)) 
            plt.imshow(img)
            plt.show()
            print(f"Label: {label}")
            print(f"Prediction: {predicted_class}")
            print(max(prediction))
        
