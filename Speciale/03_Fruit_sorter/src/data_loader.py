import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import json

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
default_device = config['training']['default_device']
csv_tag = config['files']['csv_tag']
csv_train_file = 'data_csv/train_' + csv_tag + '.csv'
csv_test_file = 'data_csv/test_' + csv_tag +'.csv'


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device=default_device):
        self.csv = pd.read_csv(annotations_file)
        self.img_labels = torch.tensor(self.csv["label"].values)
        self.img_labels = self.img_labels.to(device)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.csv.iloc[idx, 0])
        image = read_image(img_path)
        image = torch.cat((torch.split(image,1)[0],torch.split(image,1)[1],torch.split(image,1)[2]),0)/255
        image = image.to(self.device)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_dataset(image_size, data_path, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file):
    Transformations = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    training_data = CustomImageDataset(data_path + csv_train_file, data_path + "dataset/",transform=Transformations, device=device)
    test_data = CustomImageDataset(data_path + csv_test_file, data_path + "dataset/",transform=Transformations, device=device)
    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(test_data)))
    
    return training_data, test_data


def create_dataloader(data_path, batch_size = 32, image_size = 128, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file):
    training_data, test_data = create_dataset(image_size, data_path, device=device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader 


def create_single_dataset(image_size, data_path, data_tag, folder, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file):
    Transformations = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    if data_tag == 'train':
        training_data = CustomImageDataset(data_path + csv_train_file, data_path + folder, transform=Transformations, device=device)
        print('Training set has {} instances'.format(len(training_data)))
        return training_data
    if data_tag == 'test':
        test_data = CustomImageDataset(data_path + csv_test_file, data_path + folder, transform=Transformations, device=device)
        print('Validation set has {} instances'.format(len(test_data)))
        return test_data
        

def create_single_dataloader(data_path, data_tag, folder, batch_size = 32, image_size = 128, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file):
    data = create_single_dataset(image_size, data_path, data_tag, folder, device=device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)    
    return dataloader 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f">> Using device: {device}")

# move the model to the device
#self.to(device)



if __name__ == "__main__":

    # Make Dataset and Dataloader
    train_dataloader, test_dataloader = create_dataloader(data_path, batch_size = 32, image_size = 128)
    
    # Display image example and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[11].squeeze()
    label = train_labels[11]

    print(f"image was on device: {img.device}, label on {label.device}")
    img = img.to("cpu")
    print(f"image is now on device: {img.device}")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
    img = torch.permute(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")


    """
    # test dataloader for generated data
    test_generated_dataloader = create_single_dataloader(data_path, "test", "generated_data/", batch_size = 32, image_size = 128, csv_test_file="data_csv/test_generated_data_" + csv_tag +".csv")
        # Display image example and label.
    test_genereated_features, test_genereated_labels = next(iter(test_generated_dataloader))
    img = test_genereated_features[11].squeeze()
    label = test_genereated_labels[11]

    print(f"image was on device: {img.device}, label on {label.device}")
    img = img.to("cpu")
    print(f"image is now on device: {img.device}")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
    img = torch.permute(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
    """


