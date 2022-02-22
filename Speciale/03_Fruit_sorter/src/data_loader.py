import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = torch.cat((torch.split(image,1)[0],torch.split(image,1)[1],torch.split(image,1)[2]),0)/255
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_dataset(image_size, data_path):
    Transformations = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    training_data = CustomImageDataset(data_path + "data_csv/train.csv", data_path + "/dataset/",transform=Transformations)
    test_data = CustomImageDataset(data_path + "data_csv/test.csv", data_path + "/dataset/",transform=Transformations)
    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(test_data)))
    
    return training_data, test_data


def create_dataloader(data_path, batch_size = 32, image_size = 128):
    training_data, test_data = create_dataset(image_size, data_path)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader 






if __name__ == "__main__":
    data_path = "C:/Users/Asger/OneDrive/Dokumenter/Speciale/03_Fruit_sorter/data/"

    # Make Dataset and Dataloader
    train_dataloader, test_dataloader = create_dataloader(data_path, batch_size = 32, image_size = 128)

    # Display image example and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[11].squeeze()
    label = train_labels[11]
    os.environ['KMP_DUPLICATE_LIB_OK']='True' # have to set for kernel not to crash on imshow()
    img = torch.permute(img, (1, 2, 0)) 
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")

    