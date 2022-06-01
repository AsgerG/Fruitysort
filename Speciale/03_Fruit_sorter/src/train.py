from cProfile import label
import json
#afrom msilib.schema import Directory
import torch.optim as optim
import torch.nn as nn
import torch
import os

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter           # run this in your cmd at src level:   $ tensorboard --logdir=runs
from data_loader import create_dataloader, create_single_dataloader
from model import Net

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

data_path = config['files']['folder_path'] + 'data/'
categories = config['data']["categories"]
csv_tag = config['files']['csv_tag']
default_device = config['training']['default_device']
num_classes = config['data']['categories']  

# set data path and device
folder_path = config['files']['folder_path']

csv_train_file = 'data_csv/train_no_bananas_' + csv_tag + '.csv'
csv_test_file = 'data_csv/test_no_bananas_' + csv_tag +'.csv'
#csv_train_file = 'data_csv/train_' + csv_tag + '.csv'
#csv_test_file = 'data_csv/test_' + csv_tag +'.csv'


print(f">> Using device: {default_device}")


# Load meta data dictionary
meta_data_file = open(data_path + "meta_data/data.json", "r")
#meta_data_file = open(data_path + "meta_data/data_with_bananas.json", "r")

meta_dict = json.load(meta_data_file)

# Init fields
EPOCHS = config['training']['epochs']
image_size = config['data']['image_size']
batch_size = config['data']['batch_size']
learning_rate = config['training']['learning_rate']

num_of_pictures = meta_dict["train_load"]
batches_in_epoch = int(num_of_pictures/batch_size)

# Init dataloaders
train_dataloader, test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size, device=default_device, csv_train_file=csv_train_file, csv_test_file=csv_test_file)

icelandic_dataloader = create_single_dataloader(data_path, "test", "generated_data/island_cropped", batch_size = batch_size, image_size = image_size, csv_test_file="data_csv/test_iceland_" + csv_tag +".csv", device=default_device)
prototype_dataloader = create_single_dataloader(data_path, "test", "generated_data/setup_images_cropped/", batch_size = batch_size, image_size = image_size, csv_test_file="data_csv/test_prototype_" + csv_tag +".csv", device=default_device)

# Init Model
net = Net()
net = net.to(default_device)


# init loss and learning step type
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        
        #print to verify correct device
        #print(f"input: {inputs.device}, label: {labels.device}")
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs)
        
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batches_in_epoch == batches_in_epoch-1:
            last_loss = running_loss / batches_in_epoch # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
writer = SummaryWriter('runs/{}_model_{}_{}_{}_{}'.format(csv_tag, timestamp, num_classes, image_size, batch_size))
new_directory = '{}_model_{}_{}_{}_{}'.format(csv_tag, timestamp, num_classes, image_size, batch_size)
new_model_path = folder_path + "models/" + new_directory 
os.mkdir(new_model_path)
epoch_number = 0



best_vloss = 1_000_000.

def get_avg_validation_loss(test_dataloader, net, criterion):
    running_vloss = 0.0
    for i, vdata in enumerate(test_dataloader):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss



for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    net.train(False)

    running_vloss = 0.0

    """
    for i, vdata in enumerate(test_dataloader):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss.item()
    """
    
    avg_vloss = get_avg_validation_loss(test_dataloader, net, criterion)
    avg_iceland_loss = get_avg_validation_loss(icelandic_dataloader, net, criterion)
    avg_prototype_loss = get_avg_validation_loss(prototype_dataloader, net, criterion)    
    print('LOSS train {} valid {}, iceland {}, prototype {}'.format(avg_loss, avg_vloss, avg_iceland_loss, avg_prototype_loss))
    #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation

    #writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
    writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_loss, 'Validation_kaggle' : avg_vloss, 'Validation_iceland' : avg_iceland_loss, 'Validation_prototype' : avg_prototype_loss }, epoch_number + 1)

    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss or epoch == EPOCHS-1 or epoch%20 == 0:
        best_vloss = avg_vloss
        model_path = new_model_path + '/version_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1


