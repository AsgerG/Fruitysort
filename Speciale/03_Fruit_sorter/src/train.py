import json
import torch.optim as optim
import torch.nn as nn
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from data_loader import create_dataloader
from model import Net

data_path = "C:/Users/Asger/OneDrive/Dokumenter/Speciale/03_Fruit_sorter/data/"

# Load meta data dictionary
meta_data_file = open(data_path + "meta_data/data.json", "r")
meta_dict = json.load(meta_data_file)

# Init fields
image_size = 128
batch_size = 32 
num_of_pictures = meta_dict["train_load"]
batches_in_epoch = int(num_of_pictures/batch_size)

# Init dataloaders
train_dataloader, test_dataloader = create_dataloader(data_path, batch_size=batch_size, image_size=image_size)

# Init Model
net = Net()

# init loss and learning step type
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

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
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 50

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    net.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(test_dataloader):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1


