import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

import json

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

# hyperameters of the model
num_classes = config['data']['categories']  
height = config['data']['image_size']
width = config['data']['image_size']
batch_size = config['data']['batch_size']
channels = 3

# 1. Conv layer
num_filters_conv1 = 16                                            # 16 convolution filters with a
kernel_size_conv1 = 3                                             # filter size of 3x3, kernel regularizer, and bias regularizer of 0.05. It also uses random_uniform, which is a kernel initializer
stride_conv1 = 1
padding_conv1 = 1                                                 # pad with 1 to get no cahnge in dimensions

# 2. Conv layer
num_filters_conv2 = 16
kernel_size_conv2 = 5 # [height, width]
stride_conv2 = 1 # [stride_height, stride_width]
padding_conv2 = 2

# 3. Conv layer
num_filters_conv3 = 16
kernel_size_conv3 = 7 # [height, width]
stride_conv3 = 1 # [stride_height, stride_width]
padding_conv3 = 3

# 4. Fully connected
fc_layer_in = int(height/8 * width/8 * num_filters_conv3)
fc_layer_out = num_classes

# Batch norm and Max pooling 
maxpooling_size = 2
maxpooling_stride = 2

# Dropout
dropout_rate = 0.5

# The Network
class Net(nn.Module):

    def __init__(self, image_size = height, num_classes=num_classes):
        super(Net, self).__init__()
        
        self.fc_layer_in = int(image_size/8 * image_size/8 * num_filters_conv3)
        self.fc_layer_out = num_classes
        # out_dim = (input_dim - filter_dim + 2padding) / stride + 1

        #1
        self.conv_1 = Conv2d(in_channels=channels,
                             out_channels=num_filters_conv1,
                             kernel_size=kernel_size_conv1,
                             stride=stride_conv1, 
                             padding = padding_conv1)
        
        #2
        self.conv_2 = Conv2d(in_channels=num_filters_conv2,
                             out_channels=num_filters_conv2,
                             kernel_size=kernel_size_conv2,
                             stride=stride_conv2, 
                             padding = padding_conv2,)

        #3
        self.conv_3 = Conv2d(in_channels=num_filters_conv3,
                             out_channels=num_filters_conv3,
                             kernel_size=kernel_size_conv3,
                             stride=stride_conv3, 
                             padding = padding_conv3,)
        
        self.batchnorm_1 = BatchNorm2d(num_filters_conv1) # OBS: make several of these if number of channels changes throughout the system 
        self.batchnorm_2 = BatchNorm2d(num_filters_conv2)
        self.batchnorm_3 = BatchNorm2d(num_filters_conv3)
        self.maxpool = MaxPool2d(kernel_size=maxpooling_size, stride=maxpooling_stride)
        self.dropout = Dropout2d(p=dropout_rate)
        

        #4
        self.l_1 = Linear(in_features=self.fc_layer_in, 
                          out_features=self.fc_layer_out,
                          bias=True)


        #self.l_out = Linear(in_features=num_l1, 
        #                    out_features=num_classes,
        #                    bias=False)




    def forward(self, x): # x.size() = [batch, channel, height, width]
        #global x_conv2_size
        #global x_maxpool_size

        # 1.0
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # 2.0
        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # 3.0
        x = self.conv_3(x)
        x = self.batchnorm_3(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # 4.0
        #x = torch.flatten(x)
        x = x.view(-1, fc_layer_in)
        x = self.l_1(x)
        
        return softmax(x, dim=1)


if __name__ == "__main__":

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #Test the forward pass with dummy data
    x = np.random.normal(0,1, (5, 3, height, width)).astype('float32')
    out = net(Variable(torch.from_numpy(x)))
    print(out.size())
    print(out)
