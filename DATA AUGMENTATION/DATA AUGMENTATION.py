##DATA AUGMENTATION
import torch
# torch.load('rotated_data.pt')
# torch.load('normal.pt')
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import numpy as np
import torchvision.datasets as dsets
import os

def plot_cost_accuracy(checkpoint):
    #plot the cost and accuracy
    fig,ax1=plt.subplots()
    color='tab:red'
    ax2.set_ylabel('accuracy',color=color)
    ax2.set_xlabel('epoch',color=color)
    ax2.plot(checkpoint['accuracy'],color=color)
    ax2.tick_params(axis = 'y',color=color)
    fig.tight_layout()

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE,IMAGE_SIZE),cmap = 'gray')
    plt.title('y='+str(data_sample[1]))
    plt.show()

def plot_mis_classified(model,dataset):
    count=0
    for x, y in torch.utils.data.DataLoader(dataset=dataset,batch_size=1):
        z=model(x)
        _,yhat=torch.max(z,1)
        if yhat!=y:
            show_data((x,y))
            plt.show()
            count+=1
        if count>=5:
            break

#LOAD DATA
#SIZE IF THE IMAGES ARE 16 BY 16

IMAGE_SIZE=16
#CREATING A GROUP OF TRANSFORMATION IT CREATEDA ROTATED DATASET
#RESIZE THE IMAGE ,RANDOMLY ROTATE IT AND CONVERTS IT TO A TENSOR
compose_rotate=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.RandomAffine(45),transforms.ToTensor()])
#creating a group of transformations to create a non rotated dataset
#resize the image then converts it to a tensor
compose=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor()])


#the transform parameters is set to the corresponding compose
train_dataset_rotate=dsets.MNIST(root = './data',train = True,download = True,transform = compose_rotate)
train_dataset=dsets.MNIST(root = './data',train = True,download = True,transform = compose)
#load the testing dataset
validation_dataset=dsets.MNIST(root = './data',train = False,download = True,transform = compose_rotate)

#the image for the first data sample
# show_data(train_dataset[0])
#the label for the first data element
# print(train_dataset[0][1]

# show_data(train_dataset_rotate[0])


#BUILDING A CONVOLUTIONAL NEURAL NETWORK CLASS
class CNN(nn.Module):
    #constructor
    def __init__(self,out_1=16,out_2=32):
        super(CNN,self).__init__()
        #the reaseon we start 1 channel is because we have a single black and white image
        #channel width after this layer is 16
        self.cnn1=nn.Conv2d(in_channels = 1,out_channels = out_1,kernel_size = 5,padding = 2)
        #channel with after this layer is 8
        self.maxpool1=nn.MaxPool2d(kernel_size = 2)
        #channel width after this layer is 8
        self.cnn2=nn.Conv2d(in_channels = out_1,out_channels = out_2,kernel_size = 5,stride = 1,padding=2)
        #channel width after this layer is 4
        self.maxpool2=nn.MaxPool2d(kernel_size = 2)
        #in total we have out_2(32) channels which are each 4*4 in size based on the width calculation above . channels are square
        #the output is a value for each class
        self.fc1=nn.Linear(out_2*4*4,10)
    #prediction
    def forward( self,x ):
        #puts the x value through each cnn,relu and pooling layer and it is flattened for input into the fully connected layer
        x=self.cnn1(x)
        x=torch.relu(x)
        x=self.maxpool1(x)
        x=self.cnn2(x)
        x=self.maxpool2(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x
    #outputs result of each stafe of the cnn, relu and pooling layers
    def activation( self,x ):
        #outputs activation this is not necessary
        z1=self.cnn1(x)
        a1=torch.relu(z1)
        out=self.maxpool1(a1)
        z2=self.cnn2(out)
        a2=torch.relu(z2)
        out1=self.maxpool2(a2)
        out=out.view(out.size(0),-1)
        return z1,a1,z2,a2,out1,out

#CREATE THE MODEL OBJECT TO BE TREATED ON REGULAR DATA USIN CNN CLASS
model=CNN(out_1=10,out_2 = 12)
#we create a criterion which will measure loss
criterion=nn.CrossEntropyLoss()
learning_rate=0.1
#create an optimizer that updates model parameters using the learning rate and gradient
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
#create a data loader for the training data with a batch size of100
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100)
#create a data loader for the rotated validation data with a batch size og 5000
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=5000)

# Train the model
import os

# Location to save data
file_normal = os.path.join ( os.getcwd ( ) , 'normal.pt' )

# All the data we are saving
checkpoint = {
    # Saving the number of epochs the models was trained for
    'epoch' : None ,
    # Saving the models parameters which will allow us to recreate the trained model
    'model_state_dict' : None ,
    # Saving the optimizers parameters
    'optimizer_state_dict' : None ,
    # Saving the loss on the training dataset for the last batch of the last epoch
    'loss' : None ,
    # Saving the cost on the training dataset for each epoch
    'cost' : [ ] ,
    # Saving the accuracy for the testing dataset for each epoch
    'accuracy' : [ ]}

# Number of epochs to train model
n_epochs = 5

# Size of the testing dataset
N_test = len ( validation_dataset )

# Training for the number of epochs we want
for epoch in range ( n_epochs ) :
    # Variable to keep track of cost for each epoch
    cost = 0
    # For each batch in the training dataset
    for x , y in train_loader :
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad ( )
        # Makes a prediction on the image
        z = model ( x )
        # Calculate the loss between the prediction and actual class
        loss = criterion ( z , y )
        # Calculates the gradient value with respect to each weight and bias
        loss.backward ( )
        # Updates the weight and bias according to calculated gradient value
        optimizer.step ( )

        # Saves the number of epochs we trained for
        checkpoint [ 'epochs' ] = n_epochs
        # Saves the models parameters
        checkpoint [ 'model_state_dict' ] = model.state_dict ( )
        # Saves the optimizers paramters
        checkpoint [ 'optimizer_state_dict' ] = optimizer.state_dict ( )
        # Saves the loss for the last batch so ultimately this will be the loss for the last batch of the last epoch
        checkpoint [ 'loss' ] = loss
        # Accumulates the loss
        cost += loss.item ( )

    # Counter for the correct number of predictions
    correct = 0

    # For each batch in the validation dataset
    for x_test , y_test in validation_loader :
        # Make a prediction
        z = model ( x_test )
        # Get the class that has the maximum value
        _ , yhat = torch.max ( z.data , 1 )
        # Counts the number of correct predictions made
        correct += (yhat == y_test).sum ( ).item ( )

    accuracy = correct / N_test
    print ( accuracy )
    # Appends the cost of the epoch to a list
    checkpoint [ 'cost' ].append ( cost )
    # Appends the accuracy of the epoch to a list
    checkpoint [ 'accuracy' ].append ( accuracy )
    # Saves the data in checkpoint to the file location
    torch.save ( checkpoint , file_normal )



#ANALYZE RESULT

#LOADS THE DATA WHICH IS SAVED IN NOTMAL.PT
checkpoint_normal=torch.load(os.path.join(os.getcwd()),'normal.pt')
#PLOT ACCURACY AND COST VS EPOCH GRAPH
#USING THE HELPER FUNCTION DEFINED AT THE TOP AND THE COST AND ACCURACY LIST THAT WE SAVED
plot_cost_accuracy(checkpoint_normal)

#five mis classified samples
#using the model parameters we saved we loas them into a model to recreate the trainned model
model.load_state_dict(checkpoint_normal['model_state_dict'])
#setting the model to evaluation model
model.eval()
#usin the helper function plit the first five mis classified samples
plot_mis_classified(model,validation_dataset)

#rotated training data
#create the model object using CNN class
model_r=CNN(out_1 =16,out_2=32)
#ew create a criterion  which will measure loss
criterion=nn.CrossEntropyLoss()
learning_rate=0.1
#create an optimezer that updates model parameters using the learning rate and gradient
optimizer= torch.optim.SGD(model_r.parameters(),lr=learning_rate)
#create a data loader for the rotated training data wiith a batch size of 100
train_loader=torch.utils.data.DataLoader(dataset=train_dataset_rotate,batch_size=100)
#create a data loader for the rotated validation data with a branch sizez og 5000
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=5000)

# Location to save data
file_rotated = os.path.join ( os.getcwd ( ) , 'rotated_data.pt' )

# All the data we are saving
checkpoint = {
    # Saving the number of epochs the models was trained for
    'epoch' : None ,
    # Saving the models parameters which will allow us to recreate the trained model
    'model_state_dict' : None ,
    # Saving the optimizers parameters
    'optimizer_state_dict' : None ,
    # Saving the loss on the training dataset for the last batch of the last epoch
    'loss' : None ,
    # Saving the cost on the training dataset for each epoch
    'cost' : [ ] ,
    # Saving the accuracy for the testing dataset for each epoch
    'accuracy' : [ ]}

# Number of epochs to train model
n_epochs = 5

# Size of the testing dataset
N_test = len ( validation_dataset )

# Training for the number of epochs we want
for epoch in range ( n_epochs ) :
    # Variable to keep track of cost for each epoch
    cost = 0
    # For each batch in the training dataset
    for x , y in train_loader :
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad ( )
        # Makes a prediction on the image
        z = model_r ( x )
        # Calculate the loss between the prediction and actual class
        loss = criterion ( z , y )
        # Calculates the gradient value with respect to each weight and bias
        loss.backward ( )
        # Updates the weight and bias according to calculated gradient value
        optimizer.step ( )

        # Saves the number of epochs we trained for
        checkpoint [ 'epochs' ] = n_epochs
        # Saves the models parameters
        checkpoint [ 'model_state_dict' ] = model.state_dict ( )
        # Saves the optimizers paramters
        checkpoint [ 'optimizer_state_dict' ] = optimizer.state_dict ( )
        # Saves the loss for the last batch so ultimately this will be the loss for the last batch of the last epoch
        checkpoint [ 'loss' ] = loss
        # Accumulates the loss
        cost += loss.item ( )

    # Counter for the correct number of predictions
    correct = 0

    # For each batch in the validation dataset
    for x_test , y_test in validation_loader :
        # Make a prediction
        z = model_r ( x_test )
        # Get the class that has the maximum value
        _ , yhat = torch.max ( z.data , 1 )
        # Counts the number of correct predictions made
        correct += (yhat == y_test).sum ( ).item ( )

    accuracy = correct / N_test
    print ( accuracy )
    # Appends the cost of the epoch to a list
    checkpoint [ 'cost' ].append ( cost )
    # Appends the accuracy of the epoch to a list
    checkpoint [ 'accuracy' ].append ( accuracy )
    # Saves the data in checkpoint to the file location
    torch.save ( checkpoint , file_rotated )


#ANA LYZE THE RESULTS
checkpoint_rotated = torch.load ( os.path.join ( os.getcwd ( ) , 'rotated_data.pt' ) )
#USING THE HELPER FUNCTION DEFINED AT THE TOP AND THE COST AND ACCURACY LIST THAT WE SAVED
plot_cost_accuracy(checkpoint_rotated)

# Using the model parameters we saved we load them into a model to recreate the trained model
model_r.load_state_dict(checkpoint_rotated['model_state_dict'])
# Setting the model to evaluation mode
model.eval()
# Using the helper function plot the first five misclassified samples
plot_mis_classified(model_r,validation_dataset)