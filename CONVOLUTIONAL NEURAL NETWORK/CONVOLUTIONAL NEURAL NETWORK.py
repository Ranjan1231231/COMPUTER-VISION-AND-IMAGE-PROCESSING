#CONVOLUTIONAL NEURAL NETWORK

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

#define the function for plotting the channels
def plot_channels(W):
    n_out=W.shape[0]
    n_in=W.shape[1]
    w_min=W.min().item()
    w_max=W.max().item()
    fig,axes=plt.subplots(n_out,n_in)
    fig.subplots_adjust(hspace = 0.1)
    out_index=0
    in_index=0
    #plot outputs as rows inputs as columns
    for ax in axes.flat:
        if in_index>n_in-1:
            out_index=out_index+1
            in_index=0
        ax.imshow(W[out_index,in_index,:,:],vmin = w_min,vmax=w_max,cmap = 'seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index=in_index+1
    plt.show()

#define the function for plotting the parameters
def plot_parameters(W,number_rows=1,name="",i=0):
    W=W.data[:,i,:,:]
    n_filters=W.shape[0]
    w_min=W.min().item()
    w_max=W.max().item()
    fig,axes=plt.subplots(number_rows,n_filters//number_rows)
    fig.subplots_adjust(hspace = 0.4)
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            #set the label for the subplot.
            ax.set_xlabel("kernel:{0}".format(i+1))

            #plot the image
            ax.imshow(W[i,:],vmin = w_min,vmax = w_max,cmap = 'seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name,fontsize=10)
    plt.show()


#Define the function for plotting the activations
def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace = 0.9)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()
#define the function show_data to plot out data samples as images
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE,IMAGE_SIZE),cmap = 'gray')
    plt.title("y="+str(data_sample[1].item()))


#GETTING THE DATA
IMAGE_SIZE = 16
# first, the image is resized then converted to a tensor
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)
# print out the fourth label
# show_data(train_dataset[3])#print out the fourth label
# show_data(train_dataset[3])
class CNN(nn.Module):
    #constuctor
    def __init__(self,out_1=16,out_2=32):
        super(CNN, self).__init__()
        #the reason we start with one channel is because we have a single black and white image
        #channel width after this layer is 16
        self.cnnl=nn.Conv2d(in_channels = 1,out_channels=out_1,kernel_size = 5,padding = 2)
        #channel wifth after this layer is 8
        self.maxpool1=nn.MaxPool2d(kernel_size = 2)

        #channel width after this layer is 8
        self.cnn2=nn.Conv2d(in_channels = out_1,out_channels = out_2,kernel_size = 5,stride = 1,padding=2)
        #channel width after this layer is 4
        self.maxpool2=nn.MaxPool2d(kernel_size = 2)
        #in total we have out_2(32)channels which are each 4*4in size based on the width calculation above.channels are squares
        #the output is a value for each class
        self .fc1=nn.Linear(out_2*4*4,10)
    #Prediction
    def forward( self,x ):
        #puts the x value through each cnn , relu adn pooling layer and it is flattened for input into fully connected layer
        x = self.cnnl ( x )
        x = torch.relu ( x )
        x = self.maxpool1 ( x )
        x = self.cnn2 ( x )
        x = torch.relu ( x )
        x = self.maxpool2 ( x )
        x = x.view ( x.size ( 0 ) , -1 )
        x = self.fc1 ( x )
        return x

    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations ( self , x ) :
        # Outputs activation this is not necessary
        z1 = self.cnnl ( x )
        a1 = torch.relu ( z1 )
        out = self.maxpool1 ( a1 )

        z2 = self.cnn2 ( out )
        a2 = torch.relu ( z2 )
        out1 = self.maxpool2 ( a2 )
        out = out.view ( out.size ( 0 ) , -1 )
        return z1 , a1 , z2 , a2 , out1 , out

#DEFINE THE CONVOLUTIONSAL NEURAL NETWORK CLASSIFIER,CRITERION FUNCTION,OPTIMIZER ADN TRAIN THE MODEL
#create the model object using CNN CLASS
model=CNN(out_1 = 16,out_2 = 32)

# Plot the parameters

plot_parameters(model.state_dict()['cnnl.weight'], number_rows=4, name="1st layer kernels before training ")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=4, name='2nd layer kernels before training' )
# We create a criterion which will measure loss
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
# Create an optimizer that updates model parameters using the learning rate and gradient
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# Create a Data Loader for the training data with a batch size of 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
# Create a Data Loader for the validation data with a batch size of 5000
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Train the model

# Number of times we want to train on the taining dataset
n_epochs = 3
# List to keep track of cost and accuracy
cost_list = [ ]
accuracy_list = [ ]
# Size of the validation dataset
N_test = len ( validation_dataset )


# Model Training Function
def train_model ( n_epochs ) :
    # Loops for each epoch
    for epoch in range ( n_epochs ) :
        # Keeps track of cost for each epoch
        COST = 0
        # For each batch in train loader
        for x , y in train_loader :
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad ( )
            # Makes a prediction based on X value
            z = model ( x )
            # Measures the loss between prediction and acutal Y value
            loss = criterion ( z , y )
            # Calculates the gradient value with respect to each weight and bias
            loss.backward ( )
            # Updates the weight and bias according to calculated gradient value
            optimizer.step ( )
            # Cumulates loss
            COST += loss.data

        # Saves cost of training data of epoch
        cost_list.append ( COST )
        # Keeps track of correct predictions
        correct = 0
        # Perform a prediction on the validation  data
        for x_test , y_test in validation_loader :
            # Makes a prediction
            z = model ( x_test )
            # The class with the max value is the one we are predicting
            _ , yhat = torch.max ( z.data , 1 )
            # Checks if the prediction matches the actual value
            correct += (yhat == y_test).sum ( ).item ( )

        # Calcualtes accuracy and saves it
        accuracy = correct / N_test
        accuracy_list.append ( accuracy )


train_model ( n_epochs )
# Plot the Loss and Accuracy vs Epoch graph

fig , ax1 = plt.subplots ( )
color = 'tab:red'
ax1.plot ( cost_list , color = color )
ax1.set_xlabel ( 'epoch' , color = color )
ax1.set_ylabel ( 'Cost' , color = color )
ax1.tick_params ( axis = 'y' , color = color )

ax2 = ax1.twinx ( )
color = 'tab:blue'
ax2.set_ylabel ( 'accuracy' , color = color )
ax2.set_xlabel ( 'epoch' , color = color )
ax2.plot ( accuracy_list , color = color )
ax2.tick_params ( axis = 'y' , color = color )
fig.tight_layout ( )
# Plot the channels

plot_channels(model.state_dict()['cnnl.weight'])
plot_channels(model.state_dict()['cnn2.weight'])
# Show the second image

# show_data(train_dataset[1])
# Use the CNN activations class to see the steps

out = model.activations(train_dataset[1][0].view(1, 1, IMAGE_SIZE, IMAGE_SIZE))

# Plot the outputs after the first CNN

plot_activations(out[0], number_rows=4, name="Output after the 1st CNN")
# Plot the outputs after the first Relu

plot_activations(out[1], number_rows=4, name="Output after the 1st Relu")
# Plot the outputs after the second CNN

plot_activations(out[2], number_rows=32 // 4, name="Output after the 2nd CNN")
# Plot the outputs after the second Relu

plot_activations(out[3], number_rows=4, name="Output after the 2nd Relu")
# Show the third image

# show_data(train_dataset[2])
# Use the CNN activations class to see the steps

out = model.activations(train_dataset[2][0].view(1, 1, IMAGE_SIZE, IMAGE_SIZE))
# Plot the outputs after the first CNN

plot_activations(out[0], number_rows=4, name="Output after the 1st CNN")
# Plot the outputs after the first Relu

plot_activations(out[1], number_rows=4, name="Output after the 1st Relu")
# Plot the outputs after the second CNN

plot_activations(out[2], number_rows=32 // 4, name="Output after the 2nd CNN")
# Plot the outputs after the second Relu

plot_activations(out[3], number_rows=4, name="Output after the 2nd Relu")
# Plot the misclassified samples

count = 0
for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
    z = model(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat: ",yhat)
        count += 1
    if count >= 5:
        break