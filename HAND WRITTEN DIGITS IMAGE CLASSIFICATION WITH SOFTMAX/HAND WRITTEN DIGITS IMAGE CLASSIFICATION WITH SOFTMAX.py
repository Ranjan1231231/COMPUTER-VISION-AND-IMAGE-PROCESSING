#HAND WRITTEN DIGITS IMAGE CLASSIFICATION WITH SOFTMAX
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Building the function to plot parameters
def PlotParameters(model):
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            # Set the label for the subplot.
            ax.set_xlabel("Class: {0}".format(i))
            # Plot the image
            ax.imshow(W[i, :].view(28, 28).cpu().numpy(), vmin=w_min, vmax=w_max, cmap="seismic")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

# Define a function to plot the data
def show_data(data_sample):
    plt.imshow(data_sample[0].squeeze().cpu().numpy(), cmap='gray')  # Squeeze the tensor and convert to numpy array
    plt.title('y = ' + str(data_sample[1]))

# Create and print the training dataset
train_dataset = dsets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
# print("Print the training dataset:\n", train_dataset)

# Create and print the validation dataset
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
# print("Print the validation dataset:\n", validation_dataset)

# Print the first image and label
# print("First Image and Label:")
show_data(train_dataset[0])

# Print out the label of the fourth element
# Print the label
# print("The label:", train_dataset[3][1])

# Plot the fourth sample
# print("The image:")
show_data(train_dataset[3])

# Plot the third image
show_data(train_dataset[2])

#BUILDING A SOFTMAX CASSIFIER
#DEFINIGN A SOFTMAX CLASSIFIER
class SoftMax(nn.Module):
    #constructor
    def __init__(self,input_size,output_size):
        super(SoftMax,self).__init__()
        #creates a layer of given input size and output size
        self.linear=nn.Linear(input_size,output_size)
    #Prediction
    def forward( self,x ):
        #runs the x value through the single layers defined above
        z=self.linear(x)
        return z
#Print the shapeof the training dataset
# print(train_dataset[0][0].shape)
#SET INPUT SIZE AND OUTPUT SIZE
input_dim=28*28
output_dim=10
#Define the softmax xlassifier , criterion function,optimizer and train the model
model=SoftMax(input_dim,output_dim)
# print("Print the model:\n",model)
#Print the parameters
# print('W:',list(model.parameters())[0].size())
# print('b: ', list(model.parameters())[1].size())

# PlotParameters(model)

#MAKING THE PREDICTIONS
#FIRST WE GGET THE X VALUE OF THE FIRST IMAGE
X=train_dataset[0][0]
#we can see the shape is 1 by 28 by 28,we need it to be fatenned to 1 by 28 * 28(784)
# print(X.shape)
X=X.view(-1,28*28)
# print(X.shape)
#now we can make a prediction, each class has a value, and the higher it is the more confident the model is that it is that digit
model(X)
#define the lerning rate,optimizer,criterion and data loader
learning_rate=0.1
#the optimizer will updates the model parameter using the learning rate
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
# The criterion will measure the loss between the prediction and actual label values
# This is where the SoftMax occurs, it is built into the Criterion Cross Entropy Loss
criterion=nn.CrossEntropyLoss()
#created a training data loaderso we can set the batch size
train_load=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100)
#created a validation data loader so we can set the batch size
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=5000)

model_output=model(X)
actual=torch.tensor([train_dataset[0][1]])
show_data(train_dataset[0])
# print("Output :",model_output)
# print("Actual :",actual)

criterion(model_output,actual)
softmax = nn.Softmax(dim=1)
probability = softmax(model_output)
print(probability)
print(-1*torch.log(probability[0][actual]))
#train
# Number of times we train our model useing the training data
n_epochs = 10
# Lists to keep track of loss and accuracy
loss_list = [ ]
accuracy_list = [ ]
# Size of the validation data
N_test = len ( validation_dataset )


# Function to train the model based on number of epochs
def train_model ( n_epochs ) :
    # Loops n_epochs times
    for epoch in range ( n_epochs ) :
        # For each batch in the train loader
        for x , y in train_load :
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad ( )
            # Makes a prediction based on the image tensor
            z = model ( x.view ( -1 , 28 * 28 ) )
            # Calculates loss between the model output and actual class
            loss = criterion ( z , y )
            # Calculates the gradient value with respect to each weight and bias
            loss.backward ( )
            # Updates the weight and bias according to calculated gradient value
            optimizer.step ( )

        # Each epoch we check how the model performs with data it has not seen which is the validation data, we are not training here
        correct = 0
        # For each batch in the validation loader
        for x_test , y_test in validation_loader :
            # Makes prediction based on image tensor
            z = model ( x_test.view ( -1 , 28 * 28 ) )
            # Finds the class with the higest output
            _ , yhat = torch.max ( z.data , 1 )
            # Checks if the prediction matches the actual class and increments correct if it does
            correct += (yhat == y_test).sum ( ).item ( )
        # Calculates the accuracy by dividing correct by size of validation dataset
        accuracy = correct / N_test
        # Keeps track loss
        loss_list.append ( loss.data )
        # Keeps track of the accuracy
        accuracy_list.append ( accuracy )


# Function call
train_model ( n_epochs )

#ANALYZE RESULTS
# Plot the loss and accuracy

fig , ax1 = plt.subplots ( )
color = 'tab:red'
ax1.plot ( loss_list , color = color )
ax1.set_xlabel ( 'epoch' , color = color )
ax1.set_ylabel ( 'total loss' , color = color )
ax1.tick_params ( axis = 'y' , color = color )

ax2 = ax1.twinx ( )
color = 'tab:blue'
ax2.set_ylabel ( 'accuracy' , color = color )
ax2.plot ( accuracy_list , color = color )
ax2.tick_params ( axis = 'y' , color = color )
fig.tight_layout ( )
# Plot the parameters

PlotParameters(model)
# Plot the misclassified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break
# Plot the classified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat == y:
        show_data((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break