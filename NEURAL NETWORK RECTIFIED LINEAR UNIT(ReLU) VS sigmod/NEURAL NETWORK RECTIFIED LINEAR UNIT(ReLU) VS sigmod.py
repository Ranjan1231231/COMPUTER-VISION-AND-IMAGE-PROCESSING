#NEURAL NETWORK RECTIFIED LINEAR UNIT(ReLU) VS sigmod
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import  matplotlib.pylab as plt
import numpy as np
# Setting the seed will allow us to control randomness and give us reproducibility
torch.manual_seed(2)

#defining the neural network module or class with two hidden layers
#create the model class using sigmoid as the activation function
class Net(nn.Module):
    #constructior
    def __init__(self,D_in,H1,H2,D_out):
        #D_in is the input size of the first layer (size of the input layer)
        #H1 IS THE OUTPUT SIZE OF THE FIRST LAYER AND INPUT SIZE (SIZE OF THE FIRST HIDDEN LAYER)
        #H2 IS THE OUTPUT SIZE OF THE SECOND LAYER AND THE INPUT SIZE OF THE THIRD LAYER(SIZE OF THE SECOND HIDDEN LAYER)
        #D_OUT IS THE OUTPUT SIZE OF THE THIRD LAYER (SIZE OF THE OUTPUT LAYER)
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H1)
        self.linear2=nn.Linear(H1,H2)
        self.linear3=nn.Linear(H2,D_out)
    #PREDICTION
    def forward( self,x ):
        #puts x through the first layers then the sigmoid function
        x=torch.sigmoid(self.linear1(x))
        #puts result of the previous line through second layer then sigmoid function
        x=torch.sigmoid(self.linear2(x))
        #puts the result of the previous line hrough third layer
        x=self.linear3(x)
        return x

#CREATE THE MODEL CLASS USING THE RELU AS THE ACTIVATION FUNCTION
class NetRelu(nn.Module):
    #constructor
    def __init__( self ,D_in,H1,H2,D_out):
        #D_in is the input size of the first layer(size of input layer)
        #H1 is the output size of the first layer and input size of the second layer(Size of first hidden layer)
        #H2 is the output size of the second layer and the inpput size of the third layer(size of the secon hidden layer)
        #d_out is the output size of the third layer(size of output layer)
        super(NetRelu, self).__init__()
        self.linear1=nn.Linear(D_in,H1)
        self.linear2=nn.Linear(H1,H2)
        self.linear3=nn.Linear(H2,D_out)

    #prediction
    def forward( self,x ):
        #puts x through the first layers the the  relu function
        x=torch.relu(self.linear1(x))
        #put results of the previous line through second layer then relu function
        x=torch.relu(self.linear2(x))
        #puts result of the previous layer through third layer
        x=self.linear3(x)
        return x
# Define a function to  train the model, in this case, the function returns a Python dictionary to store the training loss and accuracy on the validation data
#MODEL TRAINING FUNCTION

def  train(model,criterion,train_loader,validation_loader,optimizer,epochs=100):
    i=0
    useful_stuff={"training_loss":[],'validation_accuracy':[]}
    #number of times we train on the entire dataset
    for epoch in range(epochs):
        #for each batch in the train loader
        for i,(x,y) in enumerate(train_loader):
            #resets the calculated gradient value, this must be done each time as it accumulates if we do not resets
            optimizer.zero_grad()
            #makes a predictuin on the image tensor by flattening it to a 1 by28*28 tensor
            z=model(x.view(-1,28*28))
            #calculates the loss between the prediction and actual class
            loss=criterion(z,y)
            #calculates the gradient value with respect to each weight and bias
            loss.backward()
            #updates the weight and bias ccording to calculated gradient value
            optimizer.step()
            #saves the loss
            useful_stuff['training_loss'].append(loss.data.item())
        #counter to keep track of correct predictions
        correct=0
        #for each batch in the validation dataset
        for x, y in validation_loader:
            #make a prediction
            z=model(x.view(-1,28*28))
            #get the class that has the maximum value
            _,label=torch.max(z,1)
            #check if our prediction matches the actual class
            correct+=(label==y).sum().item()

        #saves the percent accuracy
        accuracy=100*(correct/len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff

#MAKING SOME DATA
#creating the training dataset
# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform object int the argument transform
train_dataset=dsets.MNIST(root = './data',train = True,download = True,transform = transforms.ToTensor())

#Load the testing dataset by setting the parameters train to False and convert it to a tensor by placing a transform object int the argument transform
#create the validating dataset
validation_dataset=dsets.MNIST(root = './data',train = False,download = True,transform = transforms.ToTensor())

#creating the criterion function
criterion=nn.CrossEntropyLoss()
#create the training data loader and validation dataloader object
#Batch size is 2000 and shuffle=True means the data will be shuffeled at every epoch
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=2000,shuffle=True)
#Batch size is 5000 and the data wil not be shuffled every epoch
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=5000,shuffle=False)

#DEFINE NEURAL NETWORK , CRITERION FUNCTION,OPTIMIZEER AND TRAIN THE MODEL
#CREATING THE MODEL WITH 100 HIDDEN NEURONS
#set the parameters to create the model
input_dim=28*28#dimension of an image
hidden_dim1=50
hidden_dim2=50
output_dim=10#number of classes
#set the number of iteration
cust_epochs=10


#TEST SIGMOID AND RELU
#TRAINING THE MODEL WITH SIGMOID FUNCTION
learning_rate=0.01
#create an instance of the net model
model=Net(input_dim,hidden_dim1,hidden_dim2,output_dim)
#create an optimizer that updates model parameters using the learning rate and gradient
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
#train the model
training_results=train(model,criterion,train_loader,validation_loader,optimizer,epochs = cust_epochs)

#TRANING THE MODEL WITH RELU FUNCTION
learning_rate=0.01
#create an instance of the net relu model
modelRelu=NetRelu(input_dim,hidden_dim1,hidden_dim2,output_dim)
#create an optimizer that updates model paraeters using the learning rate and gradient
optimizer=torch.optim.SGD(modelRelu.parameters(),lr=learning_rate)
#train the model
training_results_relu=train(model,criterion,train_loader,validation_loader,optimizer,epochs = cust_epochs)


#ANALYZING THE RESULTS
#compairing the training loss
plt.plot(training_results['training_loss'],label='sigmoid')
plt.plot(training_results_relu['training_loss'],label='relu')
plt.ylabel('loss')
plt.title("training loss iterations")
plt.legend()
plt.show()
#compairing the validation loss
plt.plot(training_results['validation_accuracy'],label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'],label='relu')
plt.ylabel('validation accuracy')
plt.xlabel('iteration')
plt.legend()
plt.show()